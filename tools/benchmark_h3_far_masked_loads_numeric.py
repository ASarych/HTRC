"""Strict two-bank masked-G/VB far PASS=1 parity; NO candidate timing.

Research-only clone retains original BN64 split/score/PV grouping. First
require unmasked clone == original; then examine masked G, masked VB, both.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from analyze_h3_far_exclusion import summarize_exclusions
from profile_h3_far_exclusion import ENV_KEYS

HEADS = 4
TILES = 32
ROWS = 64
D = 128
TOKENS = 15275
NB = 954
KX = 512
SPLITS = 4
LPS = 256
BN = 64
EXPECTED_MASKS = (
    "bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32",
    "05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264",
)


def _alloc_partials(torch, device):
    shape = (HEADS, TILES, SPLITS, ROWS)
    return {"pm": torch.empty(shape, device=device, dtype=torch.float32),
            "pl": torch.empty(shape, device=device, dtype=torch.float32),
            "acc": torch.empty(shape + (D,), device=device, dtype=torch.float32)}


def _assert_pinned(torch, states, cfg):
    if (tuple(states["Q"].shape) != (HEADS,TILES,ROWS,D) or
        tuple(states["MU"].shape) != (HEADS,TILES,D) or
        tuple(states["G"].shape) != (HEADS,TILES,NB,D) or
        tuple(states["VB"].shape) != (HEADS,TILES,NB,D) or
        tuple(states["EXCL"].shape) != (HEADS,TILES,NB) or
        tuple(states["IDX"].shape) != (HEADS,TILES,KX) or
        tuple(states["LZ"].shape) != (HEADS,TILES,NB) or
        tuple(states["SG"].shape) != (HEADS,TILES,NB)):
        raise AssertionError("Pinned 4-head/32-tile/954-leaf state geometry changed")
    if (states["Q"].dtype != torch.bfloat16 or
        states["G"].dtype != torch.int8 or
        states["VB"].dtype != torch.bfloat16 or
        states["SG"].dtype != torch.float32 or
        states["EXCL"].dtype != torch.int8 or
        states["IDX"].dtype != torch.int32):
        raise AssertionError("Pinned Q/leaf/selection dtype changed")
    if any(not states[name].is_contiguous() for name in
           ("MU","G","VB","LZ","SG","EXCL","IDX")):
        raise AssertionError("Pinned materialized leaf state is not contiguous")
    kern = cfg.get("kern",{})
    if (int(cfg.get("tile",-1)),int(cfg.get("splits",-1)),
        int(cfg.get("exact_leaves",-1)),int(kern.get("group",1)),
        int(kern.get("tile_bn",-1)),int(kern.get("tile_warps",-1)),
        int(kern.get("tile_stages",-1))) != (64,4,512,1,64,4,2):
        raise AssertionError("Pinned original Tile32 materialized config changed")
    if not states.get("_direct_q",False) or states.get("_vb_mode") != "materialized":
        raise AssertionError("Expected native-direct Q, materialized far Vbar")


def _launch_original_far(states, tr, output):
    """Use the EXACT original _tile_kernel with PASS=1 (far only)."""
    q = states["Q"]
    qstrides = tuple(map(int,q.stride()))
    keys = int(states["K"].shape[1])
    lz = states["LZ"].view(HEADS*TILES,NB)
    return tr._tile_kernel[(HEADS*TILES,SPLITS,1)](
        q, states["MU"].view(HEADS*TILES,D),
        states["G"].view(HEADS*TILES,NB,D),lz,
        states["VB"].view(HEADS*TILES,NB,D),
        states["EXCL"].view(HEADS*TILES,NB),
        states["IDX"].view(HEADS*TILES,KX),
        states["K"].view(HEADS*keys,D),
        states["V"],
        output["pm"].view(HEADS*TILES,SPLITS,ROWS),
        output["pl"].view(HEADS*TILES,SPLITS,ROWS),
        output["acc"].view(HEADS*TILES,SPLITS,ROWS,D),
        states["SG"].view(HEADS*TILES,NB), lz,lz,lz,
        states["KS"].view(HEADS*keys),
        NB,KX,LPS,2048,0,1.0/math.sqrt(D),
        T=ROWS,D=D,BN=BN,BN_NEAR=BN,LEAF=16,
        PREC="tf32",DOT=1,GR=1,FP8=0,TAIL=0,RB=1,I8=1,
        PASS=1,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0,
        FAR_PV_ACC=0,NEAR_PV_ACC=0,
        BATCH_TILES_PER_HEAD=TILES,BATCH_KEY_ROWS=keys,
        Q_DIRECT=1,Q_SH=qstrides[0],Q_ST=qstrides[1],
        Q_SR=qstrides[2],Q_SD=qstrides[3],
        V_DIRECT=int(bool(states.get("_direct_v",False))),
        V_SH=0,V_SS=0,V_SD=0,
        num_warps=4,num_stages=2)


def _launch_research_far(states, kernel, output, *, mask_g: bool, mask_vb: bool):
    q = states["Q"]
    st = tuple(map(int, q.stride()))
    return kernel[(HEADS*TILES, SPLITS)](
        q, states["MU"].view(HEADS*TILES,D),
        states["G"].view(HEADS*TILES,NB,D),
        states["LZ"].view(HEADS*TILES,NB),
        states["VB"].view(HEADS*TILES,NB,D),
        states["EXCL"].view(HEADS*TILES,NB),
        states["SG"].view(HEADS*TILES,NB),
        output["pm"].view(HEADS*TILES,SPLITS,ROWS),
        output["pl"].view(HEADS*TILES,SPLITS,ROWS),
        output["acc"].view(HEADS*TILES,SPLITS,ROWS,D),
        NB=NB,LPS=LPS,TILES=TILES,T=ROWS,D=D,BN=BN,
        MASK_G=int(mask_g),MASK_VB=int(mask_vb),Q_DIRECT=1,
        Q_SH=st[0],Q_ST=st[1],Q_SR=st[2],Q_SD=st[3],
        num_warps=4,num_stages=2)


VARIANTS = {
    "mirror": (False, False),
    "mask_g": (True, False),
    "mask_vb": (False, True),
    "mask_both": (True, True),
}


def _compiled_resources(kernel):
    metadata = getattr(kernel,"n_regs",None)
    shared = getattr(kernel,"metadata",None)
    return {"registers_per_thread":int(metadata) if metadata is not None else None,
            "shared_bytes_per_cta":int(getattr(shared,"shared",0)) if shared else None,
            "compiler_spills":int(getattr(kernel,"n_spills",0)) if getattr(kernel,"n_spills",None) is not None else None}


def compare_far_partials(torch, control, candidate):
    """Compare raw float32 representations, not numeric torch.equal(-0,+0)."""
    report = {}
    for name in ("pm","pl","acc"):
        a,b = control[name],candidate[name]
        if a.dtype != torch.float32 or a.shape != b.shape or a.device != b.device:
            raise AssertionError(f"Bad partial {name} shape, dtype or device")
        diff = a.view(torch.int32) != b.view(torch.int32)
        count = int(diff.count_nonzero().item())
        first = torch.nonzero(diff,as_tuple=False)
        first_index = [int(v) for v in first[0].cpu().tolist()] if count else None
        finite = torch.isfinite(a) & torch.isfinite(b)
        if bool(finite.any().item()):
            max_abs = float((a[finite]-b[finite]).abs().max().item())
        else:
            max_abs = None
        report[name] = {"bit_exact":count == 0,
                        "bit_mismatch_count":count,
                        "total_elements":a.numel(),
                        "first_mismatch_index":first_index,
                        "max_finite_abs_difference":max_abs}
    return report


def run_gpu():
    # Do not import GPU dependencies unless the exact HEAD check passed.
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from htrc_far_masked_loads_triton import far_masked_load_partial
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live
    from h3 import htrc_flat_model_shadow as shared

    if (not torch.cuda.is_available() or
        torch.cuda.get_device_capability(0) != (12,0) or
        "RTX 5070 Ti" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Exact RTX 5070 Ti SM120 required")
    torch.set_grad_enabled(False)
    torch.manual_seed(42026)
    device = torch.device("cuda:0")
    shape=(1,HEADS,TOKENS,D)
    banks=[tuple(torch.randn(shape,device=device,dtype=torch.bfloat16)
                 for _ in range(3)) for _ in range(2)]
    old={key:os.environ.get(key) for key in ENV_KEYS}
    report={"banks":[],"speed_capture_blocks":[]}
    complete_outputs=[]
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-masked-far-numeric-") as tmp:
            env={
                "APL_HTRC_TILE32_STATS_BT":"16",
                "APL_HTRC_DIRECTFLOW_GROUP":"11",
                "APL_HTRC_DIRECTFLOW_WARMUPS":"4",
                "APL_HTRC_DIRECTFLOW_VB_MODE":"materialized",
                "APL_HTRC_DIRECTFLOW_GROUP_POLICY":"legacy",
                "APL_HTRC_DIRECTFLOW_REPORT":str(Path(tmp)/"first_call.json"),
                "APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP":None,
                "APL_HTRC_GRAPH_CAPTURE_SAFE_MU":"0",
                "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB":"768",
                "APL_HTRC_DIRECTFLOW_MIN_FREE_MIB":"256",
                "APL_HTRC_DIRECTFLOW_TILES32_ADMISSION":"0",
            }
            for key,value in env.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value
            flow.clear_session_cache()
            cfg=shared.flat_config("balanced",TILES)
            tr=live._triton()
            fingerprints=[]
            for bi,bank in enumerate(banks):
                output=flow.graph_attention_bthd(
                    *bank,heads=HEADS,mode="balanced",tiles=TILES,
                    budget_mib=1024,reserve_mib=256,workspace=True)
                torch.cuda.synchronize(device)
                if bi == 0:
                    gate=json.loads((Path(tmp)/"first_call.json").read_text(encoding="utf-8"))
                    if gate.get("status") != "PASS_FIRST_REAL_CALL" or gate.get("all_groups_bit_exact") is not True:
                        raise AssertionError("Original first-call full Graph/eager parity failed")
                if tuple(output.shape) != (1,TOKENS,HEADS,D):
                    raise AssertionError("Unexpected full Graph attention output shape")
                complete_outputs.append(output.detach().cpu().clone())
                item=flow._SESSIONS.get((HEADS,),{}).get("graphs",{}).get((TILES,2048))
                if item is None:
                    raise AssertionError("Pinned materialized full Tile32 window missing")
                states=item["states"]
                _assert_pinned(torch,states,cfg)
                selection=summarize_exclusions(
                    states["EXCL"].detach().cpu().tolist(),
                    states["IDX"].detach().cpu().tolist(),
                    splits=SPLITS,bn=BN,expected_near=KX)
                fingerprint=selection["selection_mask_sha256"]
                if fingerprint != EXPECTED_MASKS[bi]:
                    raise AssertionError(f"Selection mask changed for bank{bi}: {fingerprint}")
                if (selection["current_far_dot_blocks"] != 1920 or
                    selection["fully_excluded_blocks"] != 0 or
                    selection["eligible_far_leaves_all_centers"] != 56576):
                    raise AssertionError("Pinned far selection geometry changed")
                fingerprints.append(fingerprint)
                control=_alloc_partials(torch,device)
                compiled={"original_pass1":_compiled_resources(_launch_original_far(states,tr,control))}
                differences={}
                for name,flags in VARIANTS.items():
                    candidate=_alloc_partials(torch,device)
                    compiled[name]=_compiled_resources(_launch_research_far(
                        states,far_masked_load_partial,candidate,
                        mask_g=flags[0],mask_vb=flags[1]))
                    torch.cuda.synchronize(device)
                    parity=compare_far_partials(torch,control,candidate)
                    differences[name]={
                        "pm_pl_pacc":parity,
                        "all_bit_exact":all(p["bit_exact"] for p in parity.values()),
                    }
                    del candidate
                report["banks"].append({
                    "bank":bi,"selection_mask_sha256":fingerprint,
                    "far_original_bn64_blocks":1920,
                    "candidate_group_boundaries_unchanged":True,
                    "variant_parity":differences,
                    "compiled":compiled,
                })
                del control,output
            if torch.equal(complete_outputs[0],complete_outputs[1]) or fingerprints[0]==fingerprints[1]:
                raise AssertionError("QKV banks are not independent")
            eager_b=live.flat_attention_bthd(
                *banks[1],heads=HEADS,mode="balanced",tiles=TILES,
                budget_mib=1024,reserve_mib=256,workspace=True,
                query_staging="native_torch_mu",kv_staging="native_direct",
                dispatch="eager")
            if not torch.equal(eager_b.cpu(),complete_outputs[1]):
                raise AssertionError("Bank B complete Graph/eager parity failed")
            mirror=all(row["variant_parity"]["mirror"]["all_bit_exact"] for row in report["banks"])
            admitted=[name for name in ("mask_g","mask_vb","mask_both")
                      if all(row["variant_parity"][name]["all_bit_exact"] for row in report["banks"])]
            report.update({
                "two_distinct_bf16_qkv_banks":True,
                "different_selection_masks_per_bank":True,
                "first_complete_graph_eager_bit_exact":True,
                "bank_b_complete_graph_eager_bit_exact":True,
                "research_dense_mirror_bit_exact":mirror,
                "admitted_far_only_variants":admitted if mirror else [],
                "speed_measurement":"UNRUN_NUMERIC_GATE_ONLY",
                "no_cuda_event_timing":True,
                "full_masked_attention":"UNRUN",
            })
            return report
    finally:
        try: flow.clear_session_cache()
        finally:
            for key,value in old.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-head",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report={
        "schema":"htrc-far-masked-loads-numeric-v1",
        "status":"FAIL_CLOSED","completed":False,
        "source_head":None,
        "probe":"ORIGINAL_PASS1_VS_UNCHANGED_BN64_MASKED_G_VB_LOADS",
        "tokens":TOKENS,"heads":HEADS,"tiles":TILES,
        "exact_leaves":KX,"splits":SPLITS,"far_bn":BN,
        "timing_scope":"NUMERIC_ONLY_NO_ATTENTION_TIMING",
        "speed_measurement":"UNRUN_NUMERIC_GATE_ONLY",
        "h3_model_or_video":"UNRUN",
        "production_promotion":"PROHIBITED_RESEARCH_ONLY",
        "full_masked_attention":"UNRUN",
        "no_cuda_event_timing":True,
        "speed_capture_blocks":[],
    }
    try:
        actual=subprocess.check_output(
            ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
        if actual!=args.expect_head:
            raise RuntimeError(f"Exact HEAD mismatch {actual} != {args.expect_head}")
        report["source_head"]=actual
        result=run_gpu()
        report.update(result)
        if not report["research_dense_mirror_bit_exact"]:
            report["status"]="FAIL_CLOSED"
            report["parity_admission"]="NO_GO_RESEARCH_DENSE_MIRROR"
            report["error"]="Research clone does not match original PASS1; candidates not admissible"
        elif "mask_both" in report["admitted_far_only_variants"]:
            report["status"]="PASS_MASKED_FAR_NUMERIC_ONLY"
            report["completed"]=True
            report["parity_admission"]="PASS_BOTH_MASKED_FAR_ONLY"
        elif report["admitted_far_only_variants"]:
            report["status"]="PASS_MASKED_FAR_NUMERIC_ONLY"
            report["completed"]=True
            report["parity_admission"]="PASS_SUBSET_MASKED_FAR_ONLY"
        else:
            report["status"]="FAIL_CLOSED"
            report["parity_admission"]="NO_GO_MASKED_FAR_BIT_EXACT"
            report["error"]="Masked G/VB far-only PM/PL/PACC differs from baseline"
    except BaseException as exc:
        report["status"]="FAIL_CLOSED"
        report["completed"]=False
        report["error"]=f"{type(exc).__name__}: {exc}"[:1800]
        report["traceback_tail"]=traceback.format_exc()[-4500:]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"one_json={args.output} status={report['status']} parity={report.get('parity_admission','UNRUN')}")
    return 0 if report["status"]=="PASS_MASKED_FAR_NUMERIC_ONLY" else 2


if __name__=="__main__":
    raise SystemExit(main())
