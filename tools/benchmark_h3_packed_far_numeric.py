"""Strict two-bank packed far PASS=1 arithmetic parity gate; ONE JSON.

This is an opt-in research diagnostic, not production or a fast attention
implementation. On ANY PM/PL/PACC bit mismatch, speed stays UNRUN.
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
from analyze_h3_far_pack import reference_pack
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


def _pack_indices(torch, states, pack_kernel, reference):
    packed = torch.empty((HEADS,TILES,SPLITS,LPS),device=states["EXCL"].device,
                         dtype=torch.int32)
    counts = torch.empty((HEADS,TILES,SPLITS),device=packed.device,dtype=torch.int32)
    pack_kernel[(HEADS*TILES,SPLITS)](
        states["EXCL"],packed,counts,NB=NB,SPLITS=SPLITS,LPS=LPS,
        num_warps=4,num_stages=1)
    torch.cuda.synchronize(packed.device)
    want_counts = [[[len(split) for split in row] for row in head]
                   for head in reference["indices"]]
    expected_counts = torch.tensor(want_counts,dtype=torch.int32,device=counts.device)
    if not torch.equal(counts,expected_counts):
        raise AssertionError("Packed split COUNTS do not match CPU oracle")
    valid = torch.arange(LPS,device=packed.device)[None,None,None,:] < counts[...,None]
    # GPU boolean indexing selects ONLY valid elements. Never read the
    # intentionally uninitialized PACKED suffix, including on the host.
    actual_valid = packed[valid].detach().cpu().tolist()
    reference_valid = [idx for head in reference["indices"]
                       for row in head for split in row for idx in split]
    if actual_valid != reference_valid:
        raise AssertionError("Ordered packed leaf prefix differs from CPU oracle")
    if len(actual_valid) != 56576 or reference["packed_indices"] != 56576:
        raise AssertionError("Unexpected number of eligible far indices")
    return packed,counts


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


def _launch_packed_far(states, packed, counts, kernel, output):
    q = states["Q"]
    strides = tuple(map(int,q.stride()))
    return kernel[(HEADS*TILES,SPLITS)](
        q, states["MU"].view(HEADS*TILES,D),
        states["G"].view(HEADS*TILES,NB,D),
        states["LZ"].view(HEADS*TILES,NB),
        states["VB"].view(HEADS*TILES,NB,D),
        states["SG"].view(HEADS*TILES,NB),
        packed,counts,
        output["pm"].view(HEADS*TILES,SPLITS,ROWS),
        output["pl"].view(HEADS*TILES,SPLITS,ROWS),
        output["acc"].view(HEADS*TILES,SPLITS,ROWS,D),
        NB=NB,SPLITS=SPLITS,LPS=LPS,TILES=TILES,
        T=ROWS,D=D,BN=BN,Q_DIRECT=1,
        Q_SH=strides[0],Q_ST=strides[1],
        Q_SR=strides[2],Q_SD=strides[3],
        num_warps=4,num_stages=2)


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
    # All imports of torch, Triton and model runtime occur AFTER exact-HEAD gate.
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from htrc_far_pack_triton import pack_ordered_far_index
    from htrc_packed_far_numeric_triton import packed_far_partial
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
    shape = (1,HEADS,TOKENS,D)
    banks = [tuple(torch.randn(shape,device=device,dtype=torch.bfloat16)
                   for _ in range(3)) for _ in range(2)]
    old = {key:os.environ.get(key) for key in ENV_KEYS}
    report = {"banks":[],"far_partial_bit_exact":False,"speed_capture_blocks":[]}
    complete_outputs=[]
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-packed-far-numeric-") as tmp:
            env = {
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
                output = flow.graph_attention_bthd(
                    *bank,heads=HEADS,mode="balanced",tiles=TILES,
                    budget_mib=1024,reserve_mib=256,workspace=True)
                torch.cuda.synchronize(device)
                if bi == 0:
                    gate = json.loads((Path(tmp)/"first_call.json").read_text(encoding="utf-8"))
                    if gate.get("status") != "PASS_FIRST_REAL_CALL" or gate.get("all_groups_bit_exact") is not True:
                        raise AssertionError("Original first-call full Graph/eager parity failed")
                if tuple(output.shape) != (1,TOKENS,HEADS,D):
                    raise AssertionError("Unexpected original full attention shape")
                complete_outputs.append(output.detach().cpu().clone())
                item = flow._SESSIONS.get((HEADS,),{}).get("graphs",{}).get((TILES,2048))
                if item is None:
                    raise RuntimeError("Missing pinned Graph materialized one-window states")
                states=item["states"]
                _assert_pinned(torch,states,cfg)
                ref=reference_pack(states["EXCL"].detach().cpu().tolist(),
                                   states["IDX"].detach().cpu().tolist(),
                                   splits=SPLITS,bn=BN,expected_near=KX)
                fingerprint=ref["selection"]["selection_mask_sha256"]
                if fingerprint != EXPECTED_MASKS[bi]:
                    raise AssertionError(f"Bank {bi} selection mask changed: {fingerprint}")
                fingerprints.append(fingerprint)
                packed,counts=_pack_indices(torch,states,pack_ordered_far_index,ref)
                base=_alloc_partials(torch,device)
                cand=_alloc_partials(torch,device)
                control_kernel=_launch_original_far(states,tr,base)
                candidate_kernel=_launch_packed_far(states,packed,counts,
                                                    packed_far_partial,cand)
                torch.cuda.synchronize(device)
                parity=compare_far_partials(torch,base,cand)
                report["banks"].append({
                    "bank":bi,"selection_mask_sha256":fingerprint,
                    "original_split_lps":LPS,"source_far_blocks":1920,
                    "packed_far_blocks_lower_bound":ref["packed_blocks"],
                    "packed_indices_valid":ref["packed_indices"],
                    "far_only_partial_parity":parity,
                    "far_partial_all_bit_exact":all(x["bit_exact"] for x in parity.values()),
                    "compiled":{
                        "original_pass1":_compiled_resources(control_kernel),
                        "packed_pass1":_compiled_resources(candidate_kernel),
                    },
                })
                del packed,counts,base,cand,output
            if (torch.equal(complete_outputs[0],complete_outputs[1]) or
                fingerprints[0] == fingerprints[1]):
                raise AssertionError("Two QKV banks are not independent")
            eager_b=live.flat_attention_bthd(
                *banks[1],heads=HEADS,mode="balanced",tiles=TILES,
                budget_mib=1024,reserve_mib=256,workspace=True,
                query_staging="native_torch_mu",kv_staging="native_direct",
                dispatch="eager")
            if not torch.equal(eager_b.cpu(),complete_outputs[1]):
                raise AssertionError("Bank B original complete eager/Graph parity failed")
            report.update({
                "two_distinct_bf16_qkv_banks":True,
                "first_complete_graph_eager_bit_exact":True,
                "bank_b_complete_graph_eager_bit_exact":True,
                "different_selection_masks_per_bank":True,
                "far_partial_bit_exact":all(b["far_partial_all_bit_exact"] for b in report["banks"]),
                "speed_measurement":"UNRUN_NUMERIC_GATE_ONLY",
                "no_cuda_event_timing":True,
            })
            return report
    finally:
        try: flow.clear_session_cache()
        finally:
            for key,value in old.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-head",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report={
        "schema":"htrc-packed-far-numeric-v1",
        "status":"FAIL_CLOSED","completed":False,
        "source_head":None,
        "probe":"ORIGINAL_PASS1_VS_ORDERED_PACKED_FAR_QK_PV",
        "tokens":TOKENS,"heads":HEADS,"tiles":TILES,
        "exact_leaves":KX,"splits":SPLITS,"far_bn":BN,
        "timing_scope":"NUMERIC_ONLY_NO_ATTENTION_TIMING",
        "speed_measurement":"UNRUN_NUMERIC_GATE_ONLY",
        "h3_model_or_video":"UNRUN",
        "production_promotion":"PROHIBITED_RESEARCH_ONLY",
        "full_packed_attention":"UNRUN",
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
        if report["far_partial_bit_exact"]:
            report["status"]="PASS_FAR_NUMERIC_ONLY"
            report["completed"]=True
            report["parity_admission"]="PASS_FAR_ONLY_NEXT_FULL_GATE_REQUIRED"
        else:
            # Expected scientific negative: record BOTH bank diagnostics and
            # STOP before any timing or false accuracy/performance claims.
            report["status"]="FAIL_CLOSED"
            report["completed"]=False
            report["parity_admission"]="NO_GO_FAR_BIT_EXACT"
            report["error"]="Packed leaf grouping changed far PASS=1 PM/PL/PACC bits"
    except BaseException as exc:
        report["status"]="FAIL_CLOSED"
        report["completed"]=False
        report["error"]=f"{type(exc).__name__}: {exc}"[:1800]
        report["traceback_tail"]=traceback.format_exc()[-4500:]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"one_json={args.output} status={report['status']} parity={report.get('parity_admission','UNRUN')}")
    return 0 if report["status"] == "PASS_FAR_NUMERIC_ONLY" else 2


if __name__=="__main__":
    raise SystemExit(main())
