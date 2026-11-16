"""Pinned numerical gate for research K8-only near reindexing.

One JSON. Real Graph-vs-eager exactness on two independent BF16 banks;
then baseline PASS=0 PM/PL/PACC and merged output vs K8+scale packed
near operand with original unsorted V addresses. Candidate speed UNRUN.
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
from benchmark_h3_far_masked_loads_numeric import (
    _alloc_partials, _assert_pinned, compare_far_partials,
    _compiled_resources, EXPECTED_MASKS,
    HEADS, TILES, ROWS, D, TOKENS, NB, KX, SPLITS, LPS, BN,
)

NEAR_TOKENS = KX * 16
NEAR_SCRATCH_BYTES = HEADS * TILES * NEAR_TOKENS * (D + 4)
MAX_HEAD_GROUP_SCRATCH_MIB = 132


def geometry_budget():
    n=HEADS*TILES*NEAR_TOKENS
    return {"near_tokens_per_centre": NEAR_TOKENS,
            "pseudo_centres": HEADS*TILES,
            "packed_k8_bytes": n*D,
            "packed_scale_fp32_bytes": n*4,
            "packed_total_bytes": NEAR_SCRATCH_BYTES,
            "packed_total_mib": NEAR_SCRATCH_BYTES/1048576,
            "measured_original_56head_graph_extra_mib": 590,
            "optimistic_full_graph_plus_pack_mib": 590+NEAR_SCRATCH_BYTES/1048576,
            "graph_extra_gate_mib": 768,
            "memory_admission_for_56head_model": "UNRUN_NOT_ISOLATED"}


def _compare_bits(torch, a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise AssertionError("Parity tensors differ in shape/dtype/device")
    mismatch = a.view(torch.int32)!=b.view(torch.int32)
    errors = int(mismatch.count_nonzero().item())
    first = torch.nonzero(mismatch, as_tuple=False)
    at = [int(x) for x in first[0].cpu().tolist()] if errors else None
    finite = torch.isfinite(a)&torch.isfinite(b)
    maxabs = float((a[finite]-b[finite]).abs().max().item()) if bool(finite.any().item()) else None
    return {"bit_exact": errors==0,"bit_mismatch_count":errors,
            "total_elements":a.numel(),"first_mismatch_index":at,
            "max_finite_abs_difference":maxabs}


def _merge(torch, batch, states, cfg, tr, partial, tail):
    out = torch.empty((HEADS,TILES,ROWS,D),device=states["Q"].device,
                      dtype=torch.float32)
    merged, _ = batch.launch_prepared_merge_batch(
        states,cfg,tr,partial,tails=tail,output=out,merge_rows=32)
    return merged


def _pack(torch, states, kernel):
    if (tuple(states["K"].shape)!=(HEADS,15264,D)
        or tuple(states["KS"].shape)!=(HEADS,15264)
        or tuple(states["IDX"].shape)!=(HEADS,TILES,KX)
        or states["K"].dtype!=torch.int8
        or states["KS"].dtype!=torch.float32):
        raise AssertionError("Unexpected native source K8/KS/IDX")
    device = states["K"].device
    free,_ = torch.cuda.mem_get_info(device)
    if free < NEAR_SCRATCH_BYTES + 256*1048576:
        raise MemoryError("Numeric K-only near scratch requires free VRAM +256MiB floor")
    pk=torch.empty((HEADS*TILES,NEAR_TOKENS,D),device=device,dtype=torch.int8)
    pks=torch.empty((HEADS*TILES,NEAR_TOKENS),device=device,dtype=torch.float32)
    kernel[(HEADS*TILES,NEAR_TOKENS//32)](
        states["K"],states["KS"],states["IDX"],pk,pks,
        SRC_KEYS=15264,KX=KX,NEAR_TOKENS=NEAR_TOKENS,
        CENTRES_PER_HEAD=TILES,D=D,BLOCK=32,num_warps=4)
    torch.cuda.synchronize(device)
    # Validate every byte of the copied K8+KS on representative positions
    # in each head. Full attention PM/PL/PACC then validates ALL centres.
    tested=[]
    idx = states["IDX"].reshape(HEADS*TILES,KX)
    for c in (0,1,15,16,17,31,32,63,64,95,96,127):
        h=c//TILES
        j = torch.arange(NEAR_TOKENS,device=device,dtype=torch.int32)
        leaf=idx[c].index_select(0,(j//16).to(torch.long))
        tok=(leaf*16+j%16).to(torch.long)
        kref=states["K"][h].index_select(0,tok)
        ksref=states["KS"][h].index_select(0,tok)
        if not torch.equal(pk[c],kref) or not torch.equal(
                pks[c].view(torch.int32),ksref.view(torch.int32)):
            raise AssertionError("Near pack changed source K8 or FP32 scale at centre "+str(c))
        tested.append(c)
    return pk,pks,tested


def _launch_near(torch, states, tr, kernel, pk, pks, target):
    q=states["Q"].view(HEADS*TILES,1,ROWS,D)
    qs=tuple(map(int,q.stride()))
    v=states["V"]
    if not states.get("_direct_v",False) or tuple(v.shape)!=(HEADS,15264,D):
        raise AssertionError("Pinned original native BF16 V required")
    vs=tuple(map(int,v.stride()))
    lz=states["LZ"].view(HEADS*TILES,NB)
    return kernel[(HEADS*TILES,SPLITS,1)](
        q,states["MU"].view(HEADS*TILES,D),
        states["G"].view(HEADS*TILES,NB,D),lz,
        states["VB"].view(HEADS*TILES,NB,D),
        states["EXCL"].view(HEADS*TILES,NB),
        states["IDX"].view(HEADS*TILES,KX),
        pk.view(HEADS*TILES*NEAR_TOKENS,D),v,
        target["pm"].view(HEADS*TILES,SPLITS,ROWS),
        target["pl"].view(HEADS*TILES,SPLITS,ROWS),
        target["acc"].view(HEADS*TILES,SPLITS,ROWS,D),
        states["SG"].view(HEADS*TILES,NB),lz,lz,lz,
        pks.view(HEADS*TILES*NEAR_TOKENS),
        NB,KX,LPS,2048,0,1.0/math.sqrt(D),
        T=ROWS,D=D,BN=BN,BN_NEAR=BN,LEAF=16,
        PREC="tf32",DOT=1,GR=1,FP8=0,TAIL=0,RB=1,I8=1,
        PASS=0,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0,
        FAR_PV_ACC=0,NEAR_PV_ACC=0,
        BATCH_TILES_PER_HEAD=1,BATCH_KEY_ROWS=NEAR_TOKENS,
        Q_DIRECT=1,Q_SH=qs[0],Q_ST=qs[1],
        Q_SR=qs[2],Q_SD=qs[3],
        V_DIRECT=1,V_SH=vs[0],V_SS=vs[1],V_SD=vs[2],
        num_warps=4,num_stages=2)


def run_gpu():
    # Important: the exact git HEAD check runs BEFORE any torch/triton import.
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live
    from h3 import htrc_flat_model_shadow as shared
    from h3 import htrc_flat_head_batch as batch
    from htrc_near_kpack_triton import (pack_near_key_and_scale,
                                          near_kpacked_tile_kernel)
    if (not torch.cuda.is_available() or
        torch.cuda.get_device_capability(0)!=(12,0) or
        "RTX 5070 Ti" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Requires exact RTX 5070 Ti SM120 GPU")
    torch.set_grad_enabled(False)
    torch.manual_seed(42026)
    dev=torch.device("cuda:0")
    shape=(1,HEADS,TOKENS,D)
    banks=[tuple(torch.randn(shape,device=dev,dtype=torch.bfloat16)
                 for _ in range(3)) for _ in range(2)]
    old={key:os.environ.get(key) for key in ENV_KEYS}
    report={"banks":[],"candidate_speed_measurement":"UNRUN_NUMERIC_GATE_ONLY"}
    output_refs=[]
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-near-kpack-") as tmp:
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
            for bi,bank in enumerate(banks):
                full=flow.graph_attention_bthd(
                    *bank,heads=HEADS,mode="balanced",tiles=TILES,
                    budget_mib=1024,reserve_mib=256,workspace=True)
                torch.cuda.synchronize(dev)
                if bi==0:
                    g=json.loads((Path(tmp)/"first_call.json").read_text(encoding="utf-8"))
                    if g.get("status")!="PASS_FIRST_REAL_CALL" or g.get("all_groups_bit_exact") is not True:
                        raise AssertionError("Pinned original first-call Graph/eager did not pass")
                if tuple(full.shape)!=(1,TOKENS,HEADS,D):
                    raise AssertionError("Complete source Graph output wrong shape")
                output_refs.append(full.detach().cpu().clone())
                item=flow._SESSIONS.get((HEADS,),{}).get("graphs",{}).get((TILES,2048))
                if item is None: raise AssertionError("Expected original 2048-row session missing")
                states=item["states"]
                _assert_pinned(torch,states,cfg)
                selection=summarize_exclusions(
                    states["EXCL"].detach().cpu().tolist(),
                    states["IDX"].detach().cpu().tolist(),
                    splits=SPLITS,bn=BN,expected_near=KX)
                fp=selection["selection_mask_sha256"]
                if fp!=EXPECTED_MASKS[bi]:
                    raise AssertionError(f"Stale or wrong selection for QKV bank {bi}: {fp}")
                if selection["current_far_dot_blocks"]!=1920:
                    raise AssertionError("Pinned far geometry changed")

                ref=_alloc_partials(torch,dev)
                _,ctrl_compiled=batch.launch_prepared_tile_batch(
                    states,cfg,tr,buffers=ref)
                torch.cuda.synchronize(dev)
                tail=flow._SESSIONS[(HEADS,)]["tail"]
                ref_merged=_merge(torch,batch,states,cfg,tr,ref,tail)
                torch.cuda.synchronize(dev)
                origin_parity=_compare_bits(torch,item["output"],ref_merged)
                if not origin_parity["bit_exact"]:
                    raise AssertionError("Original full standalone Tile/Merge differs from Graph window")

                allocated_before=int(torch.cuda.memory_allocated(dev))
                pk,pks,sampled=_pack(torch,states,pack_near_key_and_scale)
                added_allocated=int(torch.cuda.memory_allocated(dev))-allocated_before
                if added_allocated < NEAR_SCRATCH_BYTES:
                    raise AssertionError("Packed K8/scale scratch was not accounted")
                cand=_alloc_partials(torch,dev)
                research_compiled=_launch_near(
                    torch,states,tr,near_kpacked_tile_kernel,pk,pks,cand)
                torch.cuda.synchronize(dev)
                diff=compare_far_partials(torch,ref,cand)
                merged=_merge(torch,batch,states,cfg,tr,cand,tail)
                torch.cuda.synchronize(dev)
                merged_diff=_compare_bits(torch,ref_merged,merged)
                exact=all(x["bit_exact"] for x in diff.values()) and merged_diff["bit_exact"]
                report["banks"].append({
                    "bank_index":bi,"selection_mask_sha256":fp,
                    "baseline_graph_window_output_bit_exact":origin_parity,
                    "packed_k8_scale_sampled_centres_bit_exact":sampled,
                    "candidate_partial_pm_pl_pacc":diff,
                    "candidate_merged_output":merged_diff,
                    "candidate_full_window_bit_exact":exact,
                    "compiled":{
                        "control":_compiled_resources(ctrl_compiled),
                        "near_kpacked":_compiled_resources(research_compiled)},
                    "packed_scratch_bytes":NEAR_SCRATCH_BYTES,
                    "measured_allocated_increase_bytes":added_allocated,
                })
                del cand,pk,pks,ref,merged,ref_merged,full
            if (torch.equal(output_refs[0],output_refs[1]) or
                report["banks"][0]["selection_mask_sha256"]==
                report["banks"][1]["selection_mask_sha256"]):
                raise AssertionError("Two QKV banks are not independent")
            eager=live.flat_attention_bthd(
                *banks[1],heads=HEADS,mode="balanced",tiles=TILES,
                budget_mib=1024,reserve_mib=256,workspace=True,
                query_staging="native_torch_mu",
                kv_staging="native_direct",dispatch="eager")
            if not torch.equal(eager.cpu(),output_refs[1]):
                raise AssertionError("Independent second-bank Graph/eager mismatch")
            report.update({
                "two_distinct_bf16_banks":True,
                "first_complete_graph_eager_bit_exact":True,
                "second_complete_graph_eager_bit_exact":True,
                "full_window_candidate_all_banks_bit_exact":all(
                    r["candidate_full_window_bit_exact"] for r in report["banks"]),
                "full_candidate_56head_graph":"UNRUN",
                "h3_model_or_video":"UNRUN",
                "speed_capture_blocks":[],
                "candidate_speed_measurement":"UNRUN_NUMERIC_GATE_ONLY",
                "production_modified":False})
            return report
    finally:
        try:flow.clear_session_cache()
        finally:
            for key,value in old.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expect-head",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report={"schema":"htrc-near-kpack-numeric-v1","status":"FAIL_CLOSED",
            "completed":False,
            "source_head":None,
            "probe":"ORIGINAL_PASS0_VS_ORDERED_K8_SCALE_NEAR_PACKED",
            "tokens":TOKENS,"heads":HEADS,"tiles":TILES,
            "exact_near_leaves":KX,"near_tokens":NEAR_TOKENS,
            "packed_budget":geometry_budget(),
            "speed_capture_blocks":[],
            "candidate_speed_measurement":"UNRUN_NUMERIC_GATE_ONLY",
            "production_promotion":"PROHIBITED_RESEARCH_ONLY",
            "h3_model_or_video":"UNRUN",
            "candidate_56head_graph":"UNRUN",
            "banks":[]}
    try:
        actual=subprocess.check_output(
            ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
        if actual!=args.expect_head:
            raise RuntimeError(f"Exact HEAD mismatch {actual} != {args.expect_head}")
        report["source_head"]=actual
        measured=run_gpu()
        report.update(measured)
        if report["full_window_candidate_all_banks_bit_exact"]:
            report["status"]="PASS_NEAR_KPACK_NUMERIC_ONLY"
            report["parity_admission"]="PASS_TWO_BANK_WINDOW_PARTIALS_AND_MERGE"
            report["completed"]=True
        else:
            report["status"]="FAIL_CLOSED"
            report["parity_admission"]="NO_GO_NEAR_KPACK_BIT_EXACT"
            report["error"]="Candidate K8 near pack PM/PL/PACC or Merge not bit-exact"
    except BaseException as exc:
        report["status"]="FAIL_CLOSED"
        report["completed"]=False
        report["error"]=f"{type(exc).__name__}: {exc}"[:1800]
        report["traceback_tail"]=traceback.format_exc()[-4500:]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"one_json={args.output} status={report['status']} parity={report.get('parity_admission','UNRUN')}")
    return 0 if report["status"]=="PASS_NEAR_KPACK_NUMERIC_ONLY" else 2


if __name__=="__main__":
    raise SystemExit(main())
