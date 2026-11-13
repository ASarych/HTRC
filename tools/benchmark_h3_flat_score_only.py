"""Isolated GPU A/B for HTRC stats SCORE_ONLY (no VB computation/write).

Real H3 full-attention and video are NOT tested here. Original and candidate
use the same Triton _stats_kernel; candidate sets SCORE_ONLY=1. No changes
to the live Flat path; this measures the first stage of a proposed two-pass
streaming algorithm that would later recompute VB directly inside far Tile.

Example:
  python -B tools/benchmark_h3_flat_score_only.py --tokens 40759 --heads 4
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from h3 import htrc_flat_triton as tr


def run_case(tokens: int, heads: int, mode: str, tiles: int,
             rounds: int, warmups: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required: this is not a CPU test")
    if tokens < 16 or heads < 1 or tiles not in (16,32,64):
        raise ValueError("Invalid shape/geometry")
    torch.manual_seed(260213)
    dev=torch.device("cuda")
    dim,leaf=128,16
    nkeys=tokens//leaf*leaf
    leaves=nkeys//leaf
    # Model-matched native BF16 values, distinct heads, no duplicated QKV.
    k=torch.randn((heads,nkeys,dim),dtype=torch.float32,device=dev).bfloat16()
    v=torch.randn_like(k)
    q=torch.randn((heads,tiles*64,dim),dtype=torch.float32,device=dev).bfloat16()
    mu=q.float().view(heads,tiles,64,dim).mean(dim=2).contiguous()
    centres=tiles
    bt,bl=16,4
    blocks=math.ceil(centres/bt)
    group=tr.triton.cdiv(leaves,bl)
    size=(heads,centres,leaves)
    def allocate(score_only):
        g=torch.empty(size+(dim,),device=dev,dtype=torch.int8)
        lz=torch.empty(size,device=dev,dtype=torch.float32)
        sg=torch.empty(size,device=dev,dtype=torch.float32)
        vb=(torch.full((1,),-13,device=dev,dtype=torch.bfloat16)
            if score_only else
            torch.empty(size+(dim,),device=dev,dtype=torch.bfloat16))
        return {"g":g,"vb":vb,"lz":lz,"sg":sg}
    eager=allocate(False)
    lean=allocate(True)
    scratch=torch.empty_like(eager["lz"])

    def launch(record,score_only):
        tr._stats_kernel[(heads*blocks,group)](
            mu.view(heads*centres,dim),
            k.view(heads*nkeys,dim),v.view(heads*nkeys,dim),
            record["g"].view(heads*centres,leaves,dim),
            record["lz"].view(heads*centres,leaves),
            record["vb"],
            record["sg"].view(heads*centres,leaves),
            scratch,
            heads*centres,leaves,1/math.sqrt(dim),
            D=dim,BT=bt,LEAF=leaf,BL=bl,PREC="tf32",
            FP8=0,I8=1,
            BATCH_TILES_PER_HEAD=centres,
            BATCH_KEY_ROWS=nkeys,
            BATCH_TILE_BLOCKS_PER_HEAD=blocks,
            SCORE_ONLY=int(score_only),
            num_warps=4,
        )

    def selection(record):
        # Identical rowmax/topk path for both variants. Selection IDs
        # must agree exactly; no near-tie forgiveness in this gate.
        out=torch.empty((heads*centres,leaves),device=dev,dtype=torch.float32)
        lps=tr._split(leaves,4,32)
        tr._rowmax_kernel[
            (heads*centres,tr.triton.cdiv(leaves,lps))
        ](
            q.float().view(heads*centres,64,dim),mu.view(heads*centres,dim),
            record["g"].view(heads*centres,leaves,dim),
            record["lz"].view(heads*centres,leaves),
            out,record["sg"].view(heads*centres,leaves),
            leaves,lps,T=64,TS=64,STRIDE=1,D=dim,BN=32,
            PREC="tf32",DOT=1,I8=1,num_warps=4,num_stages=1,
        )
        exact=min({"fast":256,"balanced":512}[mode],leaves)
        ids=torch.topk(out,exact,dim=-1,sorted=False).indices
        exclusion=torch.zeros_like(out,dtype=torch.int8)
        exclusion.scatter_(1,ids,1)
        return out,ids,exclusion

    tr._dll()
    for _ in range(warmups):
        launch(eager,False)
        launch(lean,True)
    torch.cuda.synchronize(dev)
    a_sel,a_ids,a_ex=selection(eager)
    b_sel,b_ids,b_ex=selection(lean)
    checks={
        "G_equal":bool(torch.equal(eager["g"],lean["g"])),
        "LZ_equal":bool(torch.equal(eager["lz"],lean["lz"])),
        "SG_equal":bool(torch.equal(eager["sg"],lean["sg"])),
        "SEL_equal":bool(torch.equal(a_sel,b_sel)),
        "IDX_equal":bool(torch.equal(a_ids,b_ids)),
        "EXCL_equal":bool(torch.equal(a_ex,b_ex)),
        "VB_sentinel_unchanged":bool(torch.all(lean["vb"] == -13)),
    }
    if not all(checks.values()):
        raise AssertionError(f"Selection-only stats parity FAILED: {checks}")
    timings={"eager":[],"score_only":[]}
    for i in range(rounds):
        # Alternate order to limit drift; GPU events exclude host sync gaps.
        modes=([("eager",eager,False),("score_only",lean,True)]
               if i%2==0 else
               [("score_only",lean,True),("eager",eager,False)])
        for label,record,flag in modes:
            start=torch.cuda.Event(enable_timing=True)
            stop=torch.cuda.Event(enable_timing=True)
            start.record()
            launch(record,flag)
            stop.record()
            stop.synchronize()
            timings[label].append(float(start.elapsed_time(stop)))
    med={name:statistics.median(t) for name,t in timings.items()}
    return {
        "tokens":tokens,"heads":heads,"distinct_generated_heads":True,
        "mode":mode,"window_tiles":tiles,"centres":centres,
        "full_key_leaves":leaves,"topk_exact_leaves":min(
            {"fast":256,"balanced":512}[mode],leaves),
        "valid_stage":"stats_only_and_rowmax_selection",
        "all_parity_checks_passed":True,"parity":checks,
        "candidate_has_no_VB_storage_or_V_dot":True,
        "saved_VB_write_bytes_per_window":heads*centres*leaves*dim*2,
        "timing_scope":"one CUDA stats kernel invocation, NOT all windows/heads/H3",
        "stats_gpu_event_ms_medians":med,
        "speedup_eager_over_score_only":med["eager"]/med["score_only"],
        "rounds":rounds,"warmups":warmups,
        "hardware":torch.cuda.get_device_name(dev),
        "far_tile_and_full_output_parity":"UNRUN_NOT_IMPLEMENTED",
        "model_resident":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens",type=int,default=40759)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--mode",choices=("fast","balanced"),default="fast")
    p.add_argument("--tiles",type=int,choices=(16,32,64),default=16)
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--warmups",type=int,default=4)
    p.add_argument("--output",type=Path,
                   default=Path(r"D:\ComfyOutput\H3\math_probe\flat_score_only_gpu_ab.json"))
    args=p.parse_args()
    if args.rounds<2 or args.warmups<1:
        raise ValueError("Rounds must be >=2 and warmups >=1")
    started=time.perf_counter()
    report={"experiment":"h3_flat_stats_score_only_gpu_ab_v1",
            "status":"FAILED","completed":False}
    try:
        report.update(run_case(args.tokens,args.heads,args.mode,args.tiles,
                               args.rounds,args.warmups))
        report["status"]="PASS_NUMERIC"
        report["completed"]=True
    except BaseException as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"]=time.perf_counter()-started
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(f"report={args.output} status={report['status']}")
    print(json.dumps({"status":report["status"],"checks":report["parity"],
                      "medians":report["stats_gpu_event_ms_medians"]},indent=2))


if __name__=="__main__":
    main()
