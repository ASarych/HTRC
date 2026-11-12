"""Exact-math H3 Flat Q input packing A/B. One report, no quality coefficient.

Q comes from realistic post-RoPE BHSD strided BF16 storage, with 4/56
genuinely distinct *synthetic* heads. Both variants retain the SAME fixed
Flat geometry/leaf count/stats/topk/Tile/Merge; only GPU Q staging differs.

Not a real model or video test. Gate ALL prepared tensors on every Q window,
and representative final Tile/Merge outputs, before accepting timing.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from h3 import htrc_flat_head_batch as batch
from h3 import htrc_flat_live_backend as live
from h3 import htrc_flat_model_shadow as shared

STATES=("Q","MU","G","VB","LZ","SG","IDX","EXCL")
PACKS=("gather","view_copy")


def window_cap(cap,begin,stop,full_rows):
    src=cap["q_qhd"][begin:stop]
    valid=stop-begin
    if valid<full_rows:
        src=torch.cat((
            src,src[-1:].expand(full_rows-valid,-1,-1)
        ),dim=0)
    window=dict(cap)
    window["q_qhd"]=src
    return window,valid


def prepare(window,ids,cfg,tri,kv,ws,tiles,valid,mode):
    st,*_=batch.prepare_batched_head_states(
        window,ids,cfg,tri,tile_count=tiles,
        quant_backend="triton",input_cast="gpu",
        kv_cache=kv,query_workspace=ws,
        valid_query_rows=valid,query_staging=mode,
    )
    return st


def output(states,cfg,tri,ws,tails):
    parts,_=batch.launch_prepared_tile_batch(
        states,cfg,tri,buffers=ws["_partials"]
    )
    result,_=batch.launch_prepared_merge_batch(
        states,cfg,tri,parts,tails=tails,
        output=ws["_out"],merge_rows=32
    )
    return result


def timed_prepare(window,ids,cfg,tri,kv,ws,tiles,valid,mode):
    start=torch.cuda.Event(enable_timing=True)
    finish=torch.cuda.Event(enable_timing=True)
    cpu_start=time.perf_counter()
    start.record()
    state=prepare(window,ids,cfg,tri,kv,ws,tiles,valid,mode)
    finish.record()
    finish.synchronize()
    ms=start.elapsed_time(finish)
    cpu_ms=(time.perf_counter()-cpu_start)*1000
    del state
    return ms,cpu_ms


def run(args,report):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA RTX GPU is required")
    free,total=torch.cuda.mem_get_info()
    report["gpu"]={
        "device":torch.cuda.get_device_name(),
        "free_before_gib":round(free/2**30,3),
        "total_gib":round(total/2**30,3),
    }
    if free<args.min_free_gib*2**30:
        raise MemoryError("Available GPU memory below --min-free-gib")
    torch.manual_seed(args.seed)
    dev=torch.device("cuda")
    # Native BHSD allocations: [1,H,S,D]. The [S,H,D] capture is a VIEW.
    # Random nonzero BF16 avoids the tied selection of zero-filled profiles.
    shape=(1,args.heads,args.tokens,128)
    q=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    k=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    v=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    cap=shared.capture_from_bhsd(q,k,v,args.heads)
    tri=live._triton()
    tri._dll()
    cfg=shared.flat_config(args.mode,args.tiles)
    if int(cfg["kern"].get("group",1))!=1:
        raise ValueError("Fixed no-coefficient test requires center_group=1")
    row_count=args.tiles*64
    report["configuration"]={
        "mode":args.mode,"quality_coefficient_used":False,
        "exact_leaves":int(cfg["exact_leaves"]),
        "center_group":int(cfg["kern"].get("group",1)),
        "tokens":args.tokens,"native_heads":args.heads,
        "dtype":"torch.bfloat16","qkv_layout":"B,H,S,D views",
        "head_group_limit":args.group_cap,
        "tile_window_rows":row_count,
        "query_windows":math.ceil(args.tokens/row_count),
        "rounds":args.rounds,"warmups":args.warmups,
    }
    full_checkpoints={0,math.ceil(args.tokens/row_count)//2,
                      math.ceil(args.tokens/row_count)-1}
    for count in args.head_cases:
        case={
            "head_count":count,
            "native_heads_unique":True,
            "status":"STARTED",
            "checked_windows":0,
            "state_checks":0,
            "mismatches":[],
            "output_full_flat_checks":0,
            "groups":[],
        }
        report["cases"].append(case)
        if count>args.heads:
            raise ValueError("Requested more than available unique model-style heads")
        for begin in range(0,count,args.group_cap):
            end=min(count,begin+args.group_cap)
            ids=list(range(begin,end))
            kv=batch.prepare_gpu_head_kv_cache(cap,ids,tri)
            ws={
                p:shared.alloc_workspace(cap,len(ids),args.tiles,cfg)
                for p in PACKS
            }
            tails=shared._tail(cap,ids)
            samples={p:[] for p in PACKS}
            cpu_samples={p:[] for p in PACKS}
            for wi in range(report["configuration"]["query_windows"]):
                lo=wi*row_count
                hi=min(args.tokens,lo+row_count)
                window,valid=window_cap(cap,lo,hi,row_count)
                a=prepare(window,ids,cfg,tri,kv,ws["gather"],args.tiles,valid,"gather")
                b=prepare(window,ids,cfg,tri,kv,ws["view_copy"],args.tiles,valid,"view_copy")
                for state in STATES:
                    if not torch.equal(a[state],b[state]):
                        diff=(a[state].float()-b[state].float()).abs().amax().item()
                        case["mismatches"].append({
                            "group":[begin,end],"window":wi,"state":state,
                            "max_abs":float(diff)
                        })
                    case["state_checks"]+=1
                if case["mismatches"]:
                    case["status"]="FAIL_MATH_MISMATCH"
                    return
                if wi in full_checkpoints:
                    oa=output(a,cfg,tri,ws["gather"],tails)
                    ob=output(b,cfg,tri,ws["view_copy"],tails)
                    if not torch.equal(oa,ob):
                        err=float((oa.float()-ob.float()).abs().amax().item())
                        case["mismatches"].append({
                            "group":[begin,end],"window":wi,
                            "state":"final_flat_output","max_abs":err
                        })
                        case["status"]="FAIL_OUTPUT_MISMATCH"
                        return
                    case["output_full_flat_checks"]+=1
                    del oa,ob
                del a,b
                for _ in range(args.warmups):
                    for pack in PACKS:
                        tmp=prepare(window,ids,cfg,tri,kv,ws[pack],
                                    args.tiles,valid,pack)
                        del tmp
                for _ in range(args.rounds):
                    for pack in ("gather","view_copy","view_copy","gather"):
                        gpu_ms,cpu_ms=timed_prepare(
                            window,ids,cfg,tri,kv,ws[pack],args.tiles,valid,pack
                        )
                        samples[pack].append(gpu_ms)
                        cpu_samples[pack].append(cpu_ms)
                case["checked_windows"]+=1
                print(f"Q_PACK {count} heads {begin}:{end} window "
                      f"{wi+1}/{report['configuration']['query_windows']}",
                      flush=True)
                del window
            med={p:statistics.median(samples[p]) for p in PACKS}
            result={
                "head_range":[begin,end],
                "windows_checked":report["configuration"]["query_windows"],
                "gpu_event_median_prepare_ms":med,
                "gpu_event_total_samples_ms":{p:sum(samples[p]) for p in PACKS},
                "cpu_enqueue_median_prepare_ms":{
                    p:statistics.median(cpu_samples[p]) for p in PACKS
                },
                "speedup_gather_over_view_copy":med["gather"]/med["view_copy"],
            }
            case["groups"].append(result)
            del ws,kv,tails
        case["status"]="PASS_BIT_EXACT_STATES_AND_OUTPUT"
        case["summary"]={
            "sum_group_median_prepare_ms":{
                p:sum(g["gpu_event_median_prepare_ms"][p] for g in case["groups"])
                for p in PACKS
            },
            "speedup_gather_over_view_copy":
                sum(g["gpu_event_median_prepare_ms"]["gather"] for g in case["groups"])/
                sum(g["gpu_event_median_prepare_ms"]["view_copy"] for g in case["groups"]),
            "state_checks":case["state_checks"],
            "representative_final_output_checks":case["output_full_flat_checks"],
        }
    report["completed"]=all(
        c["status"]=="PASS_BIT_EXACT_STATES_AND_OUTPUT" for c in report["cases"]
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokens",type=int,default=18723)
    p.add_argument("--heads",type=int,default=56)
    p.add_argument("--head-cases",default="4,56")
    p.add_argument("--group-cap",type=int,default=11)
    p.add_argument("--mode",choices=("fast","balanced"),default="fast")
    p.add_argument("--tiles",type=int,default=16)
    p.add_argument("--rounds",type=int,default=2)
    p.add_argument("--warmups",type=int,default=1)
    p.add_argument("--seed",type=int,default=20261112)
    p.add_argument("--min-free-gib",type=float,default=4.0)
    p.add_argument("--out",type=Path,default=Path(
        r"D:\ComfyOutput\H3\math_probe\flat_qprep_fixed_math_ab.json"
    ))
    args=p.parse_args()
    try:
        cases=[int(x.strip()) for x in args.head_cases.split(",")]
    except ValueError:
        p.error("head-cases must be comma-separated integers")
    if (not cases or any(x<1 or x>args.heads for x in cases) or
        len(set(cases))!=len(cases)):
        p.error("head-cases require distinct positive counts <= heads")
    if any(x<=0 for x in (
        args.tokens,args.heads,args.group_cap,args.tiles,args.rounds,args.warmups
    )) or args.min_free_gib<=0:
        p.error("invalid positive runtime arguments")
    args.head_cases=cases
    report={
        "experiment":"h3_flat_query_pack_no_quality_coefficient_v1",
        "completed":False,
        "quality_coefficient_used":False,
        "fixed_math":True,
        "parity_reference":"same pinned Flat mode, gather Q input",
        "cases":[],"errors":[],
        "notes":[
            "Original selection/leaf geometry retained; no speed-quality coefficient.",
            "All Q and intermediate prepared state tensors must be BITWISE equal.",
            "Tile/Merge output equality checked on first/middle/last Q window per group.",
            "Synthetic nonzero native-style BF16 QKV; no weights or video quality inference.",
            "CUDA-event times include kernel launch gaps; do not equate to video speed.",
            "One JSON result; the experimental live backend defaults to original gather.",
        ],
    }
    try:
        run(args,report)
    except BaseException as exc:
        report["errors"].append({
            "type":type(exc).__name__,"message":str(exc)[:1300],
            "traceback_tail":traceback.format_exc()[-2200:]
        })
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",args.out,flush=True)
    print("STATUS:", "PASS" if report["completed"] else "FAIL",flush=True)
    return 0 if report["completed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
