"""Direct native-strided Q/K/V experimental A/B, fixed Flat math, one JSON.

Nonzero synthetic BF16 H3 BHSD source, 4 and 56 truly unique synthetic
heads. Does not load a model or establish video quality. This test compares
four orthogonal data-layout choices, not CUDA Graph or Direct-Flow execution.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import torch
from h3 import htrc_flat_head_batch as batch
from h3 import htrc_flat_live_backend as live
from h3 import htrc_flat_model_shadow as shared
from h3 import htrc_flat_windows as windows

ARMS={
    "baseline":("gather",False),
    "direct_q":("native_direct",False),
    "direct_kv":("gather",True),
    "direct_qkv":("native_direct",True),
    "direct_q_torch_mu":("native_torch_mu",False),
    "direct_qkv_torch_mu":("native_torch_mu",True),
}


def kv_cache_by_arm(packed, native):
    """Derive cache selection from ARMS metadata; never hardcode arm names.

    Every future benchmark arm must declare (query_staging, native_kv).
    """
    return {
        arm: (native if use_native_kv else packed)
        for arm, (_query_staging, use_native_kv) in ARMS.items()
    }


STATES=("Q","MU","G","VB","LZ","SG","IDX","EXCL")
# Explicit fixed-math numerical gates, not video-quality acceptance thresholds.
MAX_ABS={"MU":2e-5,"G":1.0,"VB":0.01,"LZ":3e-5,"SG":3e-5}
OUTPUT_ATOL=2e-4
OUTPUT_RTOL=2e-4


def capture_window(cap, begin, end, count):
    raw=cap["q_qhd"][begin:end]
    valid=end-begin
    if valid<count:
        raw=torch.cat((raw,raw[-1:].expand(count-valid,-1,-1)),dim=0)
    result=dict(cap)
    result["q_qhd"]=raw
    return result,valid


def prepare(cap, heads, cfg, tr, kv, ws, tiles, valid, query_mode):
    result,*_=batch.prepare_batched_head_states(
        cap,heads,cfg,tr,tile_count=tiles,quant_backend="triton",
        input_cast="gpu",kv_cache=kv,query_workspace=ws,
        valid_query_rows=valid,query_staging=query_mode,
    )
    return result


def output(st,cfg,tr,ws,tail):
    part,_=batch.launch_prepared_tile_batch(
        st,cfg,tr,buffers=ws["_partials"]
    )
    merged,_=batch.launch_prepared_merge_batch(
        st,cfg,tr,part,tails=tail,output=ws["_out"],merge_rows=32
    )
    return merged


def compare(base,other,state):
    if state=="Q":
        # BF16 native pointer vs FP32 staged Q: all FP32 values must be exact.
        ok=torch.equal(base.float(),other.float())
    elif state in ("IDX","EXCL"):
        # No near-field routing changes, including near ties.
        ok=torch.equal(base,other)
    else:
        abs_error=(base.float()-other.float()).abs()
        max_abs=abs_error.amax().item()
        ok=bool(max_abs<=MAX_ABS[state])
        return ok,float(max_abs)
    max_abs=0.0 if ok else float(
        (base.float()-other.float()).abs().amax().item()
    )
    return ok,max_abs


def gated_output(base,other):
    diff=(base.float()-other.float()).abs()
    max_abs=float(diff.amax().item())
    denom=base.float().square().sum().sqrt().clamp_min(1e-30)
    relative=float(diff.square().sum().sqrt().div(denom).item())
    ok=bool(torch.allclose(base,other,atol=OUTPUT_ATOL,rtol=OUTPUT_RTOL))
    return ok,{"max_abs":max_abs,"relative_l2":relative}


def gpu_timed_full(cap,heads,cfg,tr,kv,ws,tiles,valid,query_mode,tail):
    start=torch.cuda.Event(enable_timing=True)
    stop=torch.cuda.Event(enable_timing=True)
    cpu_start=time.perf_counter()
    start.record()
    st=prepare(cap,heads,cfg,tr,kv,ws,tiles,valid,query_mode)
    result=output(st,cfg,tr,ws,tail)
    stop.record()
    stop.synchronize()
    cpu_ms=(time.perf_counter()-cpu_start)*1000.0
    gpu_ms=start.elapsed_time(stop)
    del st,result
    return float(gpu_ms),float(cpu_ms)


def run(args,report):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; RTX 5070 Ti target")
    free,total=torch.cuda.mem_get_info()
    report["gpu"]={
        "device":torch.cuda.get_device_name(),
        "free_before_gib":round(free/2**30,3),
        "total_gib":round(total/2**30,3),
    }
    if free<args.min_free_gib*2**30:
        raise MemoryError("Not enough free VRAM")
    torch.manual_seed(args.seed)
    dev=torch.device("cuda")
    shape=(1,args.heads,args.tokens,128)
    q=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    k=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    v=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    cap=shared.capture_from_bhsd(q,k,v,args.heads)
    tr=live._triton()
    tr._dll()
    cfg=shared.flat_config("fast",args.tiles)
    if (int(cfg["exact_leaves"])!=256 or
        int(cfg["kern"].get("group",1))!=1):
        raise ValueError("Fixed fast math must retain exact_leaves=256, group=1")
    nrows=args.tiles*64
    query_windows=list(windows.iter_query_windows(
        args.tokens,args.tiles,tile_rows=int(cfg["tile"]),
        center_group=int(cfg["kern"].get("group",1))
    ))
    window_count=len(query_windows)
    report["config"]={
        "mode":"fast","exact_leaves":256,"center_group":1,
        "tokens":args.tokens,"heads":args.heads,
        "tiles":args.tiles,"query_windows":window_count,
        "window_geometry":[
            {"valid_rows":w.valid_rows,"active_tiles":w.tile_count,
             "padded_rows":w.padded_rows}
            for w in query_windows
        ],
        "group_cap":args.group_cap,
        "qkv_dtype":"bfloat16","rounds":args.rounds,"warmups":args.warmups,
        "q_stride":list(cap["q_qhd"].stride()),
        "k_stride":list(cap["k_khd"].stride()),
        "v_stride":list(cap["v_khd"].stride()),
        "output_atol":OUTPUT_ATOL,"output_rtol":OUTPUT_RTOL,
        "stage_max_abs":MAX_ABS,
    }
    full_checks={0,window_count//2,window_count-1}
    for count in args.head_cases:
        case={
            "head_count":count,"unique_synthetic_heads":True,
            "completed":False,"group_reports":[],
            "state_checks":0,"output_checks":0,
            "legacy_tail_output_checks":0,
            "errors":[],"max_stage_error":{},
            "max_output_error":{"max_abs":0.0,"relative_l2":0.0},
        }
        report["cases"].append(case)
        for begin in range(0,count,args.group_cap):
            end=min(begin+args.group_cap,count)
            ids=list(range(begin,end))
            tbuild=time.perf_counter()
            packed=batch.prepare_gpu_head_kv_cache(cap,ids,tr)
            torch.cuda.synchronize()
            tpacked=(time.perf_counter()-tbuild)*1000
            tbuild=time.perf_counter()
            native=batch.prepare_gpu_head_kv_cache(
                cap,ids,tr,native_direct=True
            )
            torch.cuda.synchronize()
            tnative=(time.perf_counter()-tbuild)*1000
            if not torch.equal(packed["K"],native["K"]):
                raise AssertionError(f"Direct K8 bytes differ for heads {begin}:{end}")
            if not torch.equal(packed["KS"],native["KS"]):
                raise AssertionError(f"Direct K8 scales differ for heads {begin}:{end}")
            if not torch.equal(packed["VBF"],native["VBF"]):
                raise AssertionError(f"Native VBF differs for heads {begin}:{end}")
            kv=kv_cache_by_arm(packed,native)
            ws={
                arm:shared.alloc_workspace(cap,len(ids),args.tiles,cfg)
                for arm in ARMS
            }
            tail=shared._tail(cap,ids)
            samples={arm:[] for arm in ARMS}
            cpus={arm:[] for arm in ARMS}
            for wi,geometry in enumerate(query_windows):
                low,high=geometry.start,geometry.end
                valid=geometry.valid_rows
                active_tiles=geometry.tile_count
                window,valid=capture_window(
                    cap,low,high,geometry.padded_rows
                )
                active_ws={
                    arm:windows.compact_workspace_view(
                        ws[arm],len(ids),active_tiles,cfg
                    ) for arm in ARMS
                }
                # Compare compact output with original fully padded query
                # window. Only REAL rows must match bit-for-bit.
                legacy=None
                if active_tiles<args.tiles:
                    old_window,_=capture_window(cap,low,high,nrows)
                    old_states=prepare(
                        old_window,ids,cfg,tr,packed,ws["baseline"],
                        args.tiles,valid,"gather"
                    )
                    legacy=output(
                        old_states,cfg,tr,ws["baseline"],tail
                    ).reshape(len(ids),nrows,128)[:,:valid].clone()
                    del old_window,old_states
                states={
                    arm:prepare(window,ids,cfg,tr,kv[arm],active_ws[arm],
                                active_tiles,valid,mode[0])
                    for arm,mode in ARMS.items()
                }
                if legacy is not None:
                    compact=output(
                        states["baseline"],cfg,tr,active_ws["baseline"],tail
                    ).reshape(len(ids),geometry.padded_rows,128)[:,:valid]
                    case["legacy_tail_output_checks"]+=1
                    if not torch.equal(legacy,compact):
                        case["errors"].append({
                            "window":wi,"heads":[begin,end],
                            "arm":"baseline","stage":"compact_vs_legacy",
                            "max_abs":float(
                                (legacy-compact).abs().amax().item()
                            )
                        })
                    del legacy,compact
                    if case["errors"]:
                        return
                for arm in ARMS:
                    if arm=="baseline":
                        continue
                    for name in STATES:
                        ok,err=compare(states["baseline"][name],states[arm][name],name)
                        case["state_checks"]+=1
                        key=f"{arm}:{name}"
                        case["max_stage_error"][key]=max(
                            case["max_stage_error"].get(key,0.0),err
                        )
                        if not ok:
                            case["errors"].append({
                                "window":wi,"heads":[begin,end],"arm":arm,
                                "stage":name,"max_abs":err
                            })
                if case["errors"]:
                    return
                if wi in full_checks:
                    old=output(
                        states["baseline"],cfg,tr,active_ws["baseline"],tail
                    ).clone()
                    for arm in ARMS:
                        if arm=="baseline":
                            continue
                        other=output(states[arm],cfg,tr,active_ws[arm],tail)
                        ok,metrics=gated_output(old,other)
                        case["output_checks"]+=1
                        for field in ("max_abs","relative_l2"):
                            case["max_output_error"][field]=max(
                                case["max_output_error"][field],metrics[field]
                            )
                        if not ok:
                            case["errors"].append({
                                "window":wi,"heads":[begin,end],
                                "arm":arm,"output_error":metrics
                            })
                    del old
                if case["errors"]:
                    return
                del states
                for _ in range(args.warmups):
                    for arm,(qmode,_) in ARMS.items():
                        a=prepare(window,ids,cfg,tr,kv[arm],active_ws[arm],
                                  active_tiles,valid,qmode)
                        output(a,cfg,tr,active_ws[arm],tail)
                        del a
                for _ in range(args.rounds):
                    # Reverse sequence across adjacent rounds to reduce order bias.
                    order=list(ARMS) if _%2==0 else list(reversed(ARMS))
                    for arm in order:
                        qmode,_direct_kv=ARMS[arm]
                        gpu_ms,cpu_ms=gpu_timed_full(
                            window,ids,cfg,tr,kv[arm],active_ws[arm],
                            active_tiles,valid,qmode,tail
                        )
                        samples[arm].append(gpu_ms)
                        cpus[arm].append(cpu_ms)
                print(f"DIRECT_QKV case={count} group={begin}:{end} "
                      f"window={wi+1}/{window_count} active_tiles={active_tiles} "
                      f"valid={valid}",flush=True)
                del window,active_ws
            gr={
                "heads":[begin,end],"windows":window_count,
                "short_window_reuses_full_workspace":True,
                "k8_bytes_and_scales_equal":True,
                "kv_cache_build_ms_including_cold_start":{
                    "packed":tpacked,"native":tnative},
                "kv_cache_bytes":{
                    "packed":batch.gpu_head_kv_cache_bytes(packed),
                    "native":batch.gpu_head_kv_cache_bytes(native)
                },
                "full_call_gpu_event_median_ms":{
                    arm:statistics.median(samples[arm]) for arm in ARMS
                },
                "full_call_cpu_wall_median_ms":{
                    arm:statistics.median(cpus[arm]) for arm in ARMS
                },
            }
            case["group_reports"].append(gr)
            del ws,kv,tail,native,packed
        case["completed"]=True
        case["sum_group_gpu_event_medians_ms"]={
            arm:sum(g["full_call_gpu_event_median_ms"][arm]
                    for g in case["group_reports"])
            for arm in ARMS
        }
        base_ms=case["sum_group_gpu_event_medians_ms"]["baseline"]
        case["speedup_vs_baseline"]={
            arm:base_ms/ms for arm,ms in
            case["sum_group_gpu_event_medians_ms"].items()
        }
    report["completed"]=all(c["completed"] and not c["errors"]
                            for c in report["cases"])



def diagnose_stages(args,report):
    """Isolate Direct Q slowdown without another full 4/56-head parity run.

    The main A/B's earlier PASS must be checked separately. Event intervals
    deliberately include launch gaps, but do not synchronize between stages.
    Every arm uses the identical QKV source and the SAME fixed Flat math.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Stage diagnostics require CUDA")
    free,total=torch.cuda.mem_get_info()
    if free < args.min_free_gib * 2**30:
        raise MemoryError("Insufficient free GPU memory for stage diagnostics")
    report["gpu"]={
        "device":torch.cuda.get_device_name(),
        "free_before_gib":round(free/2**30,3),
        "total_gib":round(total/2**30,3),
    }
    torch.manual_seed(args.seed)
    dev=torch.device("cuda")
    shape=(1,args.heads,args.tokens,128)
    q=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    k=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    v=torch.randn(shape,device=dev,dtype=torch.bfloat16)
    cap=shared.capture_from_bhsd(q,k,v,args.heads)
    tr=live._triton()
    tr._dll()
    cfg=shared.flat_config("fast",args.tiles)
    group=int(cfg["kern"].get("group",1))
    if int(cfg["exact_leaves"])!=256 or group!=1:
        raise ValueError("Fixed-math stage probe requires fast 256 leaves, group=1")
    seq=list(windows.iter_query_windows(args.tokens,args.tiles,
        tile_rows=int(cfg["tile"]),center_group=group))
    positions=[("full",seq[0])]
    if seq[-1].padded_rows<args.tiles*int(cfg["tile"]):
        positions.append(("short",seq[-1]))
    counts=sorted(set((1,min(4,args.heads,args.group_cap),
                       min(11,args.heads,args.group_cap))))
    report.update({
        "diagnostic_only":True,
        "numerical_parity":"NOT_REPEATED_IN_STAGE_DIAGNOSTIC",
        "arithmetic":"fixed_fast_256_exact_leaves_group1",
        "config":{
            "source_heads":args.heads,"tokens":args.tokens,
            "tile_capacity":args.tiles,"center_group":group,
            "head_group_sizes":counts,
            "samples_per_stage":args.rounds,
            "warmups":args.warmups,
            "q_stride":list(cap["q_qhd"].stride()),
        },
        "diagnostic_cases":[],
    })
    if args.rounds<3:
        raise ValueError("Use --rounds >= 3 for stage-profile medians")
    for count in counts:
        ids=list(range(count))
        packed=batch.prepare_gpu_head_kv_cache(cap,ids,tr)
        native=batch.prepare_gpu_head_kv_cache(
            cap,ids,tr,native_direct=True
        )
        for name,geom in positions:
            window,valid=capture_window(
                cap,geom.start,geom.end,geom.padded_rows
            )
            selected={
                "head_count":count,"window":name,
                "valid_rows":valid,"active_tiles":geom.tile_count,
                "arms":{},
            }
            report["diagnostic_cases"].append(selected)
            tail=shared._tail(cap,ids)
            for arm,(mode,use_native) in ARMS.items():
                cache=native if use_native else packed
                ws=shared.alloc_workspace(cap,count,args.tiles,cfg)
                active_ws=windows.compact_workspace_view(
                    ws,count,geom.tile_count,cfg
                )
                for _ in range(args.warmups):
                    st=prepare(window,ids,cfg,tr,cache,active_ws,
                               geom.tile_count,valid,mode)
                    output(st,cfg,tr,active_ws,tail)
                    del st
                torch.cuda.synchronize()
                stage_samples={}
                cpu_wall=[]
                for _ in range(args.rounds):
                    marks=[]
                    started=time.perf_counter()
                    st,*_=batch.prepare_batched_head_states(
                        window,ids,cfg,tr,tile_count=geom.tile_count,
                        quant_backend="triton",input_cast="gpu",
                        kv_cache=cache,query_workspace=active_ws,
                        valid_query_rows=valid,query_staging=mode,
                        stage_gpu_events=marks,
                    )
                    tile_start=torch.cuda.Event(enable_timing=True)
                    tile_end=torch.cuda.Event(enable_timing=True)
                    merge_end=torch.cuda.Event(enable_timing=True)
                    tile_start.record()
                    parts,_=batch.launch_prepared_tile_batch(
                        st,cfg,tr,buffers=active_ws["_partials"]
                    )
                    tile_end.record()
                    merged,_=batch.launch_prepared_merge_batch(
                        st,cfg,tr,parts,tails=tail,
                        output=active_ws["_out"],merge_rows=32
                    )
                    merge_end.record()
                    merge_end.synchronize()
                    cpu_wall.append((time.perf_counter()-started)*1000)
                    for stage,a,b in marks:
                        stage_samples.setdefault(stage,[]).append(
                            float(a.elapsed_time(b))
                        )
                    stage_samples.setdefault("tile",[]).append(
                        float(tile_start.elapsed_time(tile_end))
                    )
                    stage_samples.setdefault("merge_tail",[]).append(
                        float(tile_end.elapsed_time(merge_end))
                    )
                    del st,parts,merged
                med={k:float(statistics.median(vals))
                     for k,vals in stage_samples.items()}
                selected["arms"][arm]={
                    "gpu_stream_stage_interval_medians_ms":med,
                    "sum_gpu_stream_intervals_ms":float(sum(med.values())),
                    "synchronized_cpu_wall_median_ms":float(
                        statistics.median(cpu_wall)
                    ),
                }
                print(f"DIAG heads={count} window={name} "
                      f"arm={arm} GPU-stage-intervals={med}",flush=True)
                del ws,active_ws
            del window,tail
        del packed,native
    report["completed"]=True
    report["notes"].append(
        "Stage events include CPU enqueue gaps/idle GPU intervals; "
        "do not label them pure kernel durations."
    )
    report["notes"].append(
        "Diagnostic status is not a new numerical parity or video PASS."
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tokens",type=int,default=18723)
    parser.add_argument("--heads",type=int,default=56)
    parser.add_argument("--head-cases",default="4,56")
    parser.add_argument("--group-cap",type=int,default=11)
    parser.add_argument("--tiles",type=int,default=16)
    parser.add_argument("--rounds",type=int,default=2)
    parser.add_argument("--warmups",type=int,default=1)
    parser.add_argument("--seed",type=int,default=20261112)
    parser.add_argument("--min-free-gib",type=float,default=4.0)
    parser.add_argument("--diagnose-only",action="store_true")
    parser.add_argument("--out",type=Path,default=Path(
        r"D:\ComfyOutput\H3\math_probe\flat_direct_qkv_ab.json"))
    args=parser.parse_args()
    try:
        args.head_cases=[int(x) for x in args.head_cases.split(",")]
    except ValueError:
        parser.error("Invalid --head-cases")
    if (not args.head_cases or len(set(args.head_cases))!=len(args.head_cases)
        or any(v<1 or v>args.heads for v in args.head_cases)
        or any(v<=0 for v in (args.heads,args.tokens,args.group_cap,
                               args.tiles,args.rounds,args.warmups))):
        parser.error("Invalid positive arguments or head-cases")
    report={
        "experiment":"flat_direct_qkv_strided_v2_torch_mu",
        "completed":False,"quality_coefficient_used":False,
        "fixed_math":True,"real_model":False,
        "cases":[],"errors":[],
        "notes":[
            f"All {len(ARMS)} arms use the same BF16 synthetic native QKV.",
            "Timing: GPU event medians per group, not whole-video wall.",
            "JIT cold cache-build cost reported separately.",
            "No model-resident or production acceptance implied.",
            "Dynamic group-aligned last window; no hardcoded tail tile count.",
            "Only real rows compared with legacy full-pad Flat on each short window.",
            "Short window aliases the original per-head workspace; no extra GPU cache.",
            "native_torch_mu avoids a materialized FP32 Q and uses ATen FP32 mean.",
            "Both native-direct Q reductions are opt-in; main default gather unchanged.",
            "Graph and persistent Direct-Flow DISPATCH are UNRUN."
        ]
    }
    try:
        if args.diagnose_only:
            report["experiment"]="flat_direct_qkv_stage_diagnostic_v2_torch_mu"
            diagnose_stages(args,report)
        else:
            run(args,report)
    except BaseException as exc:
        report["errors"].append({
            "type":type(exc).__name__,
            "message":str(exc)[:1400],
            "traceback_tail":traceback.format_exc()[-2500:]
        })
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",args.out,flush=True)
    print("STATUS:",("DIAG_COMPLETE" if args.diagnose_only else "PASS") if report["completed"] else "FAIL",flush=True)
    return 0 if report["completed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
