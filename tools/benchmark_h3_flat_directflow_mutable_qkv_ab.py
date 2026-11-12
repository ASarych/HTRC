"""Dynamic-QKV Direct-Flow proof: one graph per WINDOW GEOMETRY, changed Q/K/V.

New post-RoPE BF16 Q/K/V source addresses A and B alternate every replay.
Before each attention call, K/V are copied to a persistent 11-head staging
buffer and their K8/KS and exact 1..15 key tail updated. Each Q window is
copied to its own persistent query staging buffer and a captured graph is
replayed (full/short, no per-window recapture). Eager runs the SAME math
directly from the current bank. This is NOT an H3 model-resident benchmark.
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

CHECK_STATES=("MU","IDX","EXCL")


def geometry_key(window):
    """Reuse a graph only when both shape and true-row count are identical."""
    return (window.tile_count,window.valid_rows)


def snapshot_state(state):
    return {name:state[name].clone() for name in CHECK_STATES}


def copy_query_window(q_bhsd,geometry,target):
    """Stage current (not captured) Q; repeat only the final true query row."""
    rows=geometry.valid_rows
    if target.shape!=(q_bhsd.shape[1],geometry.padded_rows,128):
        raise ValueError("Target query staging buffer has wrong shape")
    src=q_bhsd[0,:,geometry.start:geometry.end,:]
    target[:,:rows,:].copy_(src)
    if rows<geometry.padded_rows:
        target[:,rows:,:].copy_(
            src[:,-1:,:].expand(-1,geometry.padded_rows-rows,-1)
        )


def one_window(win,heads,cfg,tr,kv,ws,tail,geometry,qmode):
    state,*_=batch.prepare_batched_head_states(
        win,heads,cfg,tr,tile_count=geometry.tile_count,
        quant_backend="triton",input_cast="gpu",
        kv_cache=kv,query_workspace=ws,
        valid_query_rows=geometry.valid_rows,query_staging=qmode
    )
    partial,_=batch.launch_prepared_tile_batch(
        state,cfg,tr,buffers=ws["_partials"]
    )
    merged,_=batch.launch_prepared_merge_batch(
        state,cfg,tr,partial,tails=tail,
        output=ws["_out"],merge_rows=32
    )
    return state,merged


def write_window(merged,geometry,out):
    h=int(out.shape[1])
    selected=merged.reshape(h,geometry.padded_rows,128)[:,:geometry.valid_rows]
    out[geometry.start:geometry.end,:,:].copy_(selected.permute(1,0,2))


def create_input_banks(args):
    generator=torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    shape=(1,args.group_heads,args.tokens,128)
    def bank():
        return tuple(torch.randn(shape,device="cuda",dtype=torch.bfloat16,
                                 generator=generator) for _ in range(3))
    return bank(),bank()


def build_stage(args,bank_a,geometries,cfg,tr):
    h=args.group_heads
    ids=list(range(h))
    # Persistent same-address K and V. Q has one same-address buffer per
    # active window geometry, not the whole source sequence length.
    stage_k=torch.empty_like(bank_a[1])
    stage_v=torch.empty_like(bank_a[2])
    stage_k.copy_(bank_a[1])
    stage_v.copy_(bank_a[2])
    kv_capture={
        "k_khd":stage_k.detach()[0].permute(1,0,2),
        "v_khd":stage_v.detach()[0].permute(1,0,2),
        "raw_head_index":ids,
    }
    kv_cache=batch.prepare_gpu_head_kv_cache(
        kv_capture,ids,tr,native_direct=True
    )
    tails=shared._tail(kv_capture,ids)
    return {
        "k":stage_k,"v":stage_v,
        "capture":kv_capture,"cache":kv_cache,
        "tail":tails,"ids":ids,"registry":{},
    }


def update_kv_stage(stage,bank,tr):
    """All updates submitted before graph replay on the SAME CUDA stream."""
    stage["k"].copy_(bank[1])
    stage["v"].copy_(bank[2])
    cache=stage["cache"]
    kbf=cache["KBF"]
    h,n,d=map(int,kbf.shape)
    br=8
    tr._quant_key_int8_native[
        (tr.triton.cdiv(h*n,br),)
    ](kbf,cache["K"].view(h*n,d),cache["KS"].view(h*n),
      n,d,h,br,*map(int,kbf.stride()),num_warps=4)
    if stage["tail"] is not None:
        tail_k,tail_v=stage["tail"]
        ntail=int(tail_k.shape[1])
        tail_k.copy_(stage["k"][0,:,-ntail:,:])
        tail_v.copy_(stage["v"][0,:,-ntail:,:])


def capture_graphs(args,bank_a,geometries,stage,cfg,tr,report):
    keys={}
    for geometry in geometries:
        keys[geometry_key(geometry)]=geometry
    before=torch.cuda.memory_reserved()
    before_free=torch.cuda.mem_get_info()[0]
    torch.cuda.reset_peak_memory_stats()
    capture_total_ms=0.0
    current=torch.cuda.current_stream()
    for signature,geom in keys.items():
        qbuf=torch.empty((args.group_heads,geom.padded_rows,128),
                         device="cuda",dtype=torch.bfloat16)
        copy_query_window(bank_a[0],geom,qbuf)
        win=dict(stage["capture"])
        win["q_qhd"]=qbuf.permute(1,0,2)
        ws=shared.alloc_workspace(win,args.group_heads,geom.tile_count,cfg)
        def perform():
            return one_window(
                win,stage["ids"],cfg,tr,stage["cache"],ws,
                stage["tail"],geom,args.qmode
            )
        warm=torch.cuda.Stream(device=qbuf.device)
        warm.wait_stream(current)
        with torch.cuda.stream(warm):
            for _ in range(args.warmups):
                st,out=perform()
                del st,out
        current.wait_stream(warm)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        started=time.perf_counter()
        with torch.cuda.graph(graph):
            states,output=perform()
        torch.cuda.synchronize()
        capture_ms=(time.perf_counter()-started)*1000
        capture_total_ms+=capture_ms
        stage["registry"][signature]={
            "graph":graph,"q_buffer":qbuf,"window_capture":win,
            "workspace":ws,"states":states,"out":output,
            "capture_ms":capture_ms,
        }
    after=torch.cuda.memory_reserved()
    after_free=torch.cuda.mem_get_info()[0]
    peak=torch.cuda.max_memory_reserved()
    extra_peak=max(0,peak-before)/(1024*1024)
    memory={
        "before_reserved_mib":before/(1024*1024),
        "after_reserved_mib":after/(1024*1024),
        "signed_reserved_delta_mib":(after-before)/(1024*1024),
        "peak_reserved_mib":peak/(1024*1024),
        "peak_above_precapture_reserved_mib":extra_peak,
        "before_free_mib":before_free/(1024*1024),
        "after_free_mib":after_free/(1024*1024),
        "live_stage_q_bytes":sum(x["q_buffer"].numel()*2
                                for x in stage["registry"].values()),
        "live_stage_kv_bytes":stage["k"].numel()*2+stage["v"].numel()*2,
        "live_graph_count":len(stage["registry"]),
        "captures_wall_ms":capture_total_ms,
    }
    if extra_peak>args.max_extra_reserved_mib:
        raise MemoryError(
            f"Graph capture peak reserved +{extra_peak:.1f} MiB exceeds "
            f"--max-extra-reserved-mib={args.max_extra_reserved_mib}"
        )
    report["graph_memory"]=memory
    return capture_total_ms


def run_eager(bank,args,geometries,cfg,tr,ws,out,inspect=False):
    ids=list(range(args.group_heads))
    cap=shared.capture_from_bhsd(*bank,args.group_heads)
    kv=batch.prepare_gpu_head_kv_cache(cap,ids,tr,native_direct=True)
    tail=shared._tail(cap,ids)
    debug={
        "K8":kv["K"].clone(),"KS":kv["KS"].clone(),
        "states":{}
    } if inspect else None
    for wi,geom in enumerate(geometries):
        src=cap["q_qhd"][geom.start:geom.end]
        if geom.valid_rows<geom.padded_rows:
            src=torch.cat(
                (src,src[-1:].expand(geom.padded_rows-geom.valid_rows,-1,-1)),
                dim=0
            )
        win=dict(cap)
        win["q_qhd"]=src
        active_ws=windows.compact_workspace_view(
            ws,args.group_heads,geom.tile_count,cfg
        )
        st,merged=one_window(
            win,ids,cfg,tr,kv,active_ws,tail,geom,args.qmode
        )
        write_window(merged,geom,out)
        if inspect and (wi==0 or wi==len(geometries)-1):
            debug["states"][str(wi)]=snapshot_state(st)
    return debug


def run_replay(bank,args,geometries,stage,tr,out,inspect=False):
    update_kv_stage(stage,bank,tr)
    debug={"states":{}} if inspect else None
    for wi,geom in enumerate(geometries):
        reg=stage["registry"][geometry_key(geom)]
        copy_query_window(bank[0],geom,reg["q_buffer"])
        reg["graph"].replay()
        write_window(reg["out"],geom,out)
        if inspect and (wi==0 or wi==len(geometries)-1):
            debug["states"][str(wi)]=snapshot_state(reg["states"])
    if inspect:
        debug["K8"]=stage["cache"]["K"].clone()
        debug["KS"]=stage["cache"]["KS"].clone()
    return debug


def compare_debug(eager,graph,eager_out,graph_out):
    checks={}
    for name in ("K8","KS"):
        checks[name]=bool(torch.equal(eager[name],graph[name]))
    for window,states in eager["states"].items():
        for name,expected in states.items():
            checks[f"{window}:{name}"]=bool(
                torch.equal(expected,graph["states"][window][name])
            )
    checks["OUTPUT_FULL"]=bool(torch.equal(eager_out,graph_out))
    if not all(checks.values()):
        failures=[name for name,passed in checks.items() if not passed]
        absmax=float((eager_out-graph_out).abs().amax().item())
        raise AssertionError(
            f"Graph used stale data or mismatched eager: {failures}, "
            f"output_max_abs={absmax}"
        )
    return checks


def measure(invoke):
    a=torch.cuda.Event(enable_timing=True)
    b=torch.cuda.Event(enable_timing=True)
    a.record()
    start=time.perf_counter()
    invoke()
    cpu_submit_ms=(time.perf_counter()-start)*1000
    b.record()
    b.synchronize()
    total_ms=(time.perf_counter()-start)*1000
    return {
        "gpu_event_ms":float(a.elapsed_time(b)),
        "cpu_submit_ms":float(cpu_submit_ms),
        "cpu_sync_wall_ms":float(total_ms),
    }


def get_medians(samples):
    return {key:statistics.median(x[key] for x in samples)
            for key in ("gpu_event_ms","cpu_submit_ms","cpu_sync_wall_ms")}


def main_run(args,report):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free,total=torch.cuda.mem_get_info()
    if free<args.min_free_gib*1024**3:
        raise MemoryError("Insufficient free VRAM for mutable-input probe")
    report["gpu"]={
        "device":torch.cuda.get_device_name(),
        "free_before_gib":free/1024**3,
        "total_gib":total/1024**3,
    }
    tr=live._triton()
    tr._dll()
    cfg=shared.flat_config("fast",args.tiles)
    group=int(cfg.get("kern",{}).get("group",1))
    if (int(cfg["exact_leaves"])!=256 or group!=1):
        raise ValueError("Only fixed fast 256 leaves / center group1")
    geoms=list(windows.iter_query_windows(
        args.tokens,args.tiles,
        tile_rows=int(cfg["tile"]),center_group=group
    ))
    sources=create_input_banks(args)
    stage=build_stage(args,sources[0],geoms,cfg,tr)
    capture_ms=capture_graphs(
        args,sources[0],geoms,stage,cfg,tr,report
    )
    # The eager workspace planner needs a complete Q/K/V capture.
    # stage["capture"] intentionally contains K/V only; using it here
    # would raise KeyError("q_qhd") before parity checks.
    eager_source_capture=shared.capture_from_bhsd(
        *sources[0],args.group_heads
    )
    eager_ws=shared.alloc_workspace(
        eager_source_capture,args.group_heads,args.tiles,cfg
    )
    eager_out=torch.empty((args.tokens,args.group_heads,128),
                          device="cuda",dtype=torch.float32)
    graph_out=torch.empty_like(eager_out)
    report["config"]={
        "tokens":args.tokens,"heads_per_group":args.group_heads,
        "window_tiles":args.tiles,"tile_rows":int(cfg["tile"]),
        "query_windows":len(geoms),
        "graph_geometries":[
            {"active_tiles":k[0],"valid_rows":k[1]}
            for k in stage["registry"]
        ],
        "q_mode":args.qmode,"kv_mode":"native_direct",
        "quant_key_refresh_each_call":True,
        "tail_refresh_each_call":True,
        "q_window_refresh_each_replay":True,
        "native_qkv_banks":2,
        "source_bank_total_bytes":sum(
            tensor.numel()*tensor.element_size()
            for bank in sources for tensor in bank
        ),
        "exact_leaves":256,"center_group":group,
        "rounds":args.rounds,"warmups":args.warmups,
        "source_geometry":"synthetic_11_or_custom_distinct_bf16_heads",
    }
    report["capture_wall_ms"]=capture_ms
    checks_per_bank={}
    saved_output=None
    for bank_index,source in enumerate(sources):
        a=run_eager(source,args,geoms,cfg,tr,eager_ws,eager_out,
                    inspect=True)
        b=run_replay(source,args,geoms,stage,tr,graph_out,
                     inspect=True)
        torch.cuda.synchronize()
        checks_per_bank[str(bank_index)]=compare_debug(
            a,b,eager_out,graph_out
        )
        if saved_output is None:
            saved_output=graph_out.clone()
        else:
            if torch.equal(saved_output,graph_out):
                raise AssertionError(
                    "Q/K/V banks differed but replay output did NOT change"
                )
        del a,b
    report["parity_checks"]=checks_per_bank
    report["different_banks_produce_different_outputs"]=True
    report["cases"].append({
        "head_count":args.group_heads,
        "window_count":len(geoms),
        "graph_count":len(stage["registry"]),
        "different_qkv_banks":2,
        "eager_vs_replay_bit_exact":True,
        "distinct_bank_outputs":True,
        "status":"PASS_NUMERIC",
    })
    # Measure entire 19-window attention call; refresh K8/KS once per
    # call, not per window. Never extrapolate single-window graph replay.
    times={"fresh_native_eager":[],"mutable_staged_graph":[]}
    for r in range(args.rounds):
        source=sources[r%len(sources)]
        order=("fresh_native_eager","mutable_staged_graph") if r%2==0 else (
            "mutable_staged_graph","fresh_native_eager"
        )
        for arm in order:
            if arm=="fresh_native_eager":
                fn=lambda:run_eager(
                    source,args,geoms,cfg,tr,eager_ws,eager_out
                )
            else:
                fn=lambda:run_replay(
                    source,args,geoms,stage,tr,graph_out
                )
            times[arm].append(measure(fn))
    report["full_attention_call_group_median_ms"]={
        arm:get_medians(vals) for arm,vals in times.items()
    }
    e=report["full_attention_call_group_median_ms"]["fresh_native_eager"]
    g=report["full_attention_call_group_median_ms"]["mutable_staged_graph"]
    report["speedups"]={
        "gpu_event":e["gpu_event_ms"]/g["gpu_event_ms"],
        "cpu_sync_wall":e["cpu_sync_wall_ms"]/g["cpu_sync_wall_ms"],
        "cpu_submit":e["cpu_submit_ms"]/g["cpu_submit_ms"],
    }
    wall_gain=e["cpu_sync_wall_ms"]-g["cpu_sync_wall_ms"]
    report["capture_break_even_full_attention_calls_by_sync_wall"]=(
        math.ceil(capture_ms/wall_gain) if wall_gain>0 else None
    )
    torch.cuda.synchronize()
    report["completed"]=True


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokens",type=int,default=18723)
    p.add_argument("--group-heads",type=int,default=11)
    p.add_argument("--tiles",type=int,default=16)
    p.add_argument("--qmode",choices=("native_torch_mu","native_direct"),
                   default="native_torch_mu")
    p.add_argument("--rounds",type=int,default=12)
    p.add_argument("--warmups",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261112)
    p.add_argument("--min-free-gib",type=float,default=4)
    p.add_argument("--max-extra-reserved-mib",type=float,default=512)
    p.add_argument("--out",type=Path,default=Path(
        r"D:\ComfyOutput\H3\math_probe\flat_directflow_mutable_qkv_ab.json"
    ))
    args=p.parse_args()
    if (args.tokens<16 or args.group_heads<1 or args.tiles<1
        or args.rounds<4 or args.warmups<3
        or args.max_extra_reserved_mib<=0 or args.min_free_gib<=0):
        p.error("Invalid positive arguments")
    report={
        "experiment":"flat_directflow_mutable_qkv_full_group_v1",
        "completed":False,"cases":[],"errors":[],
        "fixed_math":True,"exact_leaves":256,
        "quality_coefficient_used":False,
        "real_model":False,"distinct_synthetic_input_banks":2,
        "notes":[
            "ALL Q/K/V change between source banks A and B; output must change.",
            "Graph addresses are fixed STAGING buffers, NOT fixed source addresses.",
            "K8/KS refreshed once per attention call; tail K/V updated too.",
            "One full-window graph + optional short-window graph are reused across every query window.",
            "Timing covers 19 windows per 11-head group, K/V copy, Q copies, K8 quant, output copies; graph capture excluded and reported separately.",
            "Fresh eager consumes native input directly and rebuilds its K8/KS each call; both arms use identical Flat math.",
            "Eager-to-graph equality is bit exact for full output, K8, KS and selected MU/IDX/EXCL.",
            "CUDA Graph private-pool and model-resident headroom still require real H3 evaluation.",
            "This report is NOT a model/video benchmark or production acceptance."
        ]
    }
    try:
        with torch.inference_mode():
            main_run(args,report)
    except BaseException as exc:
        report["errors"].append({
            "type":type(exc).__name__,
            "message":str(exc)[:1500],
            "traceback_tail":traceback.format_exc()[-2600:],
        })
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",args.out,flush=True)
    print("STATUS:","PASS" if report["completed"] else "BLOCKED_OR_FAIL",
          flush=True)
    return 0 if report["completed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
