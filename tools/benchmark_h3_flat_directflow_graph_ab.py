"""Research-only Direct-Flow static-window CUDA Graph vs *same math* eager.

Graph replay reuses fixed CUDA addresses. It does not prove model-resident
reuse across new QKV allocations, multiple H3 layers or denoising steps.
No Graph integration into the live backend or production dispatch.
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


def window_capture(cap, geometry):
    q=cap["q_qhd"][geometry.start:geometry.end]
    if geometry.valid_rows<geometry.padded_rows:
        q=torch.cat(
            (q,q[-1:].expand(geometry.padded_rows-geometry.valid_rows,-1,-1)),
            dim=0
        )
    win=dict(cap)
    win["q_qhd"]=q
    return win


def one_call(win,heads,cfg,tr,kv,workspace,tail,geometry,q_mode):
    state,*_=batch.prepare_batched_head_states(
        win,heads,cfg,tr,tile_count=geometry.tile_count,
        quant_backend="triton",input_cast="gpu",
        kv_cache=kv,query_workspace=workspace,
        valid_query_rows=geometry.valid_rows,
        query_staging=q_mode
    )
    parts,_=batch.launch_prepared_tile_batch(
        state,cfg,tr,buffers=workspace["_partials"]
    )
    out,_=batch.launch_prepared_merge_batch(
        state,cfg,tr,parts,tails=tail,
        output=workspace["_out"],merge_rows=32
    )
    return state,out


def measure_once(invoke):
    a=torch.cuda.Event(enable_timing=True)
    b=torch.cuda.Event(enable_timing=True)
    a.record()
    begin=time.perf_counter()
    invoke()
    submitted=time.perf_counter()
    b.record()
    b.synchronize()
    done=time.perf_counter()
    return {
        "gpu_event_ms":float(a.elapsed_time(b)),
        "cpu_submit_ms":float((submitted-begin)*1000),
        "cpu_sync_wall_ms":float((done-begin)*1000),
    }


def medians(samples):
    return {
        key:statistics.median(sample[key] for sample in samples)
        for key in ("gpu_event_ms","cpu_submit_ms","cpu_sync_wall_ms")
    }


def probe_geometry(cap,args,cfg,tr,kv,ids,geom):
    win=window_capture(cap,geom)
    tails=shared._tail(cap,ids)
    ws_eager=shared.alloc_workspace(cap,len(ids),geom.tile_count,cfg)
    ws_graph=shared.alloc_workspace(cap,len(ids),geom.tile_count,cfg)

    def eager():
        return one_call(
            win,ids,cfg,tr,kv,ws_eager,tails,geom,args.q_mode
        )

    def capture_work():
        return one_call(
            win,ids,cfg,tr,kv,ws_graph,tails,geom,args.q_mode
        )

    with torch.inference_mode():
        torch.cuda.synchronize()
        ref_state,ref_output=eager()
        reference={
            "IDX":ref_state["IDX"].clone(),
            "EXCL":ref_state["EXCL"].clone(),
            "MU":ref_state["MU"].clone(),
            "OUT":ref_output.clone(),
        }
        torch.cuda.synchronize()

        # Warmup on a side stream is required before capture.
        start_stream=torch.cuda.current_stream()
        warm=torch.cuda.Stream(device=cap["q_qhd"].device)
        warm.wait_stream(start_stream)
        with torch.cuda.stream(warm):
            for _ in range(args.warmups):
                s,o=capture_work()
                del s,o
        start_stream.wait_stream(warm)
        torch.cuda.synchronize()

        reserved_before=torch.cuda.memory_reserved()
        allocated_before=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        graph=torch.cuda.CUDAGraph()
        capture_begin=time.perf_counter()
        # Fixed source tensors, shape and addresses across replay.
        with torch.cuda.graph(graph):
            static_state,static_out=capture_work()
        torch.cuda.synchronize()
        capture_wall_ms=(time.perf_counter()-capture_begin)*1000
        reserved_after=torch.cuda.memory_reserved()
        allocated_after=torch.cuda.memory_allocated()
        graph_extra_reserved_mib=max(
            0.0,(reserved_after-reserved_before)/(1024*1024)
        )
        if graph_extra_reserved_mib>args.max_extra_reserved_mib:
            raise MemoryError(
                f"Graph pool extra reserved {graph_extra_reserved_mib:.2f} MiB "
                f"> --max-extra-reserved-mib={args.max_extra_reserved_mib}"
            )
        graph.replay()
        torch.cuda.synchronize()
        equality={}
        for key,actual in (
            ("IDX",static_state["IDX"]),
            ("EXCL",static_state["EXCL"]),
            ("MU",static_state["MU"]),
            ("OUT",static_out),
        ):
            same=bool(torch.equal(reference[key],actual))
            maximum=0.0 if same else float(
                (reference[key].float()-actual.float()).abs().amax().item()
            )
            equality[key]={"bit_exact":same,"max_abs":maximum}
        if not all(x["bit_exact"] for x in equality.values()):
            raise AssertionError(f"Eager vs graph output/state mismatch: {equality}")

        samples={"eager":[],"graph":[]}
        # Alternate order to avoid systematic warm/cool bias.
        for round_index in range(args.rounds):
            order=("eager","graph") if round_index%2==0 else ("graph","eager")
            for arm in order:
                samples[arm].append(measure_once(
                    eager if arm=="eager" else graph.replay
                ))
        torch.cuda.synchronize()
        m={arm:medians(values) for arm,values in samples.items()}
        memory={
            "allocated_before_capture_mib":allocated_before/(1024*1024),
            "allocated_after_capture_mib":allocated_after/(1024*1024),
            "reserved_before_capture_mib":reserved_before/(1024*1024),
            "reserved_after_capture_mib":reserved_after/(1024*1024),
            "graph_extra_reserved_mib":graph_extra_reserved_mib,
            "peak_allocated_in_capture_and_replay_mib":
                torch.cuda.max_memory_allocated()/(1024*1024),
        }
        speed=m["eager"]["gpu_event_ms"]/m["graph"]["gpu_event_ms"]
        extra_wall=m["eager"]["cpu_sync_wall_ms"]-m["graph"]["cpu_sync_wall_ms"]
        return {
            "head_count":len(ids),
            "window_start":geom.start,
            "window_kind":"full" if geom.valid_rows==args.tiles*64 else "short",
            "real_rows":geom.valid_rows,
            "active_tiles":geom.tile_count,
            "padded_rows":geom.padded_rows,
            "input_addresses_static":True,
            "graph_capture_wall_ms":float(capture_wall_ms),
            "bit_exact_eager_vs_replay":equality,
            "timing_median_ms":m,
            "graph_speedup_gpu_event":float(speed),
            "graph_speedup_sync_wall":float(
                m["eager"]["cpu_sync_wall_ms"]/m["graph"]["cpu_sync_wall_ms"]
            ),
            "capture_break_even_replays_by_sync_wall":(
                math.ceil(capture_wall_ms/extra_wall)
                if extra_wall>0 else None
            ),
            "memory":memory,
        }


def run(args,report):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA RTX GPU required")
    free,total=torch.cuda.mem_get_info()
    report["gpu"]={
        "device":torch.cuda.get_device_name(),
        "free_before_gib":free/(1024**3),
        "total_gib":total/(1024**3),
    }
    if free < args.min_free_gib*1024**3:
        raise MemoryError("Free VRAM below --min-free-gib")

    torch.manual_seed(args.seed)
    shape=(1,args.heads,args.tokens,128)
    q=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
    k=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
    v=torch.randn(shape,device="cuda",dtype=torch.bfloat16)
    cap=shared.capture_from_bhsd(q,k,v,args.heads)
    cfg=shared.flat_config("fast",args.tiles)
    if int(cfg["exact_leaves"])!=256 or int(cfg["kern"]["group"])!=1:
        raise ValueError("Only fixed fast 256-leaf / group1 math")
    tr=live._triton()
    tr._dll()
    heads=list(range(args.group_heads))
    cache=batch.prepare_gpu_head_kv_cache(
        cap,heads,tr,native_direct=True
    )
    geometries=list(windows.iter_query_windows(
        args.tokens,args.tiles,tile_rows=int(cfg["tile"]),
        center_group=int(cfg["kern"]["group"])
    ))
    requested=[geometries[0]]
    if args.window in ("short","both"):
        if geometries[-1].tile_count==args.tiles:
            if args.window=="short":
                raise ValueError("No distinct short window in this sequence")
        else:
            if args.window=="short":
                requested=[]
            requested.append(geometries[-1])
    report["config"]={
        "tokens":args.tokens,"heads":args.heads,
        "group_heads":args.group_heads,
        "tile_capacity":args.tiles,
        "query_mode":args.q_mode,
        "kv_mode":"native_direct",
        "exact_leaves":256,"center_group":1,
        "warmups":args.warmups,"rounds":args.rounds,
        "max_extra_reserved_mib":args.max_extra_reserved_mib,
        "native_stride":list(cap["q_qhd"].stride()),
    }
    for geometry in requested:
        result=probe_geometry(cap,args,cfg,tr,cache,heads,geometry)
        report["cases"].append(result)
        print(
            f"GRAPH_AB heads={result['head_count']} "
            f"rows={result['real_rows']} tiles={result['active_tiles']} "
            f"gpu_speedup={result['graph_speedup_gpu_event']:.4f}x",
            flush=True
        )
    report["completed"]=len(report["cases"])==len(requested)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokens",type=int,default=18723)
    p.add_argument("--heads",type=int,default=56)
    p.add_argument("--group-heads",type=int,default=11)
    p.add_argument("--tiles",type=int,default=16)
    p.add_argument("--window",choices=("full","short","both"),default="both")
    p.add_argument("--q-mode",choices=("native_direct","native_torch_mu"),
                   default="native_torch_mu")
    p.add_argument("--rounds",type=int,default=25)
    p.add_argument("--warmups",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261112)
    p.add_argument("--min-free-gib",type=float,default=4.0)
    p.add_argument("--max-extra-reserved-mib",type=float,default=512.0)
    p.add_argument("--out",type=Path,default=Path(
        r"D:\ComfyOutput\H3\math_probe\flat_directflow_graph_ab.json"
    ))
    args=p.parse_args()
    if (args.tokens<16 or args.heads<1
        or args.group_heads<1 or args.group_heads>args.heads
        or args.tiles<1 or args.rounds<3 or args.warmups<3
        or args.min_free_gib<=0 or args.max_extra_reserved_mib<=0):
        p.error("Invalid positive arguments or group-head count")
    report={
        "experiment":"flat_directflow_cuda_graph_static_window_v1",
        "completed":False,"math":"fixed_fast_256_leaves",
        "quality_coefficient_used":False,
        "real_model":False,"static_qkv_addresses":True,
        "cases":[],"errors":[],
        "notes":[
            "This is an opt-in SINGLE-group STATIC-pointer CUDA Graph feasibility probe.",
            "No evidence graph capture can be reused between H3 blocks or denoising steps.",
            "All timing uses CUDA events on identical native QKV and same math.",
            "Graph capture wall and extra reserved memory reported separately.",
            "A replay timing alone omits input copy and graph recapture costs.",
            "No production or live-backend dispatch integration.",
        ]
    }
    try:
        run(args,report)
    except BaseException as exc:
        report["errors"].append({
            "type":type(exc).__name__,
            "message":str(exc)[:1500],
            "traceback_tail":traceback.format_exc()[-2600:],
        })
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",args.out,flush=True)
    print("STATUS:","DIAG_PASS" if report["completed"] else "BLOCKED_OR_FAIL",
          flush=True)
    return 0 if report["completed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
