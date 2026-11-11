"""Measure TRUE streaming head-group Tile memory and partial parity.

A/B uses pinned validated BEST Tile per head vs ONE universal
head-flattened Tile launch PER group of arbitrary size. Data come
from a trusted saved native capture; its four unique heads may be
explicitly repeated as a launch/VRAM proxy for 56 logical jobs.

This benchmark stages only one group on GPU at a time and measures
torch.cuda.max_memory_allocated. CPU->GPU preparation and top-k are
performed for each group but are not counted in the Tile CUDA-event
times. Merge and end-to-end H3 generation are not tested.

Use --budget-mib to simulate a tight available VRAM budget (always
capped by torch.cuda.mem_get_info). An actual CUDA OOM reduces the
group size by half and retries; other failures never silently retry.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location(
    "_h3_streamed_tile_base",ROOT/"tools"/"benchmark_h3_flat_tile_56head_launch.py"
)
base=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=base
SPEC.loader.exec_module(base)

PLAN_SPEC=importlib.util.spec_from_file_location(
    "_h3_streamed_tile_planner",ROOT/"h3"/"htrc_flat_head_batch.py"
)
stage=importlib.util.module_from_spec(PLAN_SPEC)
sys.modules[PLAN_SPEC.name]=stage
PLAN_SPEC.loader.exec_module(stage)


def measure_group(states, config, old_tr, new_tr, nb, kx, tiles,
                  warmups, rounds, batch, tolerance):
    H=int(states["Q"].shape[0])
    T=int(config["tile"])
    D=int(states["Q"].shape[-1])
    ns=int(config["splits"])
    kern=config["kern"]
    bn=int(kern.get("tile_bn",32))
    warps=int(kern.get("tile_warps",4))
    stages=int(kern.get("tile_stages",1))
    near_bn=int(kern.get("tile_near_bn") or bn)
    keys=int(states["K"].shape[1])
    kw=dict(
        T=T,D=D,BN=bn,BN_NEAR=near_bn,LEAF=16,PREC="tf32",
        DOT=1,GR=1,FP8=0,TAIL=0,RB=1,I8=1,
        PASS=0,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0,
        FAR_PV_ACC=0,NEAR_PV_ACC=0,
        num_warps=warps,num_stages=stages
    )
    lps=old_tr._split(nb,ns,bn)
    eps=old_tr._split(kx*16,ns,bn)
    scale=1.0/math.sqrt(D)
    partial=(H,tiles,ns,T)
    ref_buf={
        "pm":torch.empty(partial,device="cuda",dtype=torch.float32),
        "pl":torch.empty(partial,device="cuda",dtype=torch.float32),
        "acc":torch.empty(partial+(D,),device="cuda",dtype=torch.float32),
    }
    new_buf={name:torch.empty_like(t) for name,t in ref_buf.items()}

    def pinned_call():
        kernel=None
        for h in range(H):
            kernel=old_tr._tile_kernel[(tiles,ns,1)](
                states["Q"][h],states["MU"][h],states["G"][h],
                states["LZ"][h],states["VB"][h],states["EXCL"][h],
                states["IDX"][h],states["K"][h],states["V"][h],
                ref_buf["pm"][h],ref_buf["pl"][h],ref_buf["acc"][h],
                states["SG"][h],states["LZ"][h],
                states["LZ"][h],states["LZ"][h],states["KS"][h],
                nb,kx,lps,eps,0,scale,**kw
            )
        return kernel

    def batched_call():
        _,compiled=stage.launch_prepared_tile_batch(
            states,config,new_tr,buffers=new_buf
        )
        return compiled

    original_kernel=pinned_call()
    grouped_kernel=batched_call()
    torch.cuda.synchronize()
    parity={}
    for name in ("pm","pl","acc"):
        ref=ref_buf[name]
        opt=new_buf[name]
        delta=opt.float()-ref.float()
        rel=float(delta.norm()/ref.float().norm().clamp_min(1e-20))
        finite=bool(torch.isfinite(ref).all() and torch.isfinite(opt).all())
        parity[name]={
            "relative_l2":rel,"max_abs":float(delta.abs().max()),
            "finite":finite,"pass":finite and rel<=tolerance,
        }
        del delta
    passed=all(v["pass"] for v in parity.values())
    result={
        "heads":H,"tiles_per_head":tiles,
        "tile_launches_pinned":H,"tile_launches_batched":1,
        "parity":parity,"parity_pass":passed,
        "compiled_resources":{
            "pinned":base.resources(original_kernel),
            "grouped":base.resources(grouped_kernel),
        },
        "timings":None,
    }
    if passed:
        for _ in range(warmups):
            pinned_call()
            batched_call()
        torch.cuda.synchronize()
        samples={"pinned":[],"grouped":[]}
        for _ in range(rounds):
            for who in ("pinned","grouped","grouped","pinned"):
                fn=pinned_call if who=="pinned" else batched_call
                samples[who].append(base.timed_event(fn,batch))
        med={who:statistics.median(v) for who,v in samples.items()}
        result["timings"]={
            "median_tile_ms":med,
            "speedup_pinned_over_grouped":med["pinned"]/med["grouped"],
            "samples_ms":samples,
        }
    del ref_buf,new_buf
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--capture",
        default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt"
    )
    ap.add_argument("--heads",type=int,default=56)
    ap.add_argument("--repeat-available-heads",action="store_true")
    ap.add_argument("--mode",default="fast",choices=("fast","balanced"))
    ap.add_argument("--tiles",type=int,default=16)
    ap.add_argument("--budget-mib",type=float,default=None,
                    help="Optional upper bound on observed free GPU memory in MiB")
    ap.add_argument("--reserve-mib",type=float,default=256.0)
    ap.add_argument("--fixed-temp-mib",type=float,default=128.0)
    ap.add_argument("--safety-factor",type=float,default=1.5)
    ap.add_argument("--max-heads-per-group",type=int,default=None)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--warmups",type=int,default=2)
    ap.add_argument("--tolerance",type=float,default=1e-4)
    ap.add_argument("--out",default=r"D:\ComfyOutput\H3\math_probe")
    args=ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("CUDA GPU required")
    if min(args.heads,args.tiles,args.rounds,args.batch,args.warmups)<1:
        ap.error("Heads, tiles, rounds, batch and warmups must be positive")
    if any(not math.isfinite(x) or x<0 for x in
           (args.reserve_mib,args.fixed_temp_mib)):
        ap.error("Reserve and temporary MiB must be finite and nonnegative")
    if args.budget_mib is not None and (
        not math.isfinite(args.budget_mib) or args.budget_mib<=0
    ):
        ap.error("--budget-mib must be finite and positive")
    path=Path(args.capture)
    if not path.is_file():
        ap.error("Trusted capture file missing")
    cap=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    if not isinstance(cap,dict) or any(
        name not in cap for name in ("q_qhd","k_khd","v_khd","raw_head_index")
    ):
        ap.error("Not a valid H3 Q/K/V captured head file")
    indices=cap["raw_head_index"]
    available=[int(x) for x in (indices.tolist() if isinstance(indices,torch.Tensor) else indices)]
    if not available or len(available)!=len(set(available)):
        ap.error("raw_head_index missing or duplicates")
    if args.heads>len(available):
        if not args.repeat_available_heads:
            ap.error(
                f"Capture has only {len(available)} distinct heads; "
                "use --repeat-available-heads only for a launch-memory proxy."
            )
        selected=[available[i%len(available)] for i in range(args.heads)]
        kind="four_head_repeated_proxy"
    else:
        selected=available[:args.heads]
        kind="distinct_native_heads"
    nkeys=(int(cap["k_khd"].shape[0])//16)*16
    free,total=torch.cuda.mem_get_info()
    budget=int(min(
        free, args.budget_mib*stage.MIB
        if args.budget_mib is not None else free
    ))
    # Fast/balanced = exact_leaves 256/512; splits4, tile64.
    near=256 if args.mode=="fast" else 512
    plan=stage.plan_tile_head_batches(
        len(selected),free_bytes=budget,keys_per_head=nkeys,
        tile_count=args.tiles,exact_leaves=near,
        reserve_bytes=int(args.reserve_mib*stage.MIB),
        fixed_temp_bytes=int(args.fixed_temp_mib*stage.MIB),
        safety_factor=args.safety_factor,
        max_heads_per_group=args.max_heads_per_group
    )
    print(json.dumps({
        "gpu":torch.cuda.get_device_name(),
        "capture_distinct_heads":len(available),
        "logical_heads":len(selected),
        "kind":kind,
        "gpu_free_mib_before":free/stage.MIB,
        "effective_budget_mib":budget/stage.MIB,
        "planned_group_heads":plan.heads_per_group,
        "planned_group_count":plan.groups,
        "estimate_max_extra_mib":plan.estimated_max_additional_bytes/stage.MIB,
    },indent=2),flush=True)
    report={
        "experiment":"h3_flat_universal_tile_streamed_head_groups_v1",
        "gpu":torch.cuda.get_device_name(),
        "capture":str(path),
        "source_commits":{
            "pinned_best":base.BEST,
            "candidate":subprocess.check_output(
                ["git","rev-parse","HEAD"],cwd=ROOT,text=True
            ).strip(),
        },
        "input_kind":kind,
        "is_repeated_capture_proxy":kind!="distinct_native_heads",
        "raw_heads_used":selected,
        "mode":args.mode,"tiles":args.tiles,
        "free_mib_before":free/stage.MIB,
        "effective_free_budget_mib":budget/stage.MIB,
        "plan":vars(plan),
        "memory_oom_retries":[],
        "groups":[],"failures":[],
        "notes":[
            "Only ONE group of head data is prepared at a time on GPU, then freed.",
            "Tile only CUDA event timing; preparation stats/select/topk recorded as untimed pre-stage wall.",
            "Output parity PM/PL/PACC vs pinned BEST; tail and Merge excluded.",
            "Sum of group median times is NOT one end-to-end 56-head latency measurement.",
            "Peak torch.cuda.max_memory_allocated is an actual PyTorch allocated peak, not total memory reserved by other GPU processes.",
            "CUDA Graph replay for arbitrary group sizes remains a separate test.",
        ],
    }
    current_group=plan.heads_per_group
    start=0
    with tempfile.TemporaryDirectory(prefix="tile_stream_best_") as d:
        pinned,pinned_runner=base.base_helper.original_modules(
            base.BEST,Path(d)
        )
        local=base.base_helper.helpers.sw.load(
            "_flat_tile_stream_candidate","h3/htrc_flat.py"
        )
        old_tr=pinned._load_triton()
        new_tr=local._load_triton()
        while start<len(selected):
            end=min(len(selected),start+current_group)
            subset=selected[start:end]
            data=None
            try:
                torch.cuda.synchronize()
                live_start=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                prep_start=time.perf_counter()
                data,cfg,keys,nb,kx=base.make_states(
                    cap,subset,args.tiles,args.mode,pinned,pinned_runner
                )
                torch.cuda.synchronize()
                prep_wall_ms=(time.perf_counter()-prep_start)*1000
                row=measure_group(
                    data,cfg,old_tr,new_tr,nb,kx,args.tiles,
                    args.warmups,args.rounds,args.batch,args.tolerance
                )
                torch.cuda.synchronize()
                allocated_peak=torch.cuda.max_memory_allocated()
                row.update({
                    "start_job":start,"end_job":end,
                    "source_raw_head_indices":subset,
                    "prep_wall_ms_including_sync":prep_wall_ms,
                    "live_alloc_before_mib":live_start/stage.MIB,
                    "measured_peak_alloc_mib":allocated_peak/stage.MIB,
                    "measured_incremental_peak_mib":
                        max(0,allocated_peak-live_start)/stage.MIB,
                    "prepared_state_mib":sum(
                        x.numel()*x.element_size() for x in data.values()
                    )/stage.MIB,
                })
                report["groups"].append(row)
                print(json.dumps({
                    "job_range":[start,end],
                    "parity_pass":row["parity_pass"],
                    "tile_speedup":row["timings"]["speedup_pinned_over_grouped"]
                        if row["timings"] else None,
                    "incremental_peak_mib":row["measured_incremental_peak_mib"],
                },indent=2),flush=True)
                if not row["parity_pass"]:
                    raise RuntimeError("Batched Tile parity mismatch")
                start=end
            except torch.cuda.OutOfMemoryError as exc:
                if current_group<=1:
                    report["failures"].append({
                        "head_start":start,"group_size":current_group,
                        "error":f"OOM even for one head: {exc}"
                    })
                    break
                before=current_group
                current_group=max(1,current_group//2)
                report["memory_oom_retries"].append({
                    "head_start":start,"previous_group":before,
                    "new_group":current_group,
                    "error":str(exc)[:500],
                })
                print(f"OOM: group {before} -> {current_group}; retrying job {start}",
                      flush=True)
            except Exception as exc:
                report["failures"].append({
                    "head_start":start,"group_size":current_group,
                    "type":type(exc).__name__,
                    "error":str(exc)[:1800],
                    "traceback_tail":traceback.format_exc()[-2000:],
                })
                break
            finally:
                del data
                gc.collect()
                torch.cuda.empty_cache()
    valid_rows=[r for r in report["groups"] if r["parity_pass"] and r["timings"]]
    if valid_rows:
        pinned_total=sum(r["timings"]["median_tile_ms"]["pinned"] for r in valid_rows)
        batch_total=sum(r["timings"]["median_tile_ms"]["grouped"] for r in valid_rows)
        report["sum_of_group_tile_medians_not_e2e"]={
            "pinned_ms":pinned_total,"grouped_ms":batch_total,
            "speedup":pinned_total/batch_total,
        }
        report["max_observed_incremental_peak_mib"]=max(
            r["measured_incremental_peak_mib"] for r in valid_rows
        )
    report["all_logical_heads_processed"]=(
        start==len(selected) and not report["failures"]
    )
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    target=out/f"flat_tile_streamed_groups_{time.strftime('%Y%m%d-%H%M%S')}.json"
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",target)
    if not report["all_logical_heads_processed"]:
        raise SystemExit("FAIL: check JSON for OOM/parity/benchmark error")


if __name__=="__main__":
    main()
