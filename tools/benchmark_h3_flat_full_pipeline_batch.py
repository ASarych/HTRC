"""Pinned BEST versus full grouped H3 Flat Q/K/V -> final attention output.

One group at a time, from a trusted capture on RTX 5070 Ti.
Pinned branch: for each head, original preparation (stats, rowmax,
topk, K8 quant) followed by per-head Tile + 32-row tail Merge.
Candidate: batch Q/K/V gather, one head-aware stats kernel,
one grouped rowmax, one batched topk, one Tile, one tail Merge.

Each per-group A/B event includes ALL CPU input gathering and H2D
copies, GPU preparation and attention work, synchronization and
allocator activity. Unlike the Tile microbench this is deliberately
a HOST wall-time measurement, relevant to the original H3 runtime orchestration.
No file I/O or full H3 model is included in the timed region.

The saved default capture has only four unique native heads.
Use the explicit --repeat-available-heads flag for 56 logical
work items, and do not claim native 56-head model correctness.
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
    "_full_prep_best_helper",ROOT/"tools"/"benchmark_h3_flat_tile_merge_streamed.py"
)
prior=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=prior
SPEC.loader.exec_module(prior)
base=prior.base
stage=prior.stage


def eval_one(capture, raw_heads, tiles, mode, pinned, runner,
             tr_best, tr_new, path_name, keep_output=False,
             quant_mode="torch_gpu", diagnose_prep=False,
             kv_cache=None):
    """Allocate/prepare/work/sync in one bounded GPU group."""
    torch.cuda.synchronize()
    alloc_before=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    stages=None
    if path_name=="best":
        states,cfg,nkeys,nb,kx=base.make_states(
            capture,raw_heads,tiles,mode,pinned,runner
        )
    elif path_name=="batched":
        cfg=base.base_helper.configuration(pinned,runner,mode,tiles)
        backend,casting=quant_mode.split("_",1)
        stages={} if diagnose_prep else None
        states,cfg,nkeys,nb,kx=stage.prepare_batched_head_states(
            capture,raw_heads,cfg,tr_new,tile_count=tiles,
            quant_backend=backend,input_cast=casting,
            stage_timings_ms=stages,kv_cache=kv_cache
        )
    else:
        raise ValueError("Invalid A/B path name")
    torch.cuda.synchronize()
    prep_ms=(time.perf_counter()-begin)*1000.0

    tails=prior.tails_for_heads(
        capture,raw_heads,nkeys,torch.device("cuda")
    )
    h=len(raw_heads)
    out=torch.empty((h,tiles,64,128),device="cuda",dtype=torch.float32)
    workspaces=None
    partials=None
    if path_name=="best":
        workspaces=[{} for _ in raw_heads]
        kern=cfg["kern"]
        for i in range(h):
            tr_best.tile_attention(
                states["Q"][i],states["MU"][i],
                states["G"][i],states["LZ"][i],states["VB"][i],
                states["EXCL"][i],states["IDX"][i],
                states["K"][i],states["V"][i],
                1.0/math.sqrt(128),leaf=16,prec="tf32",
                splits=int(cfg["splits"]),dot=1,
                bn=int(kern.get("tile_bn",32)),
                warps=int(kern.get("tile_warps",4)),
                stages=int(kern.get("tile_stages",1)),rb=1,
                tail=(tails[0][i],tails[1][i]),
                ws=workspaces[i],group=1,sg=states["SG"][i],
                out=out[i],ks=states["KS"][i],passes="fused",
                late_near_q=False,
                near_bn=int(kern.get("tile_near_bn") or kern.get("tile_bn",32)),
                tail_in_merge=True,tail_merge_rows=32,
                far_prefetch_lz=False,pv_dot_acc="none"
            )
    else:
        partials,_=stage.launch_prepared_tile_batch(states,cfg,tr_new)
        out,_=stage.launch_prepared_merge_batch(
            states,cfg,tr_new,partials,tails=tails,output=out,merge_rows=32
        )
    torch.cuda.synchronize()
    total_ms=(time.perf_counter()-begin)*1000.0
    tile_merge_ms=total_ms-prep_ms
    peak=torch.cuda.max_memory_allocated()
    measurements={
        "preparation_stage_sync_wall_ms_diagnostic_only":stages,
        "quant_mode":quant_mode if path_name=="batched" else "pinned_original",
        "kv_cache_reused":kv_cache is not None,
        "kv_cache_resident_mib":(stage.gpu_head_kv_cache_bytes(kv_cache)/stage.MIB
                                  if kv_cache is not None else 0.0),
        "preparation_wall_ms":prep_ms,
        "tail_tile_merge_wall_ms":tile_merge_ms,
        "full_group_wall_ms":total_ms,
        "live_alloc_before_mib":alloc_before/stage.MIB,
        "peak_alloc_mib":peak/stage.MIB,
        "incremental_peak_mib":max(0,peak-alloc_before)/stage.MIB,
        "prepared_state_mib":sum(
            t.numel()*t.element_size() for t in states.values()
        )/stage.MIB,
    }
    result=out.clone() if keep_output else None
    del out,states,tails,workspaces,partials
    # Release group allocations before next group or next A/B turn.
    torch.cuda.synchronize()
    return measurements,result


def compare_outputs(reference, candidate, tolerance):
    df=candidate.float()-reference.float()
    den=reference.float().norm().clamp_min(1e-20)
    rel=float(df.norm()/den)
    mx=float(df.abs().max())
    finite=bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    return {
        "rel_l2":rel,
        "max_abs":mx,
        "finite":finite,
        "pass":finite and rel<=tolerance,
    }


def paired_group(capture, ids, tiles, mode, pinned, runner,
                 tr_best, tr_new, warmups, rounds, tolerance,
                 check_full_oracle, quant_mode="torch_gpu"):
    # Warm-up is excluded; neither cold JIT nor first allocations
    # should be mistaken for steady-state performance.
    for _ in range(warmups):
        for path in ("best","batched"):
            record,temp=eval_one(
                capture,ids,tiles,mode,pinned,runner,
                tr_best,tr_new,path,quant_mode=quant_mode
            )
            del record,temp
    gc.collect()
    torch.cuda.empty_cache()

    old_record,old=eval_one(
        capture,ids,tiles,mode,pinned,runner,tr_best,tr_new,
        "best",keep_output=True,quant_mode=quant_mode
    )
    new_record,new=eval_one(
        capture,ids,tiles,mode,pinned,runner,tr_best,tr_new,
        "batched",keep_output=True,quant_mode=quant_mode,
        diagnose_prep=True
    )
    gate=compare_outputs(old,new,tolerance)
    independent=[]
    if gate["pass"] and check_full_oracle:
        done=set()
        for i,head in enumerate(ids):
            if head in done:
                continue
            full=prior.full_pinned_output(
                pinned,runner,capture,head,
                (int(capture["k_khd"].shape[0])//16)*16,
                tiles,mode
            ).view(tiles,64,128)
            check=compare_outputs(full,new[i],tolerance)
            independent.append({"raw_head":head,**check})
            done.add(head)
            del full
    del old,new
    gc.collect()
    torch.cuda.empty_cache()

    result={
        "head_count":len(ids),
        "output_parity":gate,
        "full_oracle":independent,
        "warmup_discarded_per_path":warmups,
        "preflight_records":{"best":old_record,"batched":new_record},
        "timings":None,
    }
    if not gate["pass"] or any(not c["pass"] for c in independent):
        return result

    samples={"best":[],"batched":[]}
    for _ in range(rounds):
        for path in ("best","batched","batched","best"):
            measurement,out=eval_one(
                capture,ids,tiles,mode,pinned,runner,
                tr_best,tr_new,path,quant_mode=quant_mode
            )
            samples[path].append(measurement)
            del out,measurement
    summary={}
    for path in samples:
        summary[path]={
            key:statistics.median(s[key] for s in samples[path])
            for key in ("preparation_wall_ms","tail_tile_merge_wall_ms",
                        "full_group_wall_ms","incremental_peak_mib")
        }
    result["timings"]={
        "median_by_path":summary,
        "speedup_best_over_batched_full_wall":
            summary["best"]["full_group_wall_ms"]/
            summary["batched"]["full_group_wall_ms"],
        "speedup_best_over_batched_preparation":
            summary["best"]["preparation_wall_ms"]/
            summary["batched"]["preparation_wall_ms"],
        "sample_records":samples,
    }
    return result


def quant_key_exactness_gate(capture, raw_head, triton_module):
    """Independent exact K8 and FP32 scale check before benchmarking."""
    indices=capture["raw_head_index"]
    raw=[int(x) for x in (indices.tolist()
                          if isinstance(indices,torch.Tensor) else indices)]
    nkeys=(int(capture["k_khd"].shape[0])//16)*16
    kbf=(capture["k_khd"][:nkeys,raw.index(raw_head)]
         .to(torch.bfloat16).to("cuda").contiguous())
    kf=kbf.float()
    ref_sc=(kf.abs().amax(-1)*(1.0/127.0)).clamp_min_(1e-20)
    ref_k8=(kf/ref_sc[:,None]).round_().clamp_(-127,127).to(torch.int8)
    new_sc=torch.empty_like(ref_sc)
    new_k8=torch.empty_like(ref_k8)
    br=8
    triton_module._quant_key_int8_rows[
        (triton_module.triton.cdiv(nkeys,br),)
    ](kbf,new_k8,new_sc,nkeys,128,br,num_warps=4)
    torch.cuda.synchronize()
    mismatch=int((new_k8!=ref_k8).sum())
    scale_max_abs=float((new_sc-ref_sc).abs().max())
    scale_ref_abs_max=float(ref_sc.abs().max())
    return {
        "raw_head":raw_head,
        "key_rows":nkeys,
        "int8_mismatch_count":mismatch,
        "scale_max_abs":scale_max_abs,
        "scale_ref_max_abs":scale_ref_abs_max,
        "pass":mismatch==0 and scale_max_abs<=max(1e-20,scale_ref_abs_max*1e-6)
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--capture",
        default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt"
    )
    ap.add_argument("--heads",type=int,default=56)
    ap.add_argument("--repeat-available-heads",action="store_true")
    ap.add_argument("--mode",default="fast",choices=("fast","balanced"))
    ap.add_argument("--quant-mode",default="torch_gpu",
                    choices=("torch_gpu","torch_cpu","triton_gpu","triton_cpu"),
                    help="Candidate K8 quant/copy backend; BEST always pinned")
    ap.add_argument("--input-residency",default="cpu",choices=("cpu","gpu"),
                    help="CPU capture (H2D timed) or preloaded GPU QKV (H2D outside timing)")
    ap.add_argument("--tiles",type=int,default=16)
    ap.add_argument("--budget-mib",type=float,default=1024.0)
    ap.add_argument("--reserve-mib",type=float,default=256.0)
    ap.add_argument("--max-heads-per-group",type=int,default=None)
    ap.add_argument("--rounds",type=int,default=2)
    ap.add_argument("--warmups",type=int,default=1)
    ap.add_argument("--tolerance",type=float,default=1e-4)
    ap.add_argument("--out",default=r"D:\ComfyOutput\H3\math_probe")
    args=ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("CUDA GPU required")
    if args.input_residency=="gpu" and args.quant_mode.endswith("_cpu"):
        ap.error("GPU-resident QKV requires --quant-mode torch_gpu or triton_gpu")
    if any(v<1 for v in (
        args.heads,args.tiles,args.rounds,args.warmups
    )):
        ap.error("heads/tiles/rounds/warmups must be >=1")
    if not math.isfinite(args.budget_mib) or args.budget_mib<=0:
        ap.error("GPU budget must be finite and positive")
    if not math.isfinite(args.reserve_mib) or args.reserve_mib<0:
        ap.error("Reserve must be finite and nonnegative")
    capture_path=Path(args.capture)
    if not capture_path.is_file():
        ap.error("Trusted capture file not found")
    cap=torch.load(
        capture_path,map_location="cpu",mmap=True,weights_only=False
    )
    if not isinstance(cap,dict) or any(
        k not in cap for k in ("q_qhd","k_khd","v_khd","raw_head_index")
    ):
        ap.error("No native saved Q/K/V/raw head list in capture")
    raw=cap["raw_head_index"]
    available=[int(x) for x in (
        raw.tolist() if isinstance(raw,torch.Tensor) else raw
    )]
    if not available or len(set(available))!=len(available):
        ap.error("Empty/duplicate raw head indices in capture")
    if args.heads>len(available):
        if not args.repeat_available_heads:
            ap.error("Only 4 native heads saved; --repeat-available-heads required for 56 proxy")
        selected=[available[i%len(available)] for i in range(args.heads)]
        kind="repeated_capture_proxy"
    else:
        selected=available[:args.heads]
        kind="distinct_native"
    # Simulate model-native GPU input without mixing PCIe transfers into A/B.
    # The default CPU-mmap workload remains unchanged and independent.
    capture_gpu_mib=0.0
    if args.input_residency=="gpu":
        cap=dict(cap)
        for key in ("q_qhd","k_khd","v_khd"):
            cap[key]=cap[key].to("cuda").contiguous()
        torch.cuda.synchronize()
        capture_gpu_mib=sum(
            cap[key].numel()*cap[key].element_size()
            for key in ("q_qhd","k_khd","v_khd")
        )/stage.MIB
    free,_=torch.cuda.mem_get_info()
    budget=min(free,int(args.budget_mib*stage.MIB))
    keys=(int(cap["k_khd"].shape[0])//16)*16
    planned=stage.plan_tile_head_batches(
        len(selected),free_bytes=budget,keys_per_head=keys,
        tile_count=args.tiles,exact_leaves=256 if args.mode=="fast" else 512,
        reserve_bytes=int(args.reserve_mib*stage.MIB),
        fixed_temp_bytes=128*stage.MIB,
        safety_factor=1.5,
    )
    max_group=min(planned.heads_per_group,
                  args.max_heads_per_group or planned.heads_per_group)
    print(json.dumps({
        "head_type":kind,"logical_heads":len(selected),
        "input_residency":args.input_residency,
        "capture_gpu_resident_mib":capture_gpu_mib,
        "free_mib_before":free/stage.MIB,
        "effective_budget_mib":budget/stage.MIB,
        "head_group_size":max_group,
        "groups":math.ceil(len(selected)/max_group),
    },indent=2),flush=True)
    report={
        "experiment":("h3_flat_gpu_resident_qkv_prep_ab_v1"
                      if args.input_residency=="gpu"
                      else "h3_flat_full_qkv_quantized_prep_ab_v2"),
        "input_residency":args.input_residency,
        "capture_gpu_resident_mib":capture_gpu_mib,
        "quant_mode":args.quant_mode,
        "quantization_exact_gate":None,
        "gpu":torch.cuda.get_device_name(),
        "capture":str(capture_path),
        "input_kind":kind,
        "is_repeated_capture_proxy":kind!="distinct_native",
        "raw_head_indices":selected,
        "mode":args.mode,"tiles":args.tiles,
        "source_commits":{
            "pinned_best":base.BEST,
            "candidate":subprocess.check_output(
                ["git","rev-parse","HEAD"],cwd=ROOT,text=True
            ).strip()
        },
        "free_mib_before":free/stage.MIB,
        "gpu_budget_mib":budget/stage.MIB,
        "planned":vars(planned),
        "max_head_group_size":max_group,
        "groups":[],"errors":[],"oom_group_retries":[],
        "notes":[
            "Pinned original full preparation + individual Tile/R32 tail Merge versus batched preparation+Tile+Merge.",
            "Both paths start from SAME captured Q/K/V head tensors at the reported input residency.",
            "A/B uses synchronized HOST wall time inclusive of source-device gather/cast, GPU stats/select/topk, Tile and Merge; CPU mode also includes H2D.",
            "GPU mode preloads captured QKV outside timings; capture VRAM persists and is excluded from group incremental peaks. This is NOT a checkpoint-resident test.",
            "No file IO, model denoise, or video included in timed region.",
            "All candidate final outputs checked against pinned final output, and first group unique heads checked against full pinned Flat.",
            "Repeated 4-head proxy does NOT prove real native 56-unique-head model behavior.",
            "Each path fully rebuilds temporary tensors; this is not a graph or persistent-workspace benchmark.",
            "Quant-mode affects candidate only; pinned BEST source is unchanged.",
            "Stage wall breakdown synchronizes only the untimed candidate preflight.",
            "Report is not proof of performance with 14GiB H3 checkpoint resident.",
        ]
    }
    pos=0
    group_size=max_group
    with tempfile.TemporaryDirectory(prefix="fullprep_best_") as tmp:
        pinned,runner=base.base_helper.original_modules(
            base.BEST,Path(tmp)
        )
        candidate=base.base_helper.helpers.sw.load(
            "_h3_batch_full_prep_candidate","h3/htrc_flat.py"
        )
        best_tri=pinned._load_triton()
        cand_tri=candidate._load_triton()
        if args.quant_mode.startswith("triton"):
            try:
                report["quantization_exact_gate"]=quant_key_exactness_gate(
                    cap,selected[0],cand_tri
                )
                print("QUANT_EXACT_GATE:",
                      json.dumps(report["quantization_exact_gate"]),flush=True)
                if not report["quantization_exact_gate"]["pass"]:
                    report["errors"].append({
                        "job":0,"error":"Triton K8/scales differ from pinned torch quantizer"
                    })
            except Exception as exc:
                report["errors"].append({
                    "job":0,"type":type(exc).__name__,
                    "error":f"Quantizer compilation/reference check: {exc}",
                    "traceback_tail":traceback.format_exc()[-2500:]
                })
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        while pos<len(selected) and not report["errors"]:
            end=min(len(selected),pos+group_size)
            group_ids=selected[pos:end]
            try:
                row=paired_group(
                    cap,group_ids,args.tiles,args.mode,
                    pinned,runner,best_tri,cand_tri,
                    args.warmups,args.rounds,args.tolerance,
                    check_full_oracle=(pos==0),quant_mode=args.quant_mode
                )
                row.update({"head_job_range":[pos,end],"raw_heads":group_ids})
                report["groups"].append(row)
                print(json.dumps({
                    "jobs":[pos,end],
                    "output_parity":row["output_parity"],
                    "full_oracle":row["full_oracle"],
                    "wall_speedup":row["timings"]["speedup_best_over_batched_full_wall"]
                        if row["timings"] else None,
                    "prep_speedup":row["timings"]["speedup_best_over_batched_preparation"]
                        if row["timings"] else None,
                },indent=2),flush=True)
                if not row["output_parity"]["pass"] or row["timings"] is None:
                    raise RuntimeError("Final attention or full-oracle parity failed")
                pos=end
            except torch.cuda.OutOfMemoryError as exc:
                if group_size==1:
                    report["errors"].append({
                        "job":pos,"error":f"OutOfMemoryError at one head: {exc}"[:1200]
                    })
                    break
                before=group_size
                group_size=max(1,group_size//2)
                report["oom_group_retries"].append({
                    "job":pos,"from":before,"to":group_size,
                    "message":str(exc)[:600]
                })
            except Exception as exc:
                report["errors"].append({
                    "job":pos,"exception":type(exc).__name__,
                    "error":str(exc)[:1800],
                    "traceback_tail":traceback.format_exc()[-2500:]
                })
                break
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    report["completed"]=pos==len(selected) and not report["errors"]
    valid=[r for r in report["groups"] if r.get("timings")]
    if valid:
        old=sum(x["timings"]["median_by_path"]["best"]["full_group_wall_ms"]
                for x in valid)
        new=sum(x["timings"]["median_by_path"]["batched"]["full_group_wall_ms"]
                for x in valid)
        report["sum_of_group_median_wall_not_true_end_to_end"]={
            "pinned_ms":old,"batched_ms":new,
            "speedup":old/new
        }
        report["max_incremental_peak_mib"]=max(
            max(x["timings"]["median_by_path"][k]["incremental_peak_mib"]
                for k in ("best","batched")) for x in valid
        )
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    prefix=("flat_full_prep_gpu_qkv_ab" if args.input_residency=="gpu"
            else "flat_full_prep_quant_ab")
    report_path=out/f"{prefix}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",report_path)
    if not report["completed"]:
        raise SystemExit("FAIL: full pipeline group A/B (check JSON)")


if __name__=="__main__":
    main()
