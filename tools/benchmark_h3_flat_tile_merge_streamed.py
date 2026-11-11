"""Full Tile+exact-Tail-Merge group A/B on native H3 captured Q/K/V.

Control: pinned BEST e47cc7a, per-head Tile + R32 tail Merge.
Candidate: one head-flattened Tile and one head-flattened R32
per group, with each head's own exact tail K/V.

Stages stats and rowmax/topk prepare the *same* per-head states
outside the timed Tile+Merge region. This is a numerical gate for
FINAL attention outputs, not yet an end-to-end Flat runtime test.

Bound GPU state to one group selected from actual free memory.
If the capture has only 4 unique raw heads, explicitly permit
repeating those to benchmark 56 logical heads as a PROXY.

PowerShell:
  & $py -B tools\benchmark_h3_flat_tile_merge_streamed.py --heads 56 --repeat-available-heads --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256
"""
from __future__ import annotations

import argparse
import copy
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
    "_tile_tail_group_base",ROOT/"tools"/"benchmark_h3_flat_tile_56head_launch.py"
)
base=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=base
SPEC.loader.exec_module(base)

PLAN_SPEC=importlib.util.spec_from_file_location(
    "_tile_tail_group_plan",ROOT/"h3"/"htrc_flat_head_batch.py"
)
stage=importlib.util.module_from_spec(PLAN_SPEC)
sys.modules[PLAN_SPEC.name]=stage
PLAN_SPEC.loader.exec_module(stage)


def tails_for_heads(capture, raw_heads, nfull, device):
    index_map=capture["raw_head_index"]
    raw=[int(h) for h in (index_map.tolist()
                         if isinstance(index_map,torch.Tensor)
                         else index_map)]
    pos={head:i for i,head in enumerate(raw)}
    n_tail=int(capture["k_khd"].shape[0])-nfull
    if not 0<n_tail<=16:
        raise ValueError("This R32 tail-Merge gate requires 1..16 tail K/V rows")
    tk=torch.stack([
        capture["k_khd"][nfull:,pos[head]].to(device=device,dtype=torch.bfloat16)
        for head in raw_heads
    ]).contiguous()
    tv=torch.stack([
        capture["v_khd"][nfull:,pos[head]].to(device=device,dtype=torch.bfloat16)
        for head in raw_heads
    ]).contiguous()
    return tk,tv


def compare_prepared(states,cfg,tails,tr_best,tr_new,
                     warmups,rounds,batch,tolerance,with_graph):
    h,tiles,tile,dim=map(int,states["Q"].shape)
    kx=int(states["IDX"].shape[2])
    ns=int(cfg["splits"])
    kw=dict(cfg["kern"])
    bn=int(kw.get("tile_bn",32))
    near_bn=int(kw.get("tile_near_bn") or bn)
    warps=int(kw.get("tile_warps",4))
    stages=int(kw.get("tile_stages",1))
    if int(tails[0].shape[0])!=h or tails[0].shape!=tails[1].shape:
        raise RuntimeError("Batched tails do not match head count")
    output_shape=(h,tiles,tile,dim)
    ref_out=torch.empty(output_shape,device="cuda",dtype=torch.float32)
    new_out=torch.empty_like(ref_out)
    pp=(h,tiles,ns,tile)
    partials={
        "pm":torch.empty(pp,device="cuda",dtype=torch.float32),
        "pl":torch.empty(pp,device="cuda",dtype=torch.float32),
        "acc":torch.empty(pp+(dim,),device="cuda",dtype=torch.float32),
    }
    workspaces=[{} for _ in range(h)]

    def pinned():
        meta=None
        for head in range(h):
            meta=tr_best.tile_attention(
                states["Q"][head],states["MU"][head],
                states["G"][head],states["LZ"][head],states["VB"][head],
                states["EXCL"][head],states["IDX"][head],
                states["K"][head],states["V"][head],
                1.0/math.sqrt(dim),
                leaf=16,prec="tf32",splits=ns,dot=1,bn=bn,
                warps=warps,stages=stages,rb=1,
                tail=(tails[0][head],tails[1][head]),
                ws=workspaces[head],group=1,
                sg=states["SG"][head],sv=None,
                out=ref_out[head],ks=states["KS"][head],
                passes="fused",late_near_q=False,
                near_bn=near_bn,
                tail_in_merge=True,tail_merge_rows=32,
                far_prefetch_lz=False,pv_dot_acc="none",
            )
        return meta

    def grouped():
        _,tile_meta=stage.launch_prepared_tile_batch(
            states,cfg,tr_new,buffers=partials
        )
        _,merge_meta=stage.launch_prepared_merge_batch(
            states,cfg,tr_new,partials,tails=tails,
            output=new_out,merge_rows=32
        )
        return tile_meta,merge_meta

    pinned()
    tile_meta,merge_meta=grouped()
    torch.cuda.synchronize()
    delta=new_out.float()-ref_out.float()
    denom=ref_out.float().norm().clamp_min(1e-20)
    parity={
        "relative_l2":float(delta.norm()/denom),
        "max_abs":float(delta.abs().max()),
        "finite":bool(torch.isfinite(ref_out).all() and torch.isfinite(new_out).all()),
    }
    parity["pass"]=parity["finite"] and parity["relative_l2"]<=tolerance
    del delta
    result={
        "heads":h,"tiles_per_head":tiles,
        "tail_rows_per_head":int(tails[0].shape[1]),
        "numerical_parity_output":parity,
        "parity_pass":parity["pass"],
        "launches_pinned_tile_plus_merge":2*h,
        "launches_grouped_tile_plus_merge":2,
        "compiled_candidate_resources":{
            "tile":base.resources(tile_meta),
            "merge":base.resources(merge_meta),
        },
        "eager":None,"cuda_graph":None
    }
    if not parity["pass"]:
        return result,ref_out,new_out

    for _ in range(warmups):
        pinned();grouped()
    torch.cuda.synchronize()

    def paired(reference,optimized):
        samples={"pinned":[],"batched":[]}
        for _ in range(rounds):
            for key in ("pinned","batched","batched","pinned"):
                fn=reference if key=="pinned" else optimized
                samples[key].append(base.timed_event(fn,batch))
        med={key:statistics.median(vals) for key,vals in samples.items()}
        return {
            "median_tile_merge_ms":med,
            "speedup_pinned_over_batched":med["pinned"]/med["batched"],
            "samples_ms":samples,
        }
    result["eager"]=paired(pinned,grouped)
    if with_graph:
        try:
            # Capture the two actual GPU pipelines; graph failure is
            # recorded, not silently interpreted as eager PASS.
            graph_pinned=torch.cuda.CUDAGraph()
            graph_group=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_pinned):
                pinned()
            with torch.cuda.graph(graph_group):
                grouped()
            graph_pinned.replay();graph_group.replay()
            torch.cuda.synchronize()
            result["cuda_graph"]=paired(graph_pinned.replay,graph_group.replay)
            del graph_pinned,graph_group
        except Exception as exc:
            result["cuda_graph"]={
                "error":f"{type(exc).__name__}: {exc}",
                "traceback_tail":traceback.format_exc()[-1500:],
            }
    return result,ref_out,new_out


def full_pinned_output(flat,runner,capture,raw_head,nfull,tiles,mode):
    """Independent full Flat operator crosscheck for one head, untimed."""
    heads=[int(x) for x in capture["raw_head_index"].tolist()]
    index=heads.index(raw_head)
    dev=torch.device("cuda")
    q=capture["q_qhd"][:tiles*64,index].to(dev).float().contiguous()
    k=capture["k_khd"][:nfull,index].to(dev).contiguous()
    v=capture["v_khd"][:nfull,index].to(dev).contiguous()
    tail=(
        capture["k_khd"][nfull:,index].to(dev).float().contiguous(),
        capture["v_khd"][nfull:,index].to(dev).float().contiguous(),
    )
    cfg=base.base_helper.configuration(flat,runner,mode,tiles)
    cfg["kern"].update(tile_tail_merge=True,tile_tail_merge_rows=32)
    return flat.flat_attention(q,k,v,tail=tail,ws={},**cfg).clone()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--capture",default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt"
    )
    ap.add_argument("--heads",type=int,default=56)
    ap.add_argument("--repeat-available-heads",action="store_true")
    ap.add_argument("--mode",default="fast",choices=("fast","balanced"))
    ap.add_argument("--tiles",type=int,default=16)
    ap.add_argument("--budget-mib",type=float,default=1024.0)
    ap.add_argument("--reserve-mib",type=float,default=256.0)
    ap.add_argument("--max-heads-per-group",type=int,default=None)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--warmups",type=int,default=2)
    ap.add_argument("--cuda-graph",action="store_true")
    ap.add_argument("--tolerance",type=float,default=1e-4)
    ap.add_argument("--out",default=r"D:\ComfyOutput\H3\math_probe")
    args=ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("NVIDIA CUDA GPU required")
    if any(v<1 for v in (args.heads,args.tiles,args.rounds,args.batch,args.warmups)):
        ap.error("heads/tiles/rounds/batch/warmups must be >=1")
    if args.tiles*64>40759:
        ap.error("Cannot exceed available query rows of pinned capture")
    path=Path(args.capture)
    if not path.is_file():
        ap.error("Trusted saved capture file missing")
    capture=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    if not isinstance(capture,dict) or any(
        key not in capture for key in ("q_qhd","k_khd","v_khd","raw_head_index")
    ):
        ap.error("Invalid saved H3 head capture")
    indices=capture["raw_head_index"]
    available=[int(x) for x in (indices.tolist()
              if isinstance(indices,torch.Tensor) else indices)]
    if not available or len(set(available))!=len(available):
        ap.error("Missing/duplicate raw heads")
    if args.heads>len(available):
        if not args.repeat_available_heads:
            ap.error("Only four native heads in default file; explicit --repeat-available-heads required")
        selected=[available[i%len(available)] for i in range(args.heads)]
        kind="repeated_head_proxy"
    else:
        selected=available[:args.heads]
        kind="distinct_native"
    keys=(int(capture["k_khd"].shape[0])//16)*16
    tail_rows=int(capture["k_khd"].shape[0])-keys
    if not 0<tail_rows<=16:
        ap.error("Expected 1..16 exact tail rows")
    free,_=torch.cuda.mem_get_info()
    budget=min(free,int(args.budget_mib*stage.MIB))
    group_plan=stage.plan_tile_head_batches(
        len(selected),free_bytes=budget,keys_per_head=keys,
        tile_count=args.tiles,exact_leaves=256 if args.mode=="fast" else 512,
        reserve_bytes=int(args.reserve_mib*stage.MIB),
        fixed_temp_bytes=128*stage.MIB,safety_factor=1.5,
        max_heads_per_group=args.max_heads_per_group,
    )
    report={
        "experiment":"h3_flat_batched_tile_merge_final_output_v1",
        "gpu":torch.cuda.get_device_name(),
        "source_commits":{
            "best":base.BEST,
            "candidate":subprocess.check_output(
                ["git","rev-parse","HEAD"],cwd=ROOT,text=True
            ).strip(),
        },
        "capture":str(path),
        "raw_head_indices":selected,
        "head_type":kind,
        "is_proxy":kind!="distinct_native",
        "mode":args.mode,"tile_count_per_head":args.tiles,
        "tail_rows_per_head":tail_rows,
        "budget_mib":budget/stage.MIB,
        "plan":vars(group_plan),
        "groups":[],"failures":[],"oom_retries":[],
        "full_operator_reference_checks":[],
        "notes":[
            "Final float32 Tile+R32 exact-tail Merge compared to pinned BEST.",
            "Each head has independent tail K/V, even in 56 logical-head proxy.",
            "Stages statistics, selection, topk are prepared but OUTSIDE CUDA-event timing.",
            "Only Tile+Merge timed; full original flat_attention used as untimed crosscheck once per unique head.",
            "Streamed group output is checked before deleting GPU buffers.",
            "GPU memory peak uses PyTorch allocated peak, not total device reserved.",
            "Graph timing optional: --cuda-graph; graph failure is recorded.",
            "Not whole H3 denoise or model-resident video test.",
        ]
    }
    seen=set()
    current_group=group_plan.heads_per_group
    pos=0
    with tempfile.TemporaryDirectory(prefix="tilemerge_best_") as tmp:
        pinned,runner=base.base_helper.original_modules(
            base.BEST,Path(tmp)
        )
        candidate=base.base_helper.helpers.sw.load(
            "_tile_merge_group_candidate","h3/htrc_flat.py"
        )
        best_tr=pinned._load_triton()
        new_tr=candidate._load_triton()
        while pos<len(selected):
            end=min(len(selected),pos+current_group)
            subset=selected[pos:end]
            data=tails=None
            try:
                torch.cuda.synchronize()
                before_alloc=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start_prep=time.perf_counter()
                data,cfg,keys,nb,kx=base.make_states(
                    capture,subset,args.tiles,args.mode,pinned,runner
                )
                tails=tails_for_heads(capture,subset,keys,torch.device("cuda"))
                torch.cuda.synchronize()
                prep_ms=(time.perf_counter()-start_prep)*1000
                result,old_output,batched_output=compare_prepared(
                    data,cfg,tails,best_tr,new_tr,args.warmups,
                    args.rounds,args.batch,args.tolerance,args.cuda_graph
                )
                # The full original Flat operator is a stronger,
                # independent numerical oracle than two staged paths.
                if result["parity_pass"]:
                    for i,raw in enumerate(subset):
                        if raw in seen:
                            continue
                        ref=full_pinned_output(
                            pinned,runner,capture,raw,keys,args.tiles,args.mode
                        )
                        ref=ref.view(args.tiles,64,128)
                        old_delta=old_output[i].float()-ref.float()
                        new_delta=batched_output[i].float()-ref.float()
                        denom=ref.float().norm().clamp_min(1e-20)
                        check={
                            "raw_head":raw,
                            "pinned_staged_rel_l2":float(old_delta.norm()/denom),
                            "candidate_rel_l2":float(new_delta.norm()/denom),
                            "finite":bool(torch.isfinite(ref).all()),
                        }
                        check["pass"]=check["finite"] and max(
                            check["pinned_staged_rel_l2"],
                            check["candidate_rel_l2"]
                        )<=args.tolerance
                        seen.add(raw)
                        report["full_operator_reference_checks"].append(check)
                        if not check["pass"]:
                            raise RuntimeError(
                                f"Full pinned operator crosscheck failed for raw head {raw}"
                            )
                        del ref,old_delta,new_delta
                torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated()
                result.update({
                    "group_range":[pos,end],
                    "raw_heads":subset,
                    "prep_wall_ms_including_sync":prep_ms,
                    "measured_peak_alloc_mib":peak/stage.MIB,
                    "incremental_peak_alloc_mib":max(0,peak-before_alloc)/stage.MIB,
                })
                report["groups"].append(result)
                print(json.dumps({
                    "jobs":[pos,end],"output_parity":result["parity_pass"],
                    "relative_l2":result["numerical_parity_output"]["relative_l2"],
                    "eager_speedup":result["eager"]["speedup_pinned_over_batched"]
                         if result["eager"] else None,
                    "incremental_peak_mib":result["incremental_peak_alloc_mib"],
                    "full_operator_checks":len(report["full_operator_reference_checks"]),
                },indent=2),flush=True)
                if not result["parity_pass"]:
                    raise RuntimeError("Final attention output parity failed")
                pos=end
            except torch.cuda.OutOfMemoryError as exc:
                if current_group==1:
                    report["failures"].append({
                        "position":pos,"error":f"OOM for one head: {exc}"[:1000]
                    })
                    break
                prev=current_group
                current_group=max(1,current_group//2)
                report["oom_retries"].append({
                    "position":pos,"from":prev,"to":current_group,
                    "message":str(exc)[:500],
                })
            except Exception as exc:
                report["failures"].append({
                    "position":pos,"type":type(exc).__name__,
                    "error":str(exc)[:1800],
                    "traceback_tail":traceback.format_exc()[-2400:],
                })
                break
            finally:
                del data,tails
                gc.collect()
                torch.cuda.empty_cache()
    report["completed"]=pos==len(selected) and not report["failures"]
    if report["groups"]:
        done=[r for r in report["groups"] if r["eager"] is not None]
        if done:
            pinned_ms=sum(r["eager"]["median_tile_merge_ms"]["pinned"] for r in done)
            batch_ms=sum(r["eager"]["median_tile_merge_ms"]["batched"] for r in done)
            report["sum_of_group_medians_NOT_end_to_end"]={
                "pinned_ms":pinned_ms,"batched_ms":batch_ms,
                "speedup":pinned_ms/batch_ms
            }
        report["max_incremental_peak_mib"]=max(
            r["incremental_peak_alloc_mib"] for r in report["groups"]
        )
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    report_path=out/f"flat_tile_merge_streamed_{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",report_path)
    if not report["completed"] or any(
        not r["pass"] for r in report["full_operator_reference_checks"]
    ):
        raise SystemExit("FAIL: output/full operator parity or GPU execution")


if __name__=="__main__":
    main()
