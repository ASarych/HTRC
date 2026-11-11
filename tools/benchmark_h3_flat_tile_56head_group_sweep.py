"""56 logical-head Tile launch group-size A/B against pinned BEST.

Reuses the same native-capture preparation as
tools/benchmark_h3_flat_tile_56head_launch.py; that file's default
capture has four distinct heads, so 56 logical heads require the
explicit --repeat-available-heads proxy flag.

For each group size G, candidate launches ceil(56/G) Triton Tile
kernels instead of 56 pinned-best Tile launches.  All head inputs,
leaf stats/selection, and partial state shapes are identical; only
head flattening / launch count changes. No production integration.

The script prepares all 56 logical heads simultaneously to isolate
launch behavior. It only *estimates* the staging size of a future
bounded-VRAM group implementation; it does NOT validate a real
lower-memory streaming pipeline.

PowerShell in the experimental worktree:
  & $py -B tools\benchmark_h3_flat_tile_group_sweep.py --heads 56 --repeat-available-heads --tiles 16
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


ROOT = Path(__file__).resolve().parents[1]
HELPER_SPEC = importlib.util.spec_from_file_location(
    "_h3_tile_group_sweep_base",
    ROOT / "tools" / "benchmark_h3_flat_tile_56head_launch.py",
)
base = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = base
HELPER_SPEC.loader.exec_module(base)


def compare_groups(states, cfg, keys, nb, kx, tiles, pinned_triton,
                   candidate_triton, group_sizes, warmups, rounds,
                   batch, tolerance, with_graph):
    H = int(states["Q"].shape[0])
    T = int(cfg["tile"])
    D = int(states["Q"].shape[-1])
    splits = int(cfg["splits"])
    kern = cfg["kern"]
    bn = int(kern.get("tile_bn", 32))
    warps = int(kern.get("tile_warps", 4))
    stages = int(kern.get("tile_stages", 1))
    near_bn = int(kern.get("tile_near_bn", bn))
    lps = pinned_triton._split(nb, splits, bn)
    eps = pinned_triton._split(kx * 16, splits, bn)
    scale = 1.0 / math.sqrt(D)
    kw = {
        "T":T, "D":D, "BN":bn, "BN_NEAR":near_bn,
        "LEAF":16, "PREC":"tf32", "DOT":1, "GR":1,
        "FP8":0, "TAIL":0, "RB":1, "I8":1, "PASS":0,
        "LATE_NEAR_Q":0, "FAR_PREFETCH_LZ":0,
        "FAR_PV_ACC":0, "NEAR_PV_ACC":0,
        "num_warps":warps, "num_stages":stages,
    }
    # Two independent partial workspaces: ref and candidate.
    shape=(H, tiles, splits, T)
    ref_pm=torch.empty(shape,device="cuda",dtype=torch.float32)
    ref_pl=torch.empty_like(ref_pm)
    ref_acc=torch.empty(shape+(D,),device="cuda",dtype=torch.float32)
    new_pm=torch.empty_like(ref_pm)
    new_pl=torch.empty_like(ref_pm)
    new_acc=torch.empty_like(ref_acc)
    q,mu,g,lz,vb,ex,idx,sg=(
        states[k] for k in ("Q","MU","G","LZ","VB","EXCL","IDX","SG")
    )
    k,v,ks=(states[name] for name in ("K","V","KS"))

    def original_head(h):
        return pinned_triton._tile_kernel[(tiles,splits,1)](
            q[h], mu[h], g[h], lz[h], vb[h], ex[h], idx[h],
            k[h], v[h], ref_pm[h], ref_pl[h], ref_acc[h],
            sg[h], lz[h], lz[h], lz[h], ks[h],
            nb,kx,lps,eps,0,scale,**kw,
        )

    def separate():
        for h in range(H):
            original_head(h)

    def launch_group(start, end):
        n=end-start
        if n<=0:
            raise ValueError("empty head group")
        def flat(name):
            arr=states[name][start:end]
            # Slicing one full head group preserves contiguity.
            return arr.reshape((n*tiles,)+arr.shape[2:])
        qg=flat("Q")
        mug=flat("MU")
        gg=flat("G")
        lzg=flat("LZ")
        vbg=flat("VB")
        exg=flat("EXCL")
        idxg=flat("IDX")
        sgg=flat("SG")
        kg=k[start:end].reshape(n*keys,D)
        vg=v[start:end].reshape(n*keys,D)
        ksg=ks[start:end].reshape(n*keys)
        pmg=new_pm[start:end].reshape(n*tiles,splits,T)
        plg=new_pl[start:end].reshape(n*tiles,splits,T)
        accg=new_acc[start:end].reshape(n*tiles,splits,T,D)
        return candidate_triton._tile_kernel[(n*tiles,splits,1)](
            qg,mug,gg,lzg,vbg,exg,idxg,
            kg,vg,pmg,plg,accg,
            sgg,lzg,lzg,lzg,ksg,
            nb,kx,lps,eps,0,scale,
            BATCH_TILES_PER_HEAD=tiles,
            BATCH_KEY_ROWS=keys,**kw,
        )

    def grouped(groupsize):
        for start in range(0,H,groupsize):
            launch_group(start,min(start+groupsize,H))

    # Compilation + baseline state once.
    pinned_metadata=base.resources(original_head(0))
    candidate_metadata=base.resources(launch_group(0,1))
    separate()
    torch.cuda.synchronize()

    input_mib=sum(t.numel()*t.element_size() for t in states.values())/(1024**2)
    partial_mib=sum(t.numel()*t.element_size()
                    for t in (ref_pm,ref_pl,ref_acc,new_pm,new_pl,new_acc))/(1024**2)
    output={
        "heads":H, "tiles_per_head":tiles, "keys_per_head":keys,
        "splits":splits, "pinned_tile_resources":pinned_metadata,
        "grouped_tile_resources":candidate_metadata,
        "full_prepared_inputs_mib":input_mib,
        "full_two_partial_workspaces_mib":partial_mib,
        "estimated_group_input_mib_if_rebuilt_for_streaming":{
            str(n):input_mib*min(n,H)/H for n in group_sizes
        },
        "estimated_group_two_partial_workspaces_mib_if_rebuilt_for_streaming":{
            str(n):partial_mib*min(n,H)/H for n in group_sizes
        },
        "groups":[],
    }

    for size in group_sizes:
        candidate_metadata=base.resources(launch_group(0,min(size,H)))
        grouped(size)
        torch.cuda.synchronize()
        parity={}
        for label,ref,test in (
            ("partial_max",ref_pm,new_pm),
            ("partial_den",ref_pl,new_pl),
            ("partial_acc",ref_acc,new_acc),
        ):
            delta=test.float()-ref.float()
            rel=float(delta.norm()/ref.float().norm().clamp_min(1e-20))
            finite=bool(torch.isfinite(ref).all() and torch.isfinite(test).all())
            parity[label]={
                "relative_l2":rel,
                "max_abs":float(delta.abs().max()),
                "finite":finite,
                "pass":finite and rel<=tolerance,
            }
            del delta
        passed=all(x["pass"] for x in parity.values())
        result={
            "group_size":size, "launches":math.ceil(H/size),
            "last_group_size":H%size if H%size else size,
            "grid_ctas_per_full_group":size*tiles*splits,
            "resources":candidate_metadata,
            "parity":parity, "parity_pass":passed,
            "eager":None, "cuda_graph":None,
        }
        if not passed:
            output["groups"].append(result)
            continue

        for _ in range(warmups):
            separate()
            grouped(size)
        torch.cuda.synchronize()

        def pair_timing(reference, candidate):
            samples={"pinned_separate":[], "candidate_grouped":[]}
            for _ in range(rounds):
                for label in ("pinned_separate","candidate_grouped",
                              "candidate_grouped","pinned_separate"):
                    run=reference if label=="pinned_separate" else candidate
                    samples[label].append(base.timed_event(run,batch))
            med={key:statistics.median(vals) for key,vals in samples.items()}
            return {
                "median_ms":med,
                "speedup_pinned_over_grouped":
                    med["pinned_separate"]/med["candidate_grouped"],
                "samples_ms":samples,
            }

        result["eager"]=pair_timing(separate,lambda s=size:grouped(s))
        if with_graph:
            try:
                # Both graphs use pinned buffers, neither is allowed
                # to fall back to eager execution silently.
                ref_graph=torch.cuda.CUDAGraph()
                opt_graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(ref_graph):
                    separate()
                with torch.cuda.graph(opt_graph):
                    grouped(size)
                ref_graph.replay()
                opt_graph.replay()
                torch.cuda.synchronize()
                result["cuda_graph"]=pair_timing(ref_graph.replay,opt_graph.replay)
                del ref_graph,opt_graph
                gc.collect()
            except Exception as exc:
                result["cuda_graph"]={
                    "error":f"{type(exc).__name__}: {exc}",
                    "traceback_tail":traceback.format_exc()[-2200:],
                }
        output["groups"].append(result)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument(
        "--capture",
        default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt"
    )
    p.add_argument("--heads",type=int,default=56)
    p.add_argument("--repeat-available-heads",action="store_true")
    p.add_argument("--modes",default="fast")
    p.add_argument("--tiles",type=int,default=16)
    p.add_argument("--groups",default="1,2,4,7,8,14,28,56")
    p.add_argument("--rounds",type=int,default=5)
    p.add_argument("--warmups",type=int,default=3)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--tolerance",type=float,default=1e-4)
    p.add_argument("--no-graph",action="store_true")
    p.add_argument("--out",default=r"D:\ComfyOutput\H3\math_probe")
    args=p.parse_args()
    if not torch.cuda.is_available():
        p.error("NVIDIA GPU required")
    if any(n<1 for n in (args.heads,args.tiles,args.rounds,args.warmups,args.batch)):
        p.error("All positive integer parameters must be >=1")
    sizes=sorted({int(x) for x in args.groups.split(",") if x.strip()})
    if not sizes or sizes[0]<1 or sizes[-1]>args.heads:
        p.error("Groups must be 1..heads")
    modes=list(dict.fromkeys(x.strip() for x in args.modes.split(",") if x.strip()))
    if not modes or any(m not in ("fast","balanced") for m in modes):
        p.error("Supported modes: fast,balanced")
    path=Path(args.capture)
    if not path.is_file():
        p.error("Trusted saved capture path does not exist")
    cap=torch.load(path,map_location="cpu",mmap=True,weights_only=False)
    if not isinstance(cap,dict) or any(k not in cap
        for k in ("q_qhd","k_khd","v_khd","raw_head_index")):
        p.error("Expected native saved H3 Q/K/V and raw_head_index")
    source_head_map=cap["raw_head_index"]
    available=[int(x) for x in (
        source_head_map.tolist() if isinstance(source_head_map,torch.Tensor)
        else source_head_map
    )]
    if not available or len(set(available))!=len(available):
        p.error("Missing/duplicate raw head indices")
    if len(available)<args.heads:
        if not args.repeat_available_heads:
            p.error(
                f"Only {len(available)} distinct captured heads; requested {args.heads}. "
                "Enable --repeat-available-heads for a launch-only proxy, "
                "or provide full native capture."
            )
        chosen=[available[i%len(available)] for i in range(args.heads)]
        kind="repeated_capture_proxy"
    else:
        chosen=available[:args.heads]
        kind="native_distinct"
    report={
        "experiment":"h3_flat_tile_56_head_group_size_sweep_v1",
        "gpu":torch.cuda.get_device_name(),
        "capture":str(path),
        "heads_requested":args.heads,
        "captured_distinct_heads":len(available),
        "heads_used":chosen,
        "head_data_kind":kind,
        "is_repeated_capture_proxy":kind!="native_distinct",
        "modes":modes,"query_tiles_per_head":args.tiles,
        "group_sizes":sizes,
        "rounds":args.rounds,"event_batch_repetitions":args.batch,
        "source_commits":{
            "pinned_best":base.BEST,
            "candidate":subprocess.check_output(
                ["git","rev-parse","HEAD"],cwd=ROOT,text=True
            ).strip(),
        },
        "results":[],"failures":[],
        "notes":[
            "Only Tile stage, one query chunk: no leaf_stats/select/topk/merge.",
            "Full captured 40752 K rows per head; Q cropped to --tiles*64.",
            "All logical heads simultaneously prebuilt in memory.",
            "Per-group memory fields are estimates if a future implementation streams groups; not measured lower peak.",
            "Original pinned head launches vs grouped head-flattened Tile kernel from a clean BEST parent.",
            "GPU occupancy remains constrained by regs/shared, regardless of grid size.",
            "For real H3 throughput, evaluate selection, materialization, Merge and model residency.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="tile56_group_control_") as temp:
        pinned,pinned_runner=base.base_helper.original_modules(
            base.BEST,Path(temp)
        )
        current=base.base_helper.helpers.sw.load(
            "_tile56_group_candidate","h3/htrc_flat.py"
        )
        old_tr=pinned._load_triton()
        new_tr=current._load_triton()
        for mode in modes:
            states=None
            try:
                states,cfg,keys,nb,kx=base.make_states(
                    cap,chosen,args.tiles,mode,pinned,pinned_runner
                )
                row=compare_groups(
                    states,cfg,keys,nb,kx,args.tiles,
                    old_tr,new_tr,sizes,args.warmups,
                    args.rounds,args.batch,args.tolerance,
                    not args.no_graph
                )
                row["mode"]=mode
                report["results"].append(row)
                print(json.dumps({
                    "mode":mode,"heads":row["heads"],
                    "group_results":[{
                        "group_size":g["group_size"],
                        "launches":g["launches"],
                        "parity":g["parity_pass"],
                        "eager_speedup":g["eager"]["speedup_pinned_over_grouped"]
                            if g["eager"] else None,
                        "graph_speedup":g["cuda_graph"].get("speedup_pinned_over_grouped")
                            if g["cuda_graph"] else None,
                    } for g in row["groups"]],
                },indent=2),flush=True)
            except Exception as exc:
                report["failures"].append({
                    "mode":mode,
                    "error":f"{type(exc).__name__}: {exc}",
                    "traceback_tail":traceback.format_exc()[-3000:],
                })
            finally:
                del states
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    dest=out/f"flat_tile_56head_group_sweep_{time.strftime('%Y%m%d-%H%M%S')}.json"
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",dest)
    if not report["results"] or report["failures"] or any(
        not grp["parity_pass"]
        for r in report["results"] for grp in r["groups"]
    ):
        raise SystemExit("FAIL: candidate compile, parity, or benchmark (JSON saved)")


if __name__=="__main__":
    main()
