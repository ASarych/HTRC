"""56 native H3 heads, ONE fused Tile launch vs 56 pinned Tile launches.

This test isolates the _tile_kernel stage. For every recorded raw head
it builds *the same* native K/V, leaf stats, exact-leaf topk/exclusions
and query tile input, then compares:
  pinned e47cc7a: H _tile_kernel launches, one for each head;
  candidate: 1 _tile_kernel launch across H heads, flattened grid.
Both paths use TAIL=0 as in verified tail-in-merge R32; no merge
runs in this test. Partial m/l/acc output parity is checked.

Benchmark both eager CUDA event and (when supported) captured
CUDA Graph replay to separate CPU enqueue overhead from GPU-work
differences. Inputs and preprocessing are outside timed regions.

Not a 56-head whole-operator single kernel. It does not include
stats, select, torch.topk, merge, input transfers or denoising.

PowerShell (standalone process; unload the big H3 checkpoint):
  & $py -B tools\benchmark_h3_flat_tile_56head_launch.py --modes fast --tiles 16,64
"""
from __future__ import annotations

import argparse
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
BEST = "e47cc7a310936547ab6c7d97bd07526eb6a135fe"
SPEC = importlib.util.spec_from_file_location(
    "_tile56_baseline_helper", ROOT / "tools" / "benchmark_h3_flat_exact_baseline.py"
)
base_helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base_helper
SPEC.loader.exec_module(base_helper)
sweep = base_helper.helpers.sw


def timed_event(fn, repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / repetitions


def resources(compiled):
    meta = getattr(compiled, "metadata", None)
    return {
        "n_regs": getattr(compiled, "n_regs", None),
        "n_spills": getattr(compiled, "n_spills", None),
        "shared_B": getattr(meta, "shared", None),
        "warps": getattr(meta, "num_warps", None),
    }


def make_states(capture, heads, tiles, mode, pinned_flat, pinned_runner):
    """Materialize one native query chunk for ALL heads; no synthetic logits."""
    cfg = base_helper.configuration(pinned_flat, pinned_runner, mode, tiles)
    if cfg.get("order") != "seq" or cfg["tile"] != 64:
        raise ValueError("Only fast/balanced, seq, 64-row tiles are supported")
    if cfg.get("store") != "int8" or cfg.get("dot") != "bf16":
        raise ValueError("Requires pinned INT8 attention math")
    nb = int(capture["k_khd"].shape[0]) // 16
    nfull = nb * 16
    dim = int(capture["q_qhd"].shape[-1])
    tile = int(cfg["tile"])
    if dim != 128 or tiles * tile > capture["q_qhd"].shape[0]:
        raise ValueError("Probe supports dim128 and full non-padded query tiles")
    hcount = len(heads)
    kx = min(int(cfg["exact_leaves"]), nb)
    dev = torch.device("cuda")
    opts = dict(device=dev)
    Q = torch.empty((hcount, tiles, tile, dim), dtype=torch.float32, **opts)
    MU = torch.empty((hcount, tiles, dim), dtype=torch.float32, **opts)
    K = torch.empty((hcount, nfull, dim), dtype=torch.int8, **opts)
    V = torch.empty((hcount, nfull, dim), dtype=torch.bfloat16, **opts)
    KS = torch.empty((hcount, nfull), dtype=torch.float32, **opts)
    G = torch.empty((hcount, tiles, nb, dim), dtype=torch.int8, **opts)
    VB = torch.empty((hcount, tiles, nb, dim), dtype=torch.bfloat16, **opts)
    LZ = torch.empty((hcount, tiles, nb), dtype=torch.float32, **opts)
    SG = torch.empty_like(LZ)
    EXCL = torch.empty((hcount, tiles, nb), dtype=torch.int8, **opts)
    IDX = torch.empty((hcount, tiles, kx), dtype=torch.int32, **opts)
    row_for_raw_head = {
        int(h): i for i, h in enumerate(capture["raw_head_index"].tolist())
    }
    tr = pinned_flat._load_triton()
    for h, raw in enumerate(heads):
        source_index = row_for_raw_head[raw]
        qs = capture["q_qhd"][: tiles * tile, source_index].to(dev).float().contiguous()
        Q[h].copy_(qs.reshape(tiles, tile, dim))
        MU[h].copy_(Q[h].mean(dim=1))
        kbf = capture["k_khd"][:nfull, source_index].to(dev).to(torch.bfloat16).contiguous()
        vbf = capture["v_khd"][:nfull, source_index].to(dev).to(torch.bfloat16).contiguous()
        V[h].copy_(vbf)
        kf = kbf.float()
        scale_k = (kf.abs().amax(-1) * (1.0 / 127.0)).clamp_min_(1e-20).contiguous()
        KS[h].copy_(scale_k)
        K[h].copy_((kf / scale_k[:, None]).round_().clamp_(-127, 127).to(torch.int8))
        state_ws = {}
        g, lz, vb, sg, _ = tr.leaf_stats(
            MU[h], kbf, vbf, nb, 1.0 / math.sqrt(dim), leaf=16,
            prec="tf32", store=torch.int8, bt=16, bl=4, warps=4, ws=state_ws
        )
        G[h].copy_(g)
        VB[h].copy_(vb)
        LZ[h].copy_(lz)
        SG[h].copy_(sg)
        sel = tr.rowmax_select(
            Q[h], MU[h], G[h], LZ[h], prec="tf32",
            bn=32, warps=4, splits=int(cfg["splits"]), stride=1, dot=1,
            sg=SG[h], ws=state_ws
        )
        indices = torch.topk(sel, kx, dim=-1, sorted=False).indices
        IDX[h].copy_(indices.to(torch.int32))
        EXCL[h].zero_().scatter_(1, indices, 1)
        del qs, kbf, vbf, kf, scale_k, state_ws, sel, indices
    torch.cuda.synchronize()
    return {
        "Q":Q,"MU":MU,"K":K,"V":V,"KS":KS,"G":G,"VB":VB,
        "LZ":LZ,"SG":SG,"EXCL":EXCL,"IDX":IDX
    }, cfg, nfull, nb, kx


def compare_batch(states, cfg, keys, nb, kx, tiles, old_tr, new_tr,
                  warmups, rounds, batch, tolerance):
    H = int(states["Q"].shape[0])
    T = int(cfg["tile"])
    D = int(states["Q"].shape[-1])
    ns = int(cfg["splits"])
    kern = dict(cfg["kern"])
    bn = int(kern.get("tile_bn", 32))
    warps = int(kern.get("tile_warps", 4))
    stages = int(kern.get("tile_stages", 1))
    lps = old_tr._split(nb, ns, bn)
    eps = old_tr._split(kx * 16, ns, bn)
    scale = float(1.0 / math.sqrt(D))
    partial_shape = (H, tiles, ns, T)
    acc_shape = partial_shape + (D,)
    old_pm = torch.empty(partial_shape, device="cuda", dtype=torch.float32)
    old_pl = torch.empty_like(old_pm)
    old_acc = torch.empty(acc_shape, device="cuda", dtype=torch.float32)
    new_pm = torch.empty_like(old_pm)
    new_pl = torch.empty_like(old_pm)
    new_acc = torch.empty_like(old_acc)
    kw = dict(
        T=T, D=D, BN=bn, BN_NEAR=bn, LEAF=16,
        PREC="tf32", DOT=1, GR=1, FP8=0, TAIL=0, RB=1, I8=1,
        PASS=0, LATE_NEAR_Q=0, FAR_PREFETCH_LZ=0,
        FAR_PV_ACC=0, NEAR_PV_ACC=0,
        num_warps=warps, num_stages=stages
    )
    q=states["Q"];mu=states["MU"];g=states["G"];lz=states["LZ"]
    vb=states["VB"];ex=states["EXCL"];idx=states["IDX"]
    k=states["K"];v=states["V"];ks=states["KS"];sg=states["SG"]

    def one_call(h):
        return old_tr._tile_kernel[(tiles,ns,1)](
            q[h],mu[h],g[h],lz[h],vb[h],ex[h],idx[h],
            k[h],v[h],old_pm[h],old_pl[h],old_acc[h],
            sg[h],lz[h],lz[h],lz[h],ks[h],
            nb,kx,lps,eps,0,scale, **kw
        )

    flat = lambda name: states[name].view((-1,)+states[name].shape[2:])
    qf=flat("Q"); muf=flat("MU");gf=flat("G"); lzf=flat("LZ")
    vbf=flat("VB"); exf=flat("EXCL"); idf=flat("IDX"); sgf=flat("SG")
    kf=k.view(H*keys,D);vf=v.view(H*keys,D);ksf=ks.view(H*keys)
    pmf=new_pm.view(H*tiles,ns,T)
    plf=new_pl.view(H*tiles,ns,T)
    accf=new_acc.view(H*tiles,ns,T,D)

    def all_one():
        return new_tr._tile_kernel[(H*tiles,ns,1)](
            qf,muf,gf,lzf,vbf,exf,idf,
            kf,vf,pmf,plf,accf,
            sgf,lzf,lzf,lzf,ksf,
            nb,kx,lps,eps,0,scale,
            BATCH_TILES_PER_HEAD=tiles, BATCH_KEY_ROWS=keys, **kw
        )

    def many():
        for h in range(H):
            one_call(h)

    # Compile BOTH variants, then synchronize before graph capture or timers.
    meta_ref=one_call(0)
    meta_batch=all_one()
    many()
    torch.cuda.synchronize()
    parity={}
    for name, a, b in (
        ("partial_max",old_pm,new_pm),
        ("partial_den",old_pl,new_pl),
        ("partial_acc",old_acc,new_acc),
    ):
        delta=a.float()-b.float()
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        rel=float(delta.norm()/a.float().norm().clamp_min(1e-20))
        parity[name]={
            "relative_l2":rel, "max_abs":float(delta.abs().max()),
            "finite":finite,"pass":finite and rel<=tolerance,
        }
    valid=all(entry["pass"] for entry in parity.values())
    result={
        "heads":H,"tiles_per_head":tiles,"queries_per_head":tiles*T,
        "keys_per_head":keys,"near_leaves":kx,
        "split_count":ns,
        "grid_sequential_tile":(tiles,ns,1),
        "grid_all_heads_tile":(H*tiles,ns,1),
        "tile_launches_sequential":H,
        "tile_launches_batch":1,
        "compiled_tile_resources":{
            "pinned_per_head":resources(meta_ref),
            "single_56head_launch":resources(meta_batch),
        },
        "parity":parity,
        "parity_pass":valid,
        "eager":None,"cuda_graph":None,
        "memory_mib_prepared_inputs":sum(
            t.numel()*t.element_size() for t in states.values()
        )/(1024**2),
        "memory_mib_outputs":sum(
            t.numel()*t.element_size()
            for t in (old_pm,old_pl,old_acc,new_pm,new_pl,new_acc)
        )/(1024**2),
    }
    if not valid:
        return result

    def timed_pair(ref, candidate):
        times={"56_separate":[],"one_batched":[]}
        for _ in range(rounds):
            for label in ("56_separate","one_batched","one_batched","56_separate"):
                fn=ref if label=="56_separate" else candidate
                times[label].append(timed_event(fn,batch))
        med={name:statistics.median(samples) for name,samples in times.items()}
        return {
            "medians_ms":med,
            "speedup_separate_over_batched":med["56_separate"]/med["one_batched"],
            "samples_ms":times,
        }

    for _ in range(warmups):
        many();all_one()
    torch.cuda.synchronize()
    result["eager"]=timed_pair(many,all_one)
    try:
        g_many=torch.cuda.CUDAGraph()
        g_one=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_many):
            many()
        with torch.cuda.graph(g_one):
            all_one()
        g_many.replay();g_one.replay()
        torch.cuda.synchronize()
        result["cuda_graph"]=timed_pair(g_many.replay,g_one.replay)
    except Exception as exc:
        result["cuda_graph"]={
            "error":f"{type(exc).__name__}: {exc}",
            "traceback_tail":traceback.format_exc()[-1800:],
        }
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--capture",
        default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt"
    )
    ap.add_argument("--heads",type=int,default=56)
    ap.add_argument(
        "--repeat-available-heads",action="store_true",
        help=("For launch-pressure PROXY only: if the capture has fewer "
              "unique heads, cycle those captured heads up to --heads. "
              "These are NOT 56 distinct native model heads.")
    )
    ap.add_argument("--modes",default="fast")
    ap.add_argument("--tiles",default="16,64")
    ap.add_argument("--rounds",type=int,default=6)
    ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--warmups",type=int,default=3)
    ap.add_argument("--tolerance",type=float,default=1e-4)
    ap.add_argument("--out",default=r"D:\ComfyOutput\H3\math_probe")
    args=ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("NVIDIA CUDA GPU required")
    paths=Path(args.capture)
    if not paths.is_file():
        ap.error("Trusted capture .pt file missing")
    tiles_list=sorted({int(v) for v in args.tiles.split(",") if v.strip()})
    modes=list(dict.fromkeys(m.strip() for m in args.modes.split(",") if m.strip()))
    if min(args.heads,args.rounds,args.warmups,args.batch) <1 or not tiles_list or min(tiles_list)<1:
        ap.error("heads/rounds/warmups/batch/tiles must be positive")
    if any(mode not in ("fast","balanced") for mode in modes):
        ap.error("This launch probe supports fast and balanced")
    cap=torch.load(paths,map_location="cpu",mmap=True,weights_only=False)
    if not isinstance(cap,dict) or any(k not in cap for k in ("q_qhd","k_khd","v_khd","raw_head_index")):
        ap.error("Saved capture lacks required head tensors")
    raw_map=cap["raw_head_index"]
    available=[int(v) for v in (raw_map.tolist() if isinstance(raw_map,torch.Tensor) else raw_map)]
    if not available or len(set(available)) != len(available):
        ap.error("Capture raw_head_index must be nonempty and contain distinct indices")
    captured_count=len(available)
    if captured_count < args.heads:
        if not args.repeat_available_heads:
            ap.error(
                f"Capture contains only {captured_count} distinct heads; "
                f"requested {args.heads}. Use --heads {captured_count} for "
                "a true captured-head test, --repeat-available-heads for "
                "a 56-work-item launch-only PROXY, or point --capture "
                "at a same-block file containing all requested heads."
            )
        selected=[available[i % captured_count] for i in range(args.heads)]
        sample_kind="repeated_captured_head_proxy"
        print(
            f"PROXY: {args.heads} logical heads from only "
            f"{captured_count} distinct captured heads. "
            "This tests batching/launch behavior, not 56-head model correctness.",
            flush=True
        )
    else:
        selected=available[:args.heads]
        sample_kind="native_distinct_heads"
        print(
            f"NATIVE: {len(selected)} distinct heads from one capture.",
            flush=True
        )
    current=base_helper.helpers.sw.load("_tile56_current_flat","h3/htrc_flat.py")
    rows=[]
    with tempfile.TemporaryDirectory(prefix="tile56_best_") as temp:
        pinned,runner=base_helper.original_modules(BEST,Path(temp))
        old_tr=pinned._load_triton()
        new_tr=current._load_triton()
        for mode in modes:
            for tiles in tiles_list:
                state=None
                try:
                    state,cfg,keys,nb,kx=make_states(
                        cap,selected,tiles,mode,pinned,runner
                    )
                    result=compare_batch(
                        state,cfg,keys,nb,kx,tiles,old_tr,new_tr,
                        args.warmups,args.rounds,args.batch,args.tolerance
                    )
                    result["mode"]=mode
                    rows.append(result)
                    print(json.dumps({
                        "mode":mode, "heads":result["heads"],
                        "tiles":result["tiles_per_head"],
                        "parity":result["parity"],
                        "compiled":result["compiled_tile_resources"],
                        "eager_speedup":result["eager"]["speedup_separate_over_batched"] if result["eager"] else None,
                        "graph_speedup":result["cuda_graph"].get("speedup_separate_over_batched") if result["cuda_graph"] else None,
                        "memory_mib":result["memory_mib_prepared_inputs"]+result["memory_mib_outputs"],
                    },indent=2))
                except Exception as exc:
                    row={
                        "mode":mode,"heads":len(selected),
                        "tiles_per_head":tiles,
                        "error":f"{type(exc).__name__}: {exc}",
                        "traceback_tail":traceback.format_exc()[-2500:],
                        "parity_pass":False,
                    }
                    rows.append(row)
                    print(json.dumps(row,indent=2))
                finally:
                    del state
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

    report={
        "experiment":"h3_flat_all_56_heads_single_tile_launch_v1",
        "gpu":torch.cuda.get_device_name(),
        "capture":str(paths),
        "source_commits":{
            "pinned_best":BEST,
            "candidate":subprocess.check_output(
                ["git","rev-parse","HEAD"],cwd=ROOT,text=True
            ).strip(),
        },
        "selected_raw_heads":selected,
        "captured_distinct_heads":captured_count,
        "logical_heads_tested":len(selected),
        "input_head_kind":sample_kind,
        "is_repeated_capture_proxy":sample_kind != "native_distinct_heads",
        "modes":modes,"tiles_per_head":tiles_list,
        "rounds":args.rounds,"batch":args.batch,
        "rows":rows,
        "notes":[
            "Exactly one GPU Tile kernel launch for all heads per query chunk.",
            "Control has 56 individual Tile launches from pinned BEST source.",
            "All inputs derive from one trusted capture; repetition, if enabled, is a launch-only proxy.",
            "Repeated captured heads are not 56 distinct native model heads.",
            "Leaf stats/topk are prepared identically but NOT timed.",
            "Only TAIL=0 Tile partials compared, NOT merge, whole Flat, or denoising.",
            "Both eager and CUDA Graph Tile-stage comparisons, if captures succeed.",
            "GPU VRAM test is standalone; full 56-head input/state residency must be assessed for H3 checkpoint coexistence.",
            "More CTAs does NOT increase per-SM register/shared-limited occupancy.",
        ]
    }
    output=Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    filename=output/f"flat_tile_56head_launch_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    filename.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",filename)
    if not rows or any(not r["parity_pass"] for r in rows):
        raise SystemExit("FAIL: batched Tile parity/compile; JSON saved")


if __name__=="__main__":
    main()
