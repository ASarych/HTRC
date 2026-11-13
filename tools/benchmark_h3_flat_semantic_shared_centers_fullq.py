"""Full-Q GPU A/B: contiguous vs semantically ordered shared Flat centers.

Uses the EXISTING HTRC Flat kernels. This is K-means ORDERING + G adjacent
64-query tiles sharing one center, not exact MuSe global-cluster statistics.
All variants use exactly the same QKV, 16-key leaves, exact leaf budget,
Triton backend, and BF16/INT8 configuration. One JSON, no model inference.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

SOURCE_SHA = "32443521ddf3cd474020cbeba8ef97bdc5b21186"
SCRIPT_REV = "fullq_semantic_group_ab_v1"
DEFAULT_CAPTURE = r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt"
DEFAULT_OUTPUT = r"D:\ComfyOutput\H3\math_probe\flat_semantic_shared_centers_fullq_ab.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", default=DEFAULT_CAPTURE)
    p.add_argument("--out", default=DEFAULT_OUTPUT)
    p.add_argument("--heads", default="1,16,27,55")
    p.add_argument("--groups", default="4,8,16")
    p.add_argument("--tile-chunk", type=int, default=32)
    p.add_argument("--exact-leaves", type=int, default=256)
    p.add_argument("--eval-queries", type=int, default=256)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmups", type=int, default=1)
    return p.parse_args()


def load_flat():
    path = Path(__file__).with_name("htrc_flat.py")
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark dependency: {path}")
    spec = importlib.util.spec_from_file_location("_htrc_semantic_flat", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_capture(path):
    try:
        x = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, RuntimeError):
        x = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(x, dict):
        raise TypeError("Capture must be a dict of raw tensors")
    for key in ("q_qhd", "k_khd", "v_khd", "raw_head_index"):
        if key not in x:
            raise ValueError(f"Missing capture field: {key}")
    q, k, v = (x[key] for key in ("q_qhd", "k_khd", "v_khd"))
    if not all(torch.is_tensor(t) and t.ndim == 3 for t in (q, k, v)):
        raise ValueError("Expected Q/K/V [tokens,heads,dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("This gate requires self-attention with full Q=K=V token count")
    if int(q.shape[2]) != 128:
        raise ValueError("Expected H3 D=128")
    meta = x.get("meta", {})
    tokens = int(meta.get("sequence_tokens", q.shape[0])) if isinstance(meta, dict) else int(q.shape[0])
    if int(q.shape[0]) != tokens or int(q.shape[0]) < 4096:
        raise ValueError(f"Not full-Q capture: q={q.shape[0]}, expected={tokens}. Do not benchmark sampled Q.")
    rawheads = x["raw_head_index"]
    rawheads = rawheads.tolist() if isinstance(rawheads, torch.Tensor) else list(rawheads)
    if len(rawheads) != int(q.shape[1]):
        raise ValueError("raw_head_index does not match QKV head count")
    return x, [int(i) for i in rawheads]


def relmetrics(out, ref):
    out, ref = out.float(), ref.float()
    diff = out - ref
    denom = torch.linalg.vector_norm(ref).clamp_min(1e-20)
    row = torch.linalg.vector_norm(diff, dim=-1) / torch.linalg.vector_norm(ref, dim=-1).clamp_min(1e-20)
    return {
        "global_rel_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
        "mean_row_rel_l2": float(row.mean().item()),
        "p90_row_rel_l2": float(torch.quantile(row, 0.9).item()),
        "max_row_rel_l2": float(row.max().item()),
        "mean_cosine": float(F.cosine_similarity(out, ref, dim=-1).mean().item()),
        "finite": bool(torch.isfinite(out).all().item()),
    }


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def run_one(flat, q, k_main, v_main, tail, order, group, args):
    cfg = dict(flat.PROD)
    cfg.update(
        tile=64, exact_leaves=int(args.exact_leaves),
        order=order, tile_chunk=int(args.tile_chunk),
        sink_leaves=0, tail=tail, prof=None,
    )
    cfg["kern"] = dict(cfg["kern"], group=int(group))
    return flat.flat_attention(q, k_main, v_main, **cfg)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; run with the ComfyUI CUDA Python environment")
    import triton
    torch.set_grad_enabled(False)
    flat = load_flat()
    x, physical = load_capture(Path(args.capture))
    requested = [int(s.strip()) for s in args.heads.split(",") if s.strip()]
    available = [h for h in requested if h in physical]
    if not available:
        raise ValueError(f"No requested heads in capture. Available: {physical}")
    groups = sorted(set(int(s) for s in args.groups.split(",") if s.strip()))
    if not groups or min(groups) < 2 or max(groups) > 32:
        raise ValueError("groups must lie in [2,32]")
    configs = [("seq_g1", "seq", 1)]
    for group in groups:
        configs.append((f"seq_g{group}", "seq", group))
        configs.append((f"semantic_g{group}", "kmeans", group))

    qkv = (x[key] for key in ("q_qhd", "k_khd", "v_khd"))
    q_cpu, k_cpu, v_cpu = qkv
    N = int(q_cpu.shape[0])
    n_leaves = N // 16
    sample = min(int(args.eval_queries), N)
    if sample < 32:
        raise ValueError("eval-queries must be at least 32")
    eval_idx = torch.linspace(0, N - 1, sample, device="cuda").round().long().unique()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": SCRIPT_REV,
        "completed": False,
        "source_sha": SOURCE_SHA,
        "capture": str(Path(args.capture).resolve()),
        "capture_block": x.get("block_index"),
        "capture_call": x.get("forward_call"),
        "raw_full_queries": bool(x.get("meta", {}).get("raw_full_queries", False)),
        "sequence_tokens": N,
        "leaf_count": n_leaves,
        "captured_heads": physical,
        "tested_heads": available,
        "n_eval_queries": int(eval_idx.numel()),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "precision": "Pinned HTRC Flat PROD bf16 QKV, INT8 near K/far grad, FP32 accumulated output",
        "matching": "same exact leaf count per query and same number of shared centers for seq_gG and semantic_gG",
        "semantic_method": "EXISTING kmeans_order(): Lloyd K-means on 4096 Q sample, reorder all Q, then fixed groups of G adjacent tiles share one center; not global MuSe centroid mapping",
        "timing": "synchronized Python wall+CUDA for complete per-head attention, including K-means/sorting/preparation, warmup excluded",
        "dense_reference": "BF16 torch scaled_dot_product_attention on sampled full-Q rows against ALL current-head K/V",
        "limitations": [
            "Not full model denoising, not Sol A/B, no model checkpoint",
            "Does not reassign arbitrary nonadjacent tiles to exactly one global K-means centroid",
            "Semantic sorting alters exact-leaf selection groups in addition to center sharing",
            "GPU validation local only; standalone script is not a production patch",
        ],
        "settings": {"exact_leaves":args.exact_leaves, "tile":64, "tile_chunk":args.tile_chunk,
                     "groups":groups, "rounds":args.rounds, "warmups":args.warmups},
        "rows": [],
        "summary": [],
        "errors": [],
    }
    def save():
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str)+"\n", encoding="utf-8")

    save()
    try:
        for raw_head in available:
            index = physical.index(raw_head)
            print(f"HEAD {raw_head}: copying full QKV to GPU ...", flush=True)
            q = q_cpu[:,index,:].to("cuda", non_blocking=False).contiguous()
            k = k_cpu[:,index,:].to("cuda", non_blocking=False).contiguous()
            v = v_cpu[:,index,:].to("cuda", non_blocking=False).contiguous()
            split = n_leaves * 16
            kmain, vmain = k[:split].contiguous(), v[:split].contiguous()
            tail = (k[split:].contiguous(), v[split:].contiguous()) if split != N else None
            qsel = q.index_select(0, eval_idx)
            ref, dense_ms = timed(lambda: F.scaled_dot_product_attention(
                qsel[None,None], k[None,None], v[None,None], dropout_p=0, is_causal=False
            ).squeeze(0).squeeze(0).float())
            base_eval = None
            head_rows = []
            for name, order, group in configs:
                print(f"  {name}: warmups={args.warmups}, rounds={args.rounds}", flush=True)
                for _ in range(args.warmups):
                    y = run_one(flat, q, kmain, vmain, tail, order, group, args)
                    if not torch.isfinite(y).all().item():
                        raise RuntimeError(f"{name} non-finite output at warmup")
                    del y
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated()
                times = []
                y = None
                for _ in range(args.rounds):
                    if y is not None: del y
                    y, elapsed = timed(lambda: run_one(flat, q, kmain, vmain, tail, order, group, args))
                    times.append(elapsed)
                peak = torch.cuda.max_memory_allocated()
                selected = y.index_select(0, eval_idx).float()
                err = relmetrics(selected, ref)
                if not err["finite"]:
                    raise RuntimeError(f"{name} non-finite result")
                if name == "seq_g1":
                    base_eval = selected.detach().clone()
                change = relmetrics(selected, base_eval) if base_eval is not None else None
                del selected, y
                gc.collect()
                profiler = {}
                # Diagnostic substage timings are SYNC-instrumented and NOT used for speedup.
                cfg = dict(flat.PROD)
                cfg.update(tile=64, exact_leaves=args.exact_leaves, order=order,
                           tile_chunk=args.tile_chunk, sink_leaves=0,
                           tail=tail, prof=profiler)
                cfg["kern"] = dict(cfg["kern"], group=group)
                pp = flat.flat_attention(q,kmain,vmain,**cfg)
                torch.cuda.synchronize()
                del pp
                ncenters = math.ceil(math.ceil(N/64) / group)
                row = {
                    "head":raw_head,"case":name,"order":order,"group":group,
                    "shared_center_count":ncenters,"stats_center_leaf_pairs":ncenters*n_leaves,
                    "median_total_wall_ms":statistics.median(times),
                    "total_wall_samples_ms":times,
                    "dense_ref_ms":dense_ms,
                    "peak_incremental_alloc_mib":max(0,peak-allocated)/1024**2,
                    "error_vs_dense":err,
                    "error_vs_seq_g1":change,
                    "diagnostic_sync_stage_ms":profiler,
                }
                head_rows.append(row)
                report["rows"].append(row)
                save()
                print(f"    wall {row['median_total_wall_ms']:.2f} ms; dense rel-L2 {err['global_rel_l2']:.4f}; stats pairs {row['stats_center_leaf_pairs']}",flush=True)
            seq0 = head_rows[0]
            for row in head_rows:
                row["speedup_vs_seq_g1"] = seq0["median_total_wall_ms"] / row["median_total_wall_ms"]
            del base_eval, ref, qsel, q, k, v, kmain, vmain, tail
            gc.collect()
            torch.cuda.empty_cache()
            save()
        for name, order, group in configs:
            rows = [r for r in report["rows"] if r["case"] == name]
            report["summary"].append({
                "case":name,"center_count":rows[0]["shared_center_count"],
                "stats_center_leaf_pairs":rows[0]["stats_center_leaf_pairs"],
                "heads":len(rows),
                "mean_dense_global_rel_l2":statistics.mean(r["error_vs_dense"]["global_rel_l2"] for r in rows),
                "median_wall_ms_per_head":statistics.median(r["median_total_wall_ms"] for r in rows),
                "mean_wall_ms_per_head":statistics.mean(r["median_total_wall_ms"] for r in rows),
                "speedup_sum_wall_vs_seq_g1":sum(r["median_total_wall_ms"] for r in report["rows"] if r["case"] == "seq_g1") / sum(r["median_total_wall_ms"] for r in rows),
            })
        report["completed"] = True
        save()
        print(f"DONE: {out_path}",flush=True)
    except Exception as e:
        report["errors"].append(f"{type(e).__name__}: {e}")
        report["traceback"] = traceback.format_exc()
        save()
        print(f"FAILED: {e}; partial JSON: {out_path}",flush=True)
        raise


if __name__ == "__main__":
    main()
