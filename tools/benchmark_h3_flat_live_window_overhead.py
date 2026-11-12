"""Whole-call wall time of the live Flat backend vs query-window size, budget and staging.

Synthetic random BF16 Q/K/V in the Priestess video shape by default (18,723 tokens,
56 distinct heads, dim 128). Runs the real ``flat_attention_bthd`` code path, fixed
``fast`` math (256 exact leaves, centre group 1). No model weights, no quality claim.

Every case is compared against the first case for bitwise output equality, and the
number of implicit CUDA synchronizations inside one call is counted with
``torch.cuda.set_sync_debug_mode``. One JSON is written.

    python -B tools/benchmark_h3_flat_live_window_overhead.py
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3 import htrc_flat_live_backend as live  # noqa: E402

CASES = (
    # tiles, budget_mib, query_staging, kv_staging
    (16, 1024, "gather", "gather"),
    (32, 1024, "gather", "gather"),
    (64, 3072, "gather", "gather"),
    (16, 1024, "native_torch_mu", "native_direct"),
    (32, 1024, "native_torch_mu", "native_direct"),
    (64, 1024, "native_torch_mu", "native_direct"),
    (32, 1536, "native_torch_mu", "native_direct"),
    (32, 2048, "native_torch_mu", "native_direct"),
    (64, 2048, "native_torch_mu", "native_direct"),
)


def count_implicit_syncs(call):
    hits = collections.Counter()

    def show(_msg, _cat, _fn, _lineno, *_a):
        frames = [f for f in traceback.extract_stack() if "htrc_" in Path(f.filename).name]
        hits[" <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in frames[-2:])] += 1

    old = warnings.showwarning
    warnings.showwarning = show
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            call()
            torch.cuda.set_sync_debug_mode("default")
    finally:
        warnings.showwarning = old
    return dict(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=18723)
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe\flat_live_window_overhead.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("NVIDIA CUDA GPU required")
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1)
    shape = (1, args.heads, args.tokens, 128)
    q = torch.randn(shape, device=dev, generator=gen).to(torch.bfloat16)
    k = torch.randn(shape, device=dev, generator=gen).to(torch.bfloat16)
    v = torch.randn(shape, device=dev, generator=gen).to(torch.bfloat16)
    rows, ref = [], None
    for tiles, budget, qs, ks in CASES:
        kw = dict(heads=args.heads, tiles=tiles, budget_mib=budget, query_staging=qs, kv_staging=ks)
        row = {"tiles": tiles, "budget_mib": budget, "query_staging": qs, "kv_staging": ks}
        try:
            for _ in range(args.warmups):
                live.flat_attention_bthd(q, k, v, **kw)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            walls, hosts = [], []
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                out = live.flat_attention_bthd(q, k, v, **kw)
                t1 = time.perf_counter()
                torch.cuda.synchronize()
                walls.append((time.perf_counter() - t0) * 1000)
                hosts.append((t1 - t0) * 1000)
            seen = {}
            live.flat_attention_bthd(q, k, v, observer=lambda _n, d: seen.update(d), **kw)
            row.update(
                groups=seen["number_of_groups"], group_capacity=seen["group_capacity"],
                windows=seen["query_windows"], iterations=seen["query_windows"] * seen["number_of_groups"],
                wall_ms=round(statistics.median(walls), 2), host_return_ms=round(statistics.median(hosts), 2),
                peak_extra_mib=round((torch.cuda.max_memory_allocated() - base) / 2**20, 1),
                implicit_syncs=count_implicit_syncs(lambda: live.flat_attention_bthd(q, k, v, **kw)),
            )
            outf = out.float()
            if ref is None:
                ref = outf
            row["bitwise_equal_to_first_case"] = bool(torch.equal(outf, ref))
            row["finite"] = bool(torch.isfinite(outf).all())
            del out, outf
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "tool": Path(__file__).name, "gpu": torch.cuda.get_device_name(0),
        "tokens": args.tokens, "heads": args.heads, "mode": "fast", "rounds": args.rounds,
        "data": "synthetic random BF16 QKV; no model weights resident; not a video measurement",
        "note": "wall_ms = call + final synchronize; host_return_ms = time until the call returned",
        "cases": rows,
        "completed": all("error" not in r for r in rows),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("written", out_path)


if __name__ == "__main__":
    main()
