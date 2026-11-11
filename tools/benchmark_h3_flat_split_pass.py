"""Interleaved fused vs split-pass H3 Flat attention benchmark.

Use a real saved full-Q capture. Cropped 1942/18700 tests are scale proxies,
NOT proof of performance or image quality on a different H3 checkpoint.

PowerShell, in the ComfyNodes root:
  & $py -B tools\benchmark_h3_flat_split_pass.py --shapes 1942,18700,40759

No production activation or video inference is performed.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_flat_split_sweep", ROOT / "tools" / "sweep_h3_flat_kernels.py"
)
sw = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sw
SPEC.loader.exec_module(sw)


def select_capture(pattern, head, largest):
    for path, obj in sw.find_captures(pattern):
        data = sw.capture_head(obj, head, torch.device("cuda"))
        if data is not None and data[0].shape[0] >= largest:
            return path, data
    raise RuntimeError(
        f"No saved capture containing head {head} with >= {largest} queries; "
        "specify an appropriate --captures glob."
    )


def subset(data, n):
    q, k, v, tail = data
    main = (min(n, len(k)) // 16) * 16
    if main < 16 or n > len(q):
        raise ValueError(f"Invalid token count: {n}")
    return q[:n].contiguous(), k[:main].contiguous(), v[:main].contiguous(), tail


def event_ms(fn):
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    return float(a.elapsed_time(b))


def compare(flat, cfg, data, chunk, warmups, rounds, tolerance):
    q, k, v, tail = data
    settings = {}
    workspaces = {}
    for name in ("fused", "split"):
        c = copy.deepcopy(cfg)
        c["tile_chunk"] = chunk
        c["kern"] = dict(c.get("kern", {}), tile_passes=name)
        settings[name] = c
        workspaces[name] = {}

    def run(name):
        return flat.flat_attention(
            q, k, v, tail=tail, ws=workspaces[name], **settings[name]
        )

    for _ in range(warmups):
        run("fused")
        run("split")
    torch.cuda.synchronize()

    fused = run("fused").clone()
    split = run("split").clone()
    diff = fused.float() - split.float()
    rel_l2 = float(diff.norm() / fused.float().norm().clamp_min(1e-20))
    max_abs = float(diff.abs().max())
    finite = bool(torch.isfinite(fused).all() and torch.isfinite(split).all())
    del fused, split, diff

    times = {"fused": [], "split": []}
    # ABBA pairs help identify temperature/power/clock drift within one run.
    for _ in range(rounds):
        for name in ("fused", "split", "split", "fused"):
            times[name].append(event_ms(lambda name=name: run(name)))
    fm = statistics.median(times["fused"])
    sm = statistics.median(times["split"])
    return {
        "q": int(len(q)), "k": int(len(k)),
        "tail": 0 if tail is None else int(len(tail[0])),
        "chunk": chunk,
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "finite": finite,
        "parity_pass": finite and rel_l2 <= tolerance,
        "median_ms": {"fused": fm, "split": sm},
        "speedup_fused_over_split": fm / sm if sm > 0 else None,
        "samples_ms": times,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default=r"D:\ComfyOutput\H3\math_probe\**\*.pt")
    ap.add_argument("--shapes", default="1942,18700,40759")
    ap.add_argument("--head", type=int, default=1)
    ap.add_argument("--mode", default="fast")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    args = ap.parse_args()
    sizes = sorted({int(x.strip()) for x in args.shapes.split(",") if x.strip()})
    if not sizes or any(x < 16 for x in sizes):
        ap.error("Provide token counts >= 16")
    if min(args.chunk, args.rounds, args.warmups) <= 0:
        ap.error("chunk, rounds and warmups must be positive")

    path, data = select_capture(args.captures, args.head, max(sizes))
    flat = sw.load("_flat_split_operator", "h3/htrc_flat.py")
    runner = sw.load("_flat_split_runner", "h3/htrc_flat_runner.py")
    cfg = dict(flat.PROD)
    cfg.update(runner.MODES[args.mode])
    cfg["kern"] = dict(cfg.get("kern", {}))
    cfg["sink_leaves"] = 0

    results = []
    for n in sizes:
        row = compare(
            flat, cfg, subset(data, n), args.chunk,
            args.warmups, args.rounds, args.tolerance,
        )
        row["is_cropped_proxy"] = n != len(data[0])
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}, indent=2))

    report = {
        "experiment": "h3_flat_split_pass_v1",
        "gpu": torch.cuda.get_device_name(),
        "source_capture": str(path),
        "mode": args.mode,
        "notes": "Cropped small shapes are timing proxies, not native-model captures.",
        "rows": results,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"flat_split_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT:", report_path)
    if not all(x["parity_pass"] for x in results):
        raise SystemExit("FAIL: output parity")


if __name__ == "__main__":
    main()
