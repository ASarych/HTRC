"""Flat HTRC: test removing <=16-token tail computation from the main tile kernel.

Compare exact original (Git commit 571a7a3) vs the opt-in experiment:
  tile_tail_merge=True: compute trailing tokens in the EXISTING merge
  launch rather than in _tile_kernel. Same split partials, same K/V
  and the online-softmax merge identity.

Also compile original with tail=None as a RESOURCE CONTROL ONLY. It
intentionally omits seven tokens and is NOT a parity reference.

PowerShell from the isolated experimental worktree:
  & $py -B tools\benchmark_h3_flat_tail_merge.py --shapes 1942,18700,40759
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_flat_tail_exact_helpers",
    ROOT / "tools" / "benchmark_h3_flat_exact_baseline.py",
)
exact = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exact
SPEC.loader.exec_module(exact)


@contextmanager
def record_kernel(flat, name):
    """Record compiled resources for one named Triton kernel on warmup."""
    module = flat._load_triton()
    jit_kernel = getattr(module, name)
    observed = []

    class Recorder:
        def __getitem__(self, grid):
            launcher = jit_kernel[grid]

            def call(*args, **kwargs):
                compiled = launcher(*args, **kwargs)
                meta = getattr(compiled, "metadata", None)
                row = {
                    "n_regs": getattr(compiled, "n_regs", None),
                    "n_spills": getattr(compiled, "n_spills", None),
                    "shared_B": getattr(meta, "shared", None),
                    "num_warps": getattr(meta, "num_warps", None),
                }
                if row not in observed:
                    observed.append(row)
                return compiled
            return call

    setattr(module, name, Recorder())
    try:
        yield observed
    finally:
        setattr(module, name, jit_kernel)


def compare(original, original_runner, experimental, experiment_runner,
            data, mode, chunk, warmups, rounds, tolerance):
    q, k, v, tail = data
    old_cfg = exact.configuration(original, original_runner, mode, chunk)
    new_cfg = exact.configuration(experimental, experiment_runner, mode, chunk)
    if old_cfg != new_cfg:
        raise RuntimeError("Original and candidate PROD / mode differ")
    if tail is None or not 0 < len(tail[0]) <= 16:
        raise RuntimeError("Tail merge test requires 1-16 saved tail tokens")
    new_cfg["kern"].update(
        tile_passes="fused",
        tile_late_near_q=False,
        tile_tail_merge=True,
    )
    old_ws, new_ws, diagnostic_ws = {}, {}, {}

    def original_run(with_tail=True):
        return original.flat_attention(
            q, k, v, tail=tail if with_tail else None,
            ws=old_ws if with_tail else diagnostic_ws, **old_cfg
        )

    def experiment_run():
        return experimental.flat_attention(
            q, k, v, tail=tail, ws=new_ws, **new_cfg
        )

    with record_kernel(original, "_tile_kernel") as old_tile:
        original_run()
    with record_kernel(original, "_tile_kernel") as no_tail_tile:
        original_run(with_tail=False)
    with record_kernel(experimental, "_tile_kernel") as new_tile:
        with record_kernel(experimental, "_merge_tail_kernel") as new_merge:
            experiment_run()

    compiled = {
        "original_tail_on_tile": list(old_tile),
        "original_tail_off_tile": list(no_tail_tile),
        "experimental_tail_off_tile": list(new_tile),
        "experimental_merge_with_tail": list(new_merge),
    }

    for _ in range(warmups):
        original_run()
        experiment_run()
    torch.cuda.synchronize()

    expected = original_run().clone()
    candidate = experiment_run().clone()
    delta = expected.float() - candidate.float()
    rel_l2 = float(delta.norm() / expected.float().norm().clamp_min(1e-20))
    max_abs = float(delta.abs().max())
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(candidate).all())
    del expected, candidate, delta

    samples = {"original": [], "tail_in_merge": []}
    for _ in range(rounds):
        for name in ("original", "tail_in_merge", "tail_in_merge", "original"):
            fn = original_run if name == "original" else experiment_run
            samples[name].append(exact.helpers.event_ms(fn))
    medians = {key: statistics.median(values) for key, values in samples.items()}
    return {
        "q": int(len(q)),
        "k": int(len(k)),
        "tail": int(len(tail[0])),
        "chunk": chunk,
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "finite": finite,
        "parity_pass": finite and rel_l2 <= tolerance,
        "compiled_kernel_resources": compiled,
        "median_ms": medians,
        "speedup_original_over_tail_merge": (
            medians["original"] / medians["tail_in_merge"]
        ),
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", default=r"D:\ComfyOutput\H3\math_probe\**\*.pt")
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--shapes", default="1942,18700,40759")
    parser.add_argument("--mode", choices=("fast", "balanced", "accurate", "max"),
                        default="fast")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    args = parser.parse_args()
    shapes = sorted({int(x.strip()) for x in args.shapes.split(",") if x.strip()})
    if not shapes or shapes[0] < 16:
        parser.error("shapes must include token counts >= 16")
    if min(args.chunk, args.warmups, args.rounds) < 1:
        parser.error("chunk, warmups and rounds must be positive")

    source, capture = exact.helpers.select_capture(
        args.captures, args.head, max(shapes)
    )
    import tempfile
    with tempfile.TemporaryDirectory(prefix="flat_tail_source_") as folder:
        old, old_runner = exact.original_modules(
            exact.ORIGINAL, Path(folder)
        )
        new = exact.helpers.sw.load("_tail_merge_operator", "h3/htrc_flat.py")
        new_runner = exact.helpers.sw.load(
            "_tail_merge_runner", "h3/htrc_flat_runner.py"
        )
        rows = []
        for size in shapes:
            data = exact.helpers.subset(capture, size)
            row = compare(
                old, old_runner, new, new_runner, data,
                args.mode, args.chunk, args.warmups, args.rounds,
                args.tolerance,
            )
            row["is_cropped_proxy"] = size != len(capture[0])
            rows.append(row)
            print(json.dumps(
                {k: v for k, v in row.items() if k != "samples_ms"},
                indent=2,
            ))

    report = {
        "experiment": "h3_flat_tail_in_merge_v1",
        "gpu": torch.cuda.get_device_name(),
        "original_commit": exact.ORIGINAL,
        "experiment_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_capture": str(source),
        "mode": args.mode,
        "notes": [
            "Original and experimental parity both INCLUDE all seven tail tokens.",
            "The original tail=None compile is diagnostic only and omits those tokens.",
            "Small shapes are cropped performance proxies, not native H3 captures.",
            "Compiled spill count should not be conflated with Nsight local-memory spilling requests.",
            "Original fused settings remain the production default.",
        ],
        "rows": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"flat_tail_merge_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT:", path)
    if not all(row["parity_pass"] for row in rows):
        raise SystemExit("FAIL: tail merge numeric parity")


if __name__ == "__main__":
    main()
