"""Exact Git-source original versus experimental Flat HTRC.

Uses git show to reconstruct the original source; no extra worktree or
local changes needed. Only the old Python Triton module-cache key is
renamed to prevent the two different modules colliding in sys.modules.
Kernel source, numerical math, and configuration remain unchanged.

PowerShell in standalone HTRC experimental worktree:
    & $py -B tools\benchmark_h3_flat_exact_baseline.py
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = "571a7a3e6cbc7c76112ca21ea5d4f2a21a3d649f"  # vendored ComfyNodes original reference

HELPER_SPEC = importlib.util.spec_from_file_location(
    "_flat_exact_baseline_helpers", ROOT / "tools" / "benchmark_h3_flat_split_pass.py"
)
helpers = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = helpers
HELPER_SPEC.loader.exec_module(helpers)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The historical ComfyNodes BEST is byte-identical for all three
# pinned source files to HTRC a8fe43e7. Use HTRC history for
# stand-alone reproducibility instead of requiring a second repo.
EXTERNAL_PINNED_BEST = "e47cc7a310936547ab6c7d97bd07526eb6a135fe"
HTRC_EQUIVALENT_BEST = "a8fe43e7aab8d9f951af7b4cba8fbd54f7a61e5c"

def read_git_blob(sha, path):
    if sha == "571a7a3e6cbc7c76112ca21ea5d4f2a21a3d649f":
        if not path.startswith("h3/"):
            raise RuntimeError("Vendored baseline path must start with h3/")
        name = path.split("/", 1)[1]
        if name not in ("htrc_flat.py", "htrc_flat_triton.py", "htrc_flat_runner.py"):
            raise RuntimeError(f"Unsupported vendored original source path: {path}")
        return (ROOT / "research" / "baselines" / "comfynodes_571a" / name).read_text(encoding="utf-8")
    source_ref = HTRC_EQUIVALENT_BEST if sha == EXTERNAL_PINNED_BEST else sha
    completed = subprocess.run(
        ["git", "show", f"{source_ref}:{path}"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Git source missing: {sha}:{path}; "
            + completed.stderr.decode("utf-8", "replace")
        )
    return completed.stdout.decode("utf-8")


def original_modules(sha, directory):
    for filename in ("htrc_flat.py", "htrc_flat_triton.py", "htrc_flat_runner.py"):
        src = read_git_blob(sha, f"h3/{filename}")
        if filename == "htrc_flat.py":
            marker = 'name = "_apl_htrc_flat_triton"'
            if src.count(marker) != 1:
                raise RuntimeError("Original Python module loader unexpectedly changed")
            # Namespace-only modification. Triton source and CUDA kernels untouched.
            src = src.replace(marker, 'name = "_apl_htrc_flat_triton_exact_orig"')
        (directory / filename).write_text(src, encoding="utf-8")
    return (
        load_module("_exact_flat_original", directory / "htrc_flat.py"),
        load_module("_exact_flat_original_runner", directory / "htrc_flat_runner.py"),
    )


@contextmanager
def compiled_resources(flat):
    """Measure compiled kernel resources on warmup, never on timed calls."""
    mod = flat._load_triton()
    original_kernel = mod._tile_kernel
    observed = []

    class Recorder:
        def __getitem__(self, grid):
            call = original_kernel[grid]

            def wrapped(*args, **kwargs):
                compiled = call(*args, **kwargs)
                metadata = getattr(compiled, "metadata", None)
                record = {
                    "n_regs": getattr(compiled, "n_regs", None),
                    "n_spills": getattr(compiled, "n_spills", None),
                    "shared_B": getattr(metadata, "shared", None),
                    "num_warps": getattr(metadata, "num_warps", None),
                }
                if record not in observed:
                    observed.append(record)
                return compiled
            return wrapped

    mod._tile_kernel = Recorder()
    try:
        yield observed
    finally:
        mod._tile_kernel = original_kernel


def configuration(flat, runner, mode, chunk):
    cfg = dict(flat.PROD)
    cfg.update(runner.MODES[mode])
    cfg["kern"] = dict(cfg["kern"])
    cfg["tile_chunk"] = chunk
    cfg["sink_leaves"] = 0
    return cfg


def compare(old, new, old_runner, new_runner, data,
            mode, chunk, near_bn, warmups, rounds, tolerance):
    q, k, v, tail = data
    original_cfg = configuration(old, old_runner, mode, chunk)
    experiment_cfg = configuration(new, new_runner, mode, chunk)
    if original_cfg != experiment_cfg:
        raise RuntimeError("Original and experimental PROD or mode defaults differ")
    experiment_cfg["kern"].update(
        tile_passes="fused",
        tile_late_near_q=False,
    )
    if near_bn is not None:
        experiment_cfg["kern"]["tile_near_bn"] = near_bn

    cfgs = {"original": original_cfg, "experiment": experiment_cfg}
    modules = {"original": old, "experiment": new}
    workspaces = {"original": {}, "experiment": {}}

    def run(name):
        return modules[name].flat_attention(
            q, k, v, tail=tail, ws=workspaces[name], **cfgs[name]
        )

    resources = {}
    for name in ("original", "experiment"):
        with compiled_resources(modules[name]) as observed:
            run(name)
        resources[name] = list(observed)

    for _ in range(warmups):
        run("original")
        run("experiment")
    torch.cuda.synchronize()
    expected = run("original").clone()
    result = run("experiment").clone()
    delta = expected.float() - result.float()
    rel_l2 = float(delta.norm() / expected.float().norm().clamp_min(1e-20))
    max_abs = float(delta.abs().max())
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(result).all())
    del expected, result, delta

    samples = {"original": [], "experiment": []}
    for _ in range(rounds):
        for name in ("original", "experiment", "experiment", "original"):
            samples[name].append(helpers.event_ms(lambda name=name: run(name)))
    medians = {key: statistics.median(values) for key, values in samples.items()}
    return {
        "q": int(len(q)),
        "k": int(len(k)),
        "tail": 0 if tail is None else int(len(tail[0])),
        "chunk": chunk,
        "near_bn": near_bn,
        "relative_l2": rel_l2,
        "max_abs": max_abs,
        "finite": finite,
        "parity_pass": finite and rel_l2 <= tolerance,
        "compiled_tile_resources": resources,
        "median_ms": medians,
        "speedup_original_over_experiment": (
            medians["original"] / medians["experiment"]
        ),
        "samples_ms": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default=ORIGINAL)
    ap.add_argument("--captures", default=r"D:\ComfyOutput\H3\math_probe\**\*.pt")
    ap.add_argument("--head", type=int, default=1)
    ap.add_argument("--shapes", default="1942,18700,40759")
    ap.add_argument("--mode", choices=("fast", "balanced", "accurate", "max"), default="fast")
    ap.add_argument("--near-bns", default="default", help="default,32,16")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--warmups", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.original):
        ap.error("--original must be an exact 40-character SHA")
    shapes = sorted({int(x) for x in args.shapes.split(",") if x.strip()})
    widths = [None if v.strip() == "default" else int(v.strip())
              for v in args.near_bns.split(",") if v.strip()]
    if not shapes or shapes[0] < 16:
        ap.error("--shapes must contain counts >= 16")
    if not widths or any(w not in (None, 16, 32) for w in widths):
        ap.error("--near-bns must contain default,32 or 16")
    if min(args.chunk, args.warmups, args.rounds) < 1:
        ap.error("--chunk, --warmups and --rounds must be positive")

    source, capture = helpers.select_capture(args.captures, args.head, max(shapes))
    with tempfile.TemporaryDirectory(prefix="flat_exact_source_") as scratch:
        old, old_runner = original_modules(args.original, Path(scratch))
        new = helpers.sw.load("_exact_flat_experiment", "h3/htrc_flat.py")
        new_runner = helpers.sw.load(
            "_exact_flat_experiment_runner", "h3/htrc_flat_runner.py"
        )
        rows = []
        for q_count in shapes:
            data = helpers.subset(capture, q_count)
            for near_bn in widths:
                row = compare(
                    old, new, old_runner, new_runner, data,
                    args.mode, args.chunk, near_bn,
                    args.warmups, args.rounds, args.tolerance,
                )
                row["is_cropped_proxy"] = q_count != len(capture[0])
                rows.append(row)
                print(json.dumps(
                    {key: value for key, value in row.items()
                     if key != "samples_ms"}, indent=2,
                ))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        stdout=subprocess.PIPE, text=True, check=True,
    ).stdout.strip()
    report = {
        "experiment": "h3_flat_exact_git_source_baseline_v1",
        "gpu": torch.cuda.get_device_name(),
        "original_commit": args.original,
        "experiment_commit": commit,
        "source_capture": str(source),
        "mode": args.mode,
        "notes": [
            "Baseline from exact Git objects; only Python loader key renamed.",
            "Original and candidate Triton kernels use independent module caches.",
            "Cropped 1942 and 18700 shapes are performance proxies.",
            "Compiled resource data are not Nsight occupancy measurements.",
        ],
        "rows": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"flat_exact_baseline_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT:", destination)
    if not all(r["parity_pass"] for r in rows):
        raise SystemExit("FAIL: parity against exact original")


if __name__ == "__main__":
    main()
