"""Paired GPU-resident batched torch_gpu vs triton_gpu K8 backend A/B.

Both candidates use IDENTICAL grouped preparation/Tile/Merge, source
QKV, mode, group size and full output contract. Pinned BEST is only a
numerical reference and is not timed in this isolated backend A/B.
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
SPEC = importlib.util.spec_from_file_location(
    "_gpu_quant_full_pipeline", ROOT / "tools" / "benchmark_h3_flat_full_pipeline_batch.py"
)
full = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = full
SPEC.loader.exec_module(full)
MODES = ("torch_gpu", "triton_gpu")


def measure_group(capture, ids, tiles, mode, pinned, runner, tr_best, tr_new,
                  warmups, rounds, tolerance, first_group=False):
    def evaluate(quant, keep_output=False):
        return full.eval_one(
            capture, ids, tiles, mode, pinned, runner, tr_best, tr_new,
            "batched", keep_output=keep_output, quant_mode=quant
        )

    for _ in range(warmups):
        for quant in MODES:
            record, output = evaluate(quant)
            del record, output
    gc.collect()
    torch.cuda.empty_cache()

    ref_rec, reference = full.eval_one(
        capture, ids, tiles, mode, pinned, runner, tr_best, tr_new,
        "best", keep_output=True
    )
    preflight = {}
    cand_outputs = {}
    for quant in MODES:
        preflight[quant], cand_outputs[quant] = evaluate(quant, keep_output=True)
    parity = {
        "torch_vs_pinned": full.compare_outputs(reference, cand_outputs["torch_gpu"], tolerance),
        "triton_vs_pinned": full.compare_outputs(reference, cand_outputs["triton_gpu"], tolerance),
        "torch_vs_triton": full.compare_outputs(
            cand_outputs["torch_gpu"], cand_outputs["triton_gpu"], tolerance
        ),
    }
    oracles = []
    if first_group and all(g["pass"] for g in parity.values()):
        done = set()
        nfull = (int(capture["k_khd"].shape[0]) // 16) * 16
        for i, raw in enumerate(ids):
            if raw in done:
                continue
            independent = full.prior.full_pinned_output(
                pinned, runner, capture, raw, nfull, tiles, mode
            ).view(tiles, 64, 128)
            oracles.append({
                "raw_head": raw,
                "torch_gpu": full.compare_outputs(independent, cand_outputs["torch_gpu"][i], tolerance),
                "triton_gpu": full.compare_outputs(independent, cand_outputs["triton_gpu"][i], tolerance),
            })
            done.add(raw)
            del independent
    del reference, cand_outputs
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "head_count": len(ids), "parity": parity, "full_oracles": oracles,
        "preflight_records": {"pinned": ref_rec, **preflight},
        "timings": None,
    }
    if not all(g["pass"] for g in parity.values()) or any(
        not o[q]["pass"] for o in oracles for q in MODES
    ):
        return result

    samples = {m: [] for m in MODES}
    for _ in range(rounds):
        # Paired ABBA order controls drift without intermixing baseline timing.
        for quant in ("torch_gpu", "triton_gpu", "triton_gpu", "torch_gpu"):
            record, output = evaluate(quant)
            samples[quant].append(record)
            del record, output
    keys = ("preparation_wall_ms", "tail_tile_merge_wall_ms",
            "full_group_wall_ms", "incremental_peak_mib")
    medians = {m: {k: statistics.median(r[k] for r in samples[m])
                   for k in keys} for m in MODES}
    result["timings"] = {
        "median_by_path": medians,
        "speedup_torch_over_triton_full_wall":
            medians["torch_gpu"]["full_group_wall_ms"] /
            medians["triton_gpu"]["full_group_wall_ms"],
        "sample_records": samples,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt")
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--repeat-available-heads", action="store_true")
    ap.add_argument("--mode", choices=("fast", "balanced"), default="fast")
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--budget-mib", type=float, default=1024)
    ap.add_argument("--reserve-mib", type=float, default=256)
    ap.add_argument("--max-heads-per-group", type=int, default=None)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error("CUDA is required")
    if min(args.heads, args.tiles, args.rounds, args.warmups) < 1:
        ap.error("heads, tiles, rounds and warmups must be positive")
    if (not math.isfinite(args.budget_mib) or args.budget_mib <= 0
        or not math.isfinite(args.reserve_mib) or args.reserve_mib < 0):
        ap.error("invalid GPU memory budget or reserve")
    if args.max_heads_per_group is not None and args.max_heads_per_group < 1:
        ap.error("max-heads-per-group must be positive")
    capture_path = Path(args.capture)
    if not capture_path.is_file():
        ap.error("Trusted QKV capture not found")
    cap = torch.load(capture_path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(cap, dict) or any(k not in cap for k in
        ("raw_head_index", "q_qhd", "k_khd", "v_khd")):
        ap.error("Missing raw_head_index/q_qhd/k_khd/v_khd in capture")
    idx = cap["raw_head_index"]
    available = [int(x) for x in (idx.tolist() if isinstance(idx, torch.Tensor) else idx)]
    if not available or len(available) != len(set(available)):
        ap.error("Invalid raw head list")
    if args.heads > len(available) and not args.repeat_available_heads:
        ap.error("Requested more heads than captured; use --repeat-available-heads for proxy")
    selected = [available[i % len(available)] for i in range(args.heads)]
    proxy = args.heads > len(available)
    cap = dict(cap)
    for k in ("q_qhd", "k_khd", "v_khd"):
        cap[k] = cap[k].to("cuda").contiguous()
    torch.cuda.synchronize()
    cap_mib = sum(cap[k].numel() * cap[k].element_size()
                  for k in ("q_qhd", "k_khd", "v_khd")) / full.stage.MIB
    free, _ = torch.cuda.mem_get_info()
    budget = min(free, int(args.budget_mib * full.stage.MIB))
    nkeys = (int(cap["k_khd"].shape[0]) // 16) * 16
    plan = full.stage.plan_tile_head_batches(
        len(selected), free_bytes=budget, keys_per_head=nkeys,
        tile_count=args.tiles, exact_leaves=256 if args.mode == "fast" else 512,
        reserve_bytes=int(args.reserve_mib * full.stage.MIB),
        fixed_temp_bytes=128 * full.stage.MIB, safety_factor=1.5,
        max_heads_per_group=args.max_heads_per_group,
    )
    report = {
        "experiment": "h3_flat_gpu_resident_torch_vs_triton_K8_v1",
        "gpu": torch.cuda.get_device_name(),
        "capture": str(capture_path),
        "capture_gpu_resident_mib": cap_mib,
        "source_commits": {
            "pinned_best": full.base.BEST,
            "candidate": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        },
        "input_residency": "gpu", "input_kind": "repeated_capture_proxy" if proxy else "distinct_native",
        "raw_head_indices": selected, "mode": args.mode, "tiles": args.tiles,
        "planned": vars(plan), "quantization_exact_gates": [],
        "groups": [], "errors": [], "oom_group_retries": [],
        "completed": False,
        "notes": [
            "Paired ABBA within IDENTICAL batched preparation+Tile+Merge; quant mode alone differs.",
            "Pinned BEST is a correctness oracle only and excluded from timed A/B.",
            "Both paths consume same preloaded GPU QKV; no H2D in timed region.",
            "Sum of group medians is not one true full-model end-to-end timing.",
            "Four-native-head repeated proxy is not 56 distinct model heads.",
            "No model checkpoint, full Q tokens, graph, or video runtime.",
        ],
    }
    group_size = plan.heads_per_group
    pos = 0
    try:
        with tempfile.TemporaryDirectory(prefix="htrc_gpu_quant_best_") as tmp:
            pinned, runner = full.base.base_helper.original_modules(full.base.BEST, Path(tmp))
            cand = full.base.base_helper.helpers.sw.load(
                "_h3_gpu_quant_candidate", "h3/htrc_flat.py"
            )
            tr_best, tr_new = pinned._load_triton(), cand._load_triton()
            for raw in dict.fromkeys(selected):
                gate = full.quant_key_exactness_gate(cap, raw, tr_new)
                report["quantization_exact_gates"].append(gate)
                if not gate["pass"]:
                    raise RuntimeError("K8 quantization differs from PyTorch on raw head " + str(raw))
            while pos < len(selected):
                end = min(len(selected), pos + group_size)
                try:
                    row = measure_group(
                        cap, selected[pos:end], args.tiles, args.mode,
                        pinned, runner, tr_best, tr_new,
                        args.warmups, args.rounds, args.tolerance,
                        first_group=(pos == 0),
                    )
                    row["head_job_range"] = [pos, end]
                    row["raw_heads"] = selected[pos:end]
                    report["groups"].append(row)
                    if row["timings"] is None:
                        raise RuntimeError("Quant candidate parity/full oracle FAIL")
                    print("GROUP", pos, end, "torch/triton speedup",
                          row["timings"]["speedup_torch_over_triton_full_wall"], flush=True)
                    pos = end
                except torch.cuda.OutOfMemoryError as exc:
                    if group_size == 1:
                        raise
                    before = group_size
                    group_size = max(1, group_size // 2)
                    report["oom_group_retries"].append({"job": pos,
                        "from": before, "to": group_size, "message": str(exc)[:500]})
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
    except Exception as exc:
        report["errors"].append({"job": pos, "exception": type(exc).__name__,
                                 "error": str(exc)[:1800],
                                 "traceback_tail": traceback.format_exc()[-2500:]})

    report["completed"] = pos == len(selected) and not report["errors"]
    valid = [r for r in report["groups"] if r.get("timings")]
    if valid:
        totals = {q: sum(r["timings"]["median_by_path"][q]["full_group_wall_ms"]
                         for r in valid) for q in MODES}
        report["sum_of_group_median_wall_not_true_end_to_end"] = {
            "torch_gpu_ms": totals["torch_gpu"],
            "triton_gpu_ms": totals["triton_gpu"],
            "speedup_torch_over_triton": totals["torch_gpu"] / totals["triton_gpu"],
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"flat_gpu_quant_isolation_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT:", report_path, flush=True)
    if not report["completed"]:
        raise SystemExit("FAIL: backend-only GPU quant A/B (see JSON)")


if __name__ == "__main__":
    main()
