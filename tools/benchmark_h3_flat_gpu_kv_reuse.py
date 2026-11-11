"""K/V-preparation reuse across the full Q axis, with one final JSON.

Uncached control and cached candidate both use fused Triton K8 and the
same grouped Tile/Merge. Only K/V staging + quant frequency changes.
Pinned BEST and independent full Flat are numerical oracles only.
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
    "_htrc_kv_reuse_full", ROOT / "tools" / "benchmark_h3_flat_full_pipeline_batch.py"
)
full = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = full
SPEC.loader.exec_module(full)
MODES = ("uncached", "reuse")


def paired_window(cap, ids, cache, tiles, mode, pinned, runner, best_tr, new_tr,
                  rounds, warmups, tolerance, independent=False):
    def evaluate(which, output=False):
        return full.eval_one(
            cap, ids, tiles, mode, pinned, runner, best_tr, new_tr,
            "batched", keep_output=output, quant_mode="triton_gpu",
            kv_cache=cache if which == "reuse" else None
        )

    for _ in range(warmups):
        for which in MODES:
            item, out = evaluate(which)
            del item, out
    gc.collect()
    torch.cuda.empty_cache()

    ref_record, ref = full.eval_one(
        cap, ids, tiles, mode, pinned, runner,
        best_tr, new_tr, "best", keep_output=True
    )
    records, outputs = {}, {}
    for which in MODES:
        records[which], outputs[which] = evaluate(which, output=True)
    comparisons = {
        "uncached_vs_pinned": full.compare_outputs(ref, outputs["uncached"], tolerance),
        "reuse_vs_pinned": full.compare_outputs(ref, outputs["reuse"], tolerance),
        "reuse_vs_uncached": full.compare_outputs(outputs["uncached"], outputs["reuse"], tolerance),
    }
    independent_checks = []
    if independent and all(value["pass"] for value in comparisons.values()):
        nfull = (int(cap["k_khd"].shape[0]) // 16) * 16
        visited = set()
        for i, raw in enumerate(ids):
            if raw in visited:
                continue
            oracle = full.prior.full_pinned_output(
                pinned, runner, cap, raw, nfull, tiles, mode
            ).view(tiles, 64, 128)
            independent_checks.append({
                "raw_head": raw,
                "uncached": full.compare_outputs(oracle, outputs["uncached"][i], tolerance),
                "reuse": full.compare_outputs(oracle, outputs["reuse"][i], tolerance),
            })
            visited.add(raw)
            del oracle
    del ref, outputs
    gc.collect()
    torch.cuda.empty_cache()

    row = {
        "parity": comparisons,
        "independent_full_flat_oracles": independent_checks,
        "preflight": {"pinned": ref_record, **records},
        "timings": None,
    }
    if not all(value["pass"] for value in comparisons.values()) or any(
        not check[which]["pass"] for check in independent_checks for which in MODES
    ):
        return row

    samples = {which: [] for which in MODES}
    for _ in range(rounds):
        for which in ("uncached", "reuse", "reuse", "uncached"):
            item, out = evaluate(which)
            samples[which].append(item)
            del item, out
    keys = (
        "preparation_wall_ms", "tail_tile_merge_wall_ms",
        "full_group_wall_ms", "incremental_peak_mib", "peak_alloc_mib",
    )
    medians = {
        which: {
            key: statistics.median(item[key] for item in samples[which])
            for key in keys
        } for which in MODES
    }
    row["timings"] = {
        "median_by_path": medians,
        "speedup_uncached_over_reuse_steady":
            medians["uncached"]["full_group_wall_ms"] /
            medians["reuse"]["full_group_wall_ms"],
        "sample_records": samples,
    }
    return row


def single_case(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    capture_path = Path(args.capture)
    if not capture_path.is_file():
        raise ValueError("Trusted QKV capture not found")
    cap = torch.load(capture_path, map_location="cpu", mmap=True, weights_only=False)
    required = ("raw_head_index", "q_qhd", "k_khd", "v_khd")
    if not isinstance(cap, dict) or any(name not in cap for name in required):
        raise ValueError("Capture missing Q/K/V or raw_head_index")
    raw = cap["raw_head_index"]
    available = [int(x) for x in (raw.tolist() if isinstance(raw, torch.Tensor) else raw)]
    if not available or len(set(available)) != len(available):
        raise ValueError("Invalid captured native head IDs")
    count = args._case_heads
    if count > len(available) and not args.repeat_available_heads:
        raise ValueError("Use --repeat-available-heads for proxy counts")
    selected = [available[i % len(available)] for i in range(count)]
    proxy = count > len(available)
    cap = dict(cap)
    for key in ("q_qhd", "k_khd", "v_khd"):
        cap[key] = cap[key].to("cuda").contiguous()
    torch.cuda.synchronize()

    q_rows = int(cap["q_qhd"].shape[0])
    if q_rows < 1:
        raise ValueError("Capture has no query rows")
    tiles = args.tiles
    window_rows = tiles * 64
    windows = math.ceil(q_rows / window_rows)
    keys = int(cap["k_khd"].shape[0]) // 16 * 16
    capture_mib = sum(
        cap[key].numel() * cap[key].element_size()
        for key in ("q_qhd", "k_khd", "v_khd")
    ) / full.stage.MIB
    free, _ = torch.cuda.mem_get_info()
    budget = min(free, int(args.budget_mib * full.stage.MIB))
    plan = full.stage.plan_tile_head_batches(
        count, free_bytes=budget, keys_per_head=keys,
        tile_count=tiles, exact_leaves=256 if args.mode == "fast" else 512,
        reserve_bytes=int(args.reserve_mib * full.stage.MIB),
        fixed_temp_bytes=128 * full.stage.MIB, safety_factor=1.5,
        max_heads_per_group=args.max_heads_per_group,
    )
    # The old planner has no resident KBF/VBF/K8/KS. Include it explicitly:
    cache_per_head = keys * (128 * 2 + 128 * 2 + 128 + 4)
    allowed = budget - int(args.reserve_mib * full.stage.MIB) - 128 * full.stage.MIB
    with_cache = allowed // (
        plan.estimated_bytes_per_head + math.ceil(cache_per_head * 1.25)
    )
    if with_cache < 1:
        raise MemoryError("Not enough allowed GPU budget for even one cached head")
    group_size = min(plan.heads_per_group, int(with_cache))
    report = {
        "experiment": "h3_flat_gpu_qkv_preparation_reuse_full_query_ab_v1",
        "input_residency": "gpu",
        "gpu": torch.cuda.get_device_name(),
        "capture": str(capture_path),
        "head_count": count,
        "input_kind": "repeated_capture_proxy" if proxy else "distinct_native",
        "native_head_ids": available,
        "raw_head_indices": selected,
        "query_valid_rows_per_head": q_rows,
        "query_window_rows": window_rows,
        "query_windows_per_head": windows,
        "query_last_window_pad_rows": windows * window_rows - q_rows,
        "capture_gpu_resident_mib": capture_mib,
        "free_gpu_mib_before": free / full.stage.MIB,
        "effective_gpu_budget_mib": budget / full.stage.MIB,
        "planner": vars(plan),
        "max_group_size_including_persistent_cache": group_size,
        "mode": args.mode,
        "tiles": tiles,
        "source_commits": {
            "pinned_best": full.base.BEST,
            "candidate": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "quantization_exact_gates": [],
        "groups": [],
        "oom_group_retries": [],
        "errors": [],
        "completed": False,
        "summary": None,
        "notes": [
            "Uncached and reuse use the SAME fused Triton K8 + batched Tile/Merge.",
            "Only GPU K/V gather, BF16 staging, K8 quant and KS scales are reused.",
            "MU-dependent leaf stats, rowmax/topk and Tile/Merge still recalculate per window.",
            "K/V cache is per single attention call and per head group; never across denoising steps.",
            "Both paths consume same captured GPU QKV. Original BEST is numerical oracle, not timed.",
            "Cache build time is included in amortized comparison, not per-window steady timing.",
            "Uncached ABBA runs with the unused persistent cache still allocated, for symmetric interleaving.",
            "Full-Q figures are sums of per-window group medians, not a whole-model step.",
            "56 logical heads repeat four native captures; no model or video inference.",
        ],
    }
    pos = 0
    try:
        with tempfile.TemporaryDirectory(prefix="htrc_kv_reuse_pinned_") as tmp:
            pinned, runner = full.base.base_helper.original_modules(
                full.base.BEST, Path(tmp)
            )
            candidate = full.base.base_helper.helpers.sw.load(
                "_h3_kv_reuse_candidate", "h3/htrc_flat.py"
            )
            tr_best, tr_new = pinned._load_triton(), candidate._load_triton()
            for raw_head in dict.fromkeys(selected):
                gate = full.quant_key_exactness_gate(cap, raw_head, tr_new)
                report["quantization_exact_gates"].append(gate)
                if not gate["pass"]:
                    raise RuntimeError("K8 byte/scale mismatch on raw head " + str(raw_head))

            while pos < count:
                stop = min(count, pos + group_size)
                ids = selected[pos:stop]
                cache = None
                pending = []
                try:
                    torch.cuda.synchronize()
                    live_before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    cache = full.stage.prepare_gpu_head_kv_cache(cap, ids, tr_new)
                    torch.cuda.synchronize()
                    cache_build_wall_ms = (time.perf_counter() - start) * 1000.0
                    cache_mib = full.stage.gpu_head_kv_cache_bytes(cache) / full.stage.MIB
                    cache_build_peak_mib = (
                        torch.cuda.max_memory_allocated() - live_before
                    ) / full.stage.MIB
                    checkpoints = {0, windows // 2, windows - 1}
                    for wi in range(windows):
                        qstart = wi * window_rows
                        qstop = min(q_rows, qstart + window_rows)
                        valid = qstop - qstart
                        chunk = cap["q_qhd"][qstart:qstop]
                        if valid < window_rows:
                            padding = torch.zeros(
                                (window_rows - valid, *chunk.shape[1:]),
                                device=chunk.device, dtype=chunk.dtype
                            )
                            chunk = torch.cat((chunk, padding), dim=0)
                        window_cap = dict(cap)
                        window_cap["q_qhd"] = chunk
                        row = paired_window(
                            window_cap, ids, cache, tiles, args.mode,
                            pinned, runner, tr_best, tr_new,
                            args.rounds, args.warmups, args.tolerance,
                            independent=wi in checkpoints,
                        )
                        row.update({
                            "window_index": wi,
                            "query_valid_range": [qstart, qstop],
                            "query_padded_rows": window_rows - valid,
                        })
                        if row["timings"] is None:
                            raise RuntimeError(
                                f"Uncached/cached/pinned parity failed at window {wi}"
                            )
                        pending.append(row)
                        print(
                            "HEADS", pos, stop, "WINDOW", wi + 1, "/", windows,
                            "uncached/cache",
                            f'{row["timings"]["speedup_uncached_over_reuse_steady"]:.4f}x',
                            flush=True
                        )
                    cold = sum(
                        x["timings"]["median_by_path"]["uncached"]["full_group_wall_ms"]
                        for x in pending
                    )
                    hot = sum(
                        x["timings"]["median_by_path"]["reuse"]["full_group_wall_ms"]
                        for x in pending
                    )
                    group = {
                        "head_job_range": [pos, stop],
                        "raw_heads": ids,
                        "kv_cache_build_wall_ms": cache_build_wall_ms,
                        "kv_cache_live_mib": cache_mib,
                        "kv_cache_build_incremental_peak_mib": cache_build_peak_mib,
                        "full_query_window_results": pending,
                        "sum_of_window_median_wall_ms": {
                            "uncached": cold,
                            "reuse_steady": hot,
                            "reuse_amortized_including_build": hot + cache_build_wall_ms,
                            "steady_speedup": cold / hot,
                            "amortized_speedup": cold / (hot + cache_build_wall_ms),
                        },
                        "max_peak_alloc_mib_including_resident_cache": {
                            which: max(
                                x["timings"]["median_by_path"][which]["peak_alloc_mib"]
                                for x in pending
                            ) for which in MODES
                        },
                    }
                    report["groups"].append(group)
                    pos = stop
                    del group, cache, pending
                    gc.collect()
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError as exc:
                    del cache, pending
                    gc.collect()
                    torch.cuda.empty_cache()
                    if group_size == 1:
                        raise
                    before = group_size
                    group_size = max(1, group_size // 2)
                    report["oom_group_retries"].append({
                        "job": pos, "from": before, "to": group_size,
                        "error": str(exc)[:600],
                    })
                except Exception:
                    del cache, pending
                    raise
    except Exception as exc:
        report["errors"].append({
            "job": pos, "exception": type(exc).__name__,
            "error": str(exc)[:1400],
            "traceback_tail": traceback.format_exc()[-3000:],
        })
    report["completed"] = (pos == count and not report["errors"])
    if report["completed"]:
        cold = sum(g["sum_of_window_median_wall_ms"]["uncached"]
                   for g in report["groups"])
        hot = sum(g["sum_of_window_median_wall_ms"]["reuse_steady"]
                  for g in report["groups"])
        cache_build = sum(g["kv_cache_build_wall_ms"] for g in report["groups"])
        report["summary"] = {
            "sum_window_group_medians_uncached_ms": cold,
            "sum_window_group_medians_reuse_steady_ms": hot,
            "sum_kv_cache_build_wall_ms": cache_build,
            "reuse_amortized_ms": hot + cache_build,
            "speedup_steady": cold / hot,
            "speedup_amortized_with_build": cold / (hot + cache_build),
            "max_kv_cache_live_mib": max(g["kv_cache_live_mib"]
                                         for g in report["groups"]),
            "max_peak_alloc_mib_with_cache": {
                which: max(g["max_peak_alloc_mib_including_resident_cache"][which]
                           for g in report["groups"]) for which in MODES
            },
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture",
        default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt")
    ap.add_argument("--head-cases", default="4,56")
    ap.add_argument("--repeat-available-heads", action="store_true")
    ap.add_argument("--mode", choices=("fast", "balanced"), default="fast")
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--budget-mib", type=float, default=1024)
    ap.add_argument("--reserve-mib", type=float, default=256)
    ap.add_argument("--max-heads-per-group", type=int, default=None)
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    ap.add_argument("--_case-heads", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if min(args.tiles, args.rounds, args.warmups) < 1:
        ap.error("tiles, rounds and warmups must be positive")
    if (not math.isfinite(args.budget_mib) or args.budget_mib <= 0
        or not math.isfinite(args.reserve_mib) or args.reserve_mib < 0):
        ap.error("invalid GPU budget/reserve")
    if args.max_heads_per_group is not None and args.max_heads_per_group < 1:
        ap.error("max-heads-per-group must be positive")
    if args._case_heads is not None:
        if args._case_heads < 1:
            ap.error("case heads must be positive")
        report = single_case(args)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"kv_reuse_case_{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("CASE_REPORT:", path, flush=True)
        if not report["completed"]:
            raise SystemExit("FAIL: full-Q KV reuse child case (see JSON)")
        return
    tokens = [x.strip() for x in args.head_cases.split(",")]
    if not tokens or any(not x.isdecimal() or int(x) < 1 for x in tokens):
        ap.error("--head-cases must be comma-separated positive integers")
    heads = [int(x) for x in tokens]
    if len(set(heads)) != len(heads):
        ap.error("--head-cases must contain unique integers")
    combined = {
        "experiment": "h3_flat_gpu_kv_preparation_reuse_combined_v1",
        "head_cases": heads, "input_residency": "gpu",
        "source_commits": {
            "pinned_best": full.base.BEST,
            "candidate": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "cases": [], "summary": [], "errors": [], "completed": False,
        "notes": [
            "ONE persistent JSON; intermediate case JSONs are temporary.",
            "Each head count runs in a separate process to isolate CUDA residency.",
            "Both A/B paths use fused K8; KV cache reused only across Q windows of one call.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="htrc_kv_reuse_cases_") as tmp:
        for count in heads:
            case_out = Path(tmp) / str(count)
            case_out.mkdir()
            cmd = [
                sys.executable, "-B", str(Path(__file__).resolve()),
                "--_case-heads", str(count),
                "--capture", args.capture, "--mode", args.mode,
                "--tiles", str(args.tiles), "--rounds", str(args.rounds),
                "--warmups", str(args.warmups),
                "--tolerance", str(args.tolerance),
                "--budget-mib", str(args.budget_mib),
                "--reserve-mib", str(args.reserve_mib),
                "--out", str(case_out),
            ]
            if args.repeat_available_heads:
                cmd.append("--repeat-available-heads")
            if args.max_heads_per_group is not None:
                cmd.extend(["--max-heads-per-group", str(args.max_heads_per_group)])
            print("HEAD_CASE:", count, flush=True)
            tail = []
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace"
            ) as process:
                if process.stdout is None:
                    raise RuntimeError("Child output stream unavailable")
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    if not line.startswith("CASE_REPORT:"):
                        print(line, flush=True)
                    tail.append(line)
                    if len(tail) > 25:
                        tail.pop(0)
                returncode = process.wait()
            print("HEAD_CASE_DONE:", count, "exit", returncode, flush=True)
            files = list(case_out.glob("kv_reuse_case_*.json"))
            if len(files) != 1:
                combined["errors"].append({
                    "head_count": count, "returncode": returncode,
                    "error": "Expected one temporary case report",
                    "child_output_tail": tail[-12:],
                })
                break
            case = json.loads(files[0].read_text(encoding="utf-8"))
            combined["cases"].append(case)
            if case.get("summary"):
                combined["summary"].append({
                    "head_count": count,
                    "input_kind": case["input_kind"],
                    **case["summary"],
                })
            if returncode or not case.get("completed"):
                combined["errors"].append({
                    "head_count": count, "returncode": returncode,
                    "error": "Case failed; inspect embedded case errors",
                    "child_output_tail": tail[-12:],
                })
                break
    combined["completed"] = (
        len(combined["cases"]) == len(heads)
        and not combined["errors"]
        and all(case.get("completed") for case in combined["cases"])
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"flat_gpu_kv_reuse_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("REPORT:", path, flush=True)
    if not combined["completed"]:
        raise SystemExit("FAIL: full-Q KV reuse A/B (see one JSON)")


if __name__ == "__main__":
    main()
