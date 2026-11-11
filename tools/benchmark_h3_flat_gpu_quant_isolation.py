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



def run_head_cases(args, parser):
    """Run multiple head counts and publish exactly ONE persistent JSON report.

    Each existing single-case benchmark executes in its own child process;
    its per-case JSON is temporary and is embedded in the final combined JSON.
    This preserves previously validated single-case semantics, avoids VRAM
    cross-contamination and leaves no extra reports in the user's output dir.
    """
    parts = [part.strip() for part in args.head_cases.split(",")]
    if not parts or any(not x or not x.isdecimal() for x in parts):
        parser.error("--head-cases must be comma-separated positive head counts, e.g. 4,56")
    counts = [int(x) for x in parts]
    if any(x < 1 for x in counts) or len(set(counts)) != len(counts):
        parser.error("--head-cases must contain unique positive integers")
    if args.max_heads_per_group is not None and args.max_heads_per_group < 1:
        parser.error("--max-heads-per-group must be positive")

    report = {
        "experiment": "h3_flat_gpu_resident_torch_vs_triton_K8_combined_v1",
        "head_cases": counts,
        "query_coverage": args.query_coverage,
        "input_residency": "gpu",
        "capture": args.capture,
        "source_commits": {
            "pinned_best": full.base.BEST,
            "candidate": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "cases": [],
        "summary": [],
        "errors": [],
        "completed": False,
        "notes": [
            "Exactly one persistent combined JSON is written; child case reports are removed.",
            "Each count is a separate GPU process with its own memory planner and timing.",
            "Comparison is torch_gpu versus triton_gpu in identical batched paths.",
            "Repeated-head cases are labeled proxy and are not distinct native heads.",
            "Group-median sums are not model end-to-end latency.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix="htrc_quant_cases_") as tmp:
        for heads in counts:
            case_out = Path(tmp) / str(heads)
            case_out.mkdir()
            cmd = [
                sys.executable, "-B", str(Path(__file__).resolve()),
                "--capture", args.capture,
                "--heads", str(heads),
                "--mode", args.mode,
                "--tiles", str(args.tiles),
                "--query-coverage", args.query_coverage,
                "--budget-mib", str(args.budget_mib),
                "--reserve-mib", str(args.reserve_mib),
                "--rounds", str(args.rounds),
                "--warmups", str(args.warmups),
                "--tolerance", str(args.tolerance),
                "--out", str(case_out),
            ]
            if args.repeat_available_heads:
                cmd.append("--repeat-available-heads")
            if args.max_heads_per_group is not None:
                cmd.extend(["--max-heads-per-group", str(args.max_heads_per_group)])
            print(f"HEAD_CASE: {heads}", flush=True)
            # Do NOT buffer child output until its full-Q work finishes.
            # Forward each WINDOW/GROUP event immediately to PowerShell.
            # Only the short trailing output is retained for failure diagnosis.
            tail_output = []
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace"
            ) as process:
                if process.stdout is None:
                    raise RuntimeError("Child benchmark stdout pipe unavailable")
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    if not line.startswith("REPORT:"):
                        print(line, flush=True)
                    tail_output.append(line)
                    if len(tail_output) > 32:
                        tail_output.pop(0)
                returncode = process.wait()
            print(f"HEAD_CASE_DONE: {heads}, exit={returncode}", flush=True)
            matches = list(case_out.glob("flat_gpu_quant_isolation_ab_*.json"))
            if len(matches) != 1:
                report["errors"].append({
                    "head_count": heads, "returncode": returncode,
                    "error": "Expected one temporary case JSON, found " + str(len(matches)),
                    "child_output_tail": "\n".join(tail_output[-12:]),
                })
                break
            case = json.loads(matches[0].read_text(encoding="utf-8"))
            report["cases"].append(case)
            totals = case.get("sum_of_group_median_wall_not_true_end_to_end")
            if totals:
                report["summary"].append({
                    "head_count": heads,
                    "input_kind": case["input_kind"],
                    **totals,
                    "max_incremental_peak_mib": {
                        quant: max(
                            g["timings"]["median_by_path"][quant]["incremental_peak_mib"]
                            for g in case["groups"] if g["timings"] is not None
                        ) for quant in MODES
                    },
                    "parity_pass": all(
                        all(v["pass"] for v in g["parity"].values())
                        for g in case["groups"]
                    ),
                })
            if returncode != 0 or not case.get("completed"):
                report["errors"].append({
                    "head_count": heads, "returncode": returncode,
                    "error": "Case failed; see embedded case errors and console output",
                })
                break
    report["completed"] = (len(report["cases"]) == len(counts)
                           and not report["errors"]
                           and all(case.get("completed") for case in report["cases"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"flat_gpu_quant_isolation_combined_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT:", path, flush=True)
    if not report["completed"]:
        raise SystemExit("FAIL: combined GPU K8 A/B (see single JSON report)")


def run_full_query_windows(args, capture, selected, plan, pinned, runner, tr_best, tr_new, report):
    """Stream all Q rows through paired batched quant A/B, padding only the final Q window.

    The full QKV source remains GPU-resident across windows. Each final padded
    Q output is included in reference parity, but NOT in valid-query accounting.
    """
    query = capture["q_qhd"]
    n_query = int(query.shape[0])
    if n_query < 1:
        raise ValueError("Q capture has no query rows")
    window_rows = args.tiles * 64
    count = math.ceil(n_query / window_rows)
    checks = {0, count // 2, count - 1}
    report["query_total_valid_rows_per_head"] = n_query
    report["query_window_rows"] = window_rows
    report["query_window_count"] = count
    report["query_windows_completed"] = 0
    report["query_total_padded_rows_per_head"] = count * window_rows - n_query
    for wi in range(count):
        start = wi * window_rows
        end = min(n_query, start + window_rows)
        valid = end - start
        q = query[start:end]
        if valid < window_rows:
            # Padding is only local to the final query window, never written to K/V.
            pad = torch.zeros((window_rows - valid, *query.shape[1:]),
                              device=query.device, dtype=query.dtype)
            q = torch.cat((q, pad), dim=0)
        window_capture = dict(capture)
        window_capture["q_qhd"] = q
        pos, group_size = 0, plan.heads_per_group
        while pos < len(selected):
            group_end = min(len(selected), pos + group_size)
            try:
                row = measure_group(
                    window_capture, selected[pos:group_end], args.tiles, args.mode,
                    pinned, runner, tr_best, tr_new,
                    args.warmups, args.rounds, args.tolerance,
                    first_group=(pos == 0 and wi in checks),
                )
                row.update({
                    "head_job_range": [pos, group_end],
                    "raw_heads": selected[pos:group_end],
                    "query_window_index": wi,
                    "query_row_range_valid": [start, end],
                    "query_pad_rows": window_rows - valid,
                })
                report["groups"].append(row)
                if row["timings"] is None:
                    raise RuntimeError("Full-Q query-window output parity / oracle FAIL")
                print("WINDOW", wi + 1, "/", count, "HEADS", pos, group_end,
                      "torch/triton", row["timings"]["speedup_torch_over_triton_full_wall"],
                      flush=True)
                pos = group_end
            except torch.cuda.OutOfMemoryError as exc:
                if group_size == 1:
                    report["errors"].append({
                        "query_window_index": wi, "job": pos,
                        "exception": type(exc).__name__, "error": str(exc)[:1200],
                    })
                    return
                before = group_size
                group_size = max(1, group_size // 2)
                report["oom_group_retries"].append({
                    "query_window_index": wi, "job": pos,
                    "from": before, "to": group_size, "message": str(exc)[:600],
                })
            except Exception as exc:
                report["errors"].append({
                    "query_window_index": wi, "job": pos,
                    "exception": type(exc).__name__, "error": str(exc)[:1200],
                })
                return
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        report["query_windows_completed"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt")
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-cases", type=str, default=None,
                    help="Run comma-separated head counts in ONE JSON (e.g. 4,56)")
    ap.add_argument("--repeat-available-heads", action="store_true")
    ap.add_argument("--mode", choices=("fast", "balanced"), default="fast")
    ap.add_argument("--query-coverage", choices=("prefix", "full"), default="prefix",
                    help="full streams all query rows in fixed-size windows, with final Q padding")
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--budget-mib", type=float, default=1024)
    ap.add_argument("--reserve-mib", type=float, default=256)
    ap.add_argument("--max-heads-per-group", type=int, default=None)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    args = ap.parse_args()
    if args.head_cases is not None:
        return run_head_cases(args, ap)
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
        "input_residency": "gpu", "query_coverage": args.query_coverage,
        "input_kind": "repeated_capture_proxy" if proxy else "distinct_native",
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
            "No model checkpoint, graph, or video runtime; full Q is only covered with --query-coverage full.",
            "Full-Q streaming redoes K quant/leaf stats per Q window; NOT a persistent KV-cache benchmark.",
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
            if args.query_coverage == "full":
                run_full_query_windows(
                    args, cap, selected, plan, pinned, runner, tr_best, tr_new, report
                )
                pos = (len(selected)
                       if report["query_windows_completed"] == report["query_window_count"]
                       and not report["errors"] else 0)
            else:
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
