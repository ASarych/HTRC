"""One JSON: mandatory short H3 Flat regression + optional length scaling.

Plans without CUDA by default. --execute calls existing isolated
two-bank full-output GPU A/B runners, not full ComfyUI or video. All
long scopes start with BOTH short controls; no long-only shortcut.

Examples:
  python -B tools/benchmark_h3_flat_scale_matrix.py --scope scaling
  python -B tools/benchmark_h3_flat_scale_matrix.py --execute --scope short
  python -B tools/benchmark_h3_flat_scale_matrix.py --execute --scope scaling --variants split_stream
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.h3_flat_scale_protocol import GPU_VARIANTS, plan

RUNNERS = {
    "split_stream": ROOT / "tools" / "benchmark_h3_flat_streamed_vb_ab.py",
    "fused_vb": ROOT / "tools" / "benchmark_h3_flat_fused_vb_ab.py",
}
DEFAULT_OUTPUT = Path(
    r"D:\ComfyOutput\H3\math_probe\flat_short_long_matrix.json"
)


def _load_runner(variant: str):
    if variant not in RUNNERS:
        raise ValueError(f"No full-output GPU runner for variant {variant}")
    path = RUNNERS[variant]
    spec = importlib.util.spec_from_file_location(
        "htrc_matrix_runner_" + variant, path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GPU runner {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run_ab", None)):
        raise RuntimeError(f"GPU runner missing run_ab: {path}")
    return module.run_ab


def _save_once(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def execute(report: dict, *, output_path: Path) -> dict:
    """Run short controls before any medium/long GPU workload.

    Aborts on a failed numerical gate. Does not mark UNRUN results PASS.
    GPU timing is preliminary because modes run in fixed order; no Sol
    or actual-video conclusions can be drawn from this matrix.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use plan-only mode")
    config = report["plan"]
    variants = config["variants"]
    gpu_cases = [c for c in config["cases"] if c["gpu_eligible"]]
    runners = {v: _load_runner(v) for v in variants}
    report["gpu"] = torch.cuda.get_device_name(0)
    report["gpu_execution"] = "STARTED"
    report["results"] = [
        {"case": case["id"], "tokens": case["tokens"],
         "band": case["band"], "variant": v, "status": "UNRUN"}
        for case in gpu_cases for v in variants
    ]
    report["status"] = "RUNNING"
    _save_once(output_path, report)
    for record in report["results"]:
        record["status"] = "RUNNING"
        _save_once(output_path, report)
        try:
            case = runners[record["variant"]](
                record["tokens"], config["heads"], config["mode"],
                config["window_tiles"], config["rounds_per_case"],
                config["budget_mib"], config["reserve_mib"], output_path
            )
            if case.get("status") != "PASS_NUMERIC" or not case.get("completed"):
                raise AssertionError("GPU A/B did not pass its numerical gate")
            warm = case["warm_graph_timings"]
            base = float(warm["materialized"]["sync_wall_ms_per_call"])
            candidate = float(
                warm[record["variant"]]["sync_wall_ms_per_call"]
            )
            if base <= 0 or candidate <= 0:
                raise AssertionError("Invalid GPU timings")
            record.update({
                "status": "PASS_NUMERIC",
                "full_output_parity": case["numeric"],
                "first_call_gates": case["eager_graph_first_call_gates"],
                "materialized_wall_ms_per_call": base,
                "candidate_wall_ms_per_call": candidate,
                "materialized_gpu_event_ms_per_call": warm[
                    "materialized"]["gpu_event_ms_per_call"],
                "candidate_gpu_event_ms_per_call": warm[
                    record["variant"]]["gpu_event_ms_per_call"],
                "speedup_vs_materialized": base / candidate,
                "speed_claim": "PRELIMINARY_A_B_NOT_INTERLEAVED",
                "scope": case["scope"],
            })
        except BaseException as exc:
            record.update({
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}"[:1800],
            })
            report["status"] = "FAIL_CLOSED"
            report["gpu_execution"] = "STOPPED_AT_FAILURE"
            _save_once(output_path, report)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        _save_once(output_path, report)
    report["gpu_execution"] = "COMPLETED"
    report["status"] = "PASS_NUMERIC"
    report["speed_claim"] = "NO_SOL_OR_VIDEO_SPEEDUP_PROVEN"
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true",
                    help="Run GPU A/B; otherwise write plan only")
    ap.add_argument("--scope", choices=("short","scaling","capacity"),
                    default="short")
    ap.add_argument("--variants",default="split_stream,fused_vb",
                    help="Comma list: split_stream,fused_vb")
    ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--mode",choices=("fast","balanced"),default="fast")
    ap.add_argument("--tiles",type=int,choices=(16,32,64),default=16)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--budget-mib",type=int,default=1024)
    ap.add_argument("--reserve-mib",type=int,default=256)
    ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    variants = tuple(v.strip() for v in args.variants.split(","))
    matrix = plan(
        variants,args.scope,heads=args.heads,tiles=args.tiles,
        mode=args.mode,rounds=args.rounds,
        budget_mib=args.budget_mib,reserve_mib=args.reserve_mib,
    )
    report = {
        "experiment": "h3_flat_universal_vs_long_context_matrix_v1",
        "status": "PLANNED_ONLY",
        "completed_gpu": False,
        "timestamp_kind": "tool_wall_time_not_H3_runtime",
        "plan": matrix,
        "results": [
            {"case": c["id"], "tokens": c["tokens"], "band": c["band"],
             "variant": v, "status": "UNRUN"}
            for c in matrix["cases"] if c["gpu_eligible"]
            for v in variants
        ],
        "capacity_cases": [
            {"case": c["id"], "tokens": c["tokens"],
             "status": "ANALYTICAL_ONLY_NOT_GPU_ADMISSION"}
            for c in matrix["cases"] if not c["gpu_eligible"]
        ],
        "gpu_execution": "UNRUN",
        "warnings": [
            "Every GPU long-context run includes 15275 and 18723 tokens first.",
            "Short data are essential for speedups claimed at any length and "
            "for regressions of long-only capacity candidates.",
            "These are synthetic native BF16 full-attention inputs, not "
            "actual H3 forward video and not a Sol comparison.",
            "Benchmarks each use the existing two-bank eager parity gate "
            "and a non-interleaved warmed sample. This is not a final speed ranking.",
            "223605-token Full HD is analytical/preflight only, never a GPU case.",
            "A numerical failure aborts all later length cases; no fallback.",
        ],
    }
    started=time.perf_counter()
    try:
        if args.execute:
            execute(report,output_path=args.output)
            report["completed_gpu"] = True
    except BaseException as exc:
        report["status"] = "FAIL_CLOSED"
        report["error"] = f"{type(exc).__name__}: {exc}"[:1800]
        raise
    finally:
        report["orchestrator_wall_seconds"] = round(time.perf_counter()-started,3)
        _save_once(args.output,report)
        print(f"one_report={args.output} status={report['status']}")


if __name__ == "__main__":
    main()
