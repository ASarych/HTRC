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
sys.path.insert(0, str(ROOT / "tools"))
from h3_flat_scale_protocol import GPU_VARIANTS, plan
from estimate_h3_flat_long_context import estimate_case

# One shared quality runner compares all modes to the same dense reference.
# Older per-variant bit-exact A/B scripts are retained for diagnostics but
# MUST NOT determine whether a research candidate runs.
from h3_flat_quality_ab import run_case as run_quality_case

DEFAULT_OUTPUT = Path(
    r"D:\ComfyOutput\H3\math_probe\flat_short_long_matrix.json"
)


def _save_once(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def execute(report: dict, *, output_path: Path) -> dict:
    """Continue on NUMERICAL differences: percent error is the signal.

    Numerical disagreement with eager or materialized never cancels the
    next variant/length. CUDA compile/OOM/runtime failures are reported,
    with best-effort GPU cleanup and later cases still attempted.
    """
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use plan-only mode")
    cfg=report["plan"]
    cases=[c for c in cfg["cases"] if c["gpu_eligible"]]
    report["gpu"]=torch.cuda.get_device_name(0)
    report["gpu_execution"]="STARTED"
    report["status"]="RUNNING"
    report["results"]=[
        {"case":c["id"],"band":c["band"],"tokens":c["tokens"],
         "status":"UNRUN","variants":{}}
        for c in cases
    ]
    _save_once(output_path,report)
    for row in report["results"]:
        row["status"]="RUNNING"
        _save_once(output_path,report)
        def on_update(name, outcome):
            row["variants"][name]=outcome
            _save_once(output_path,report)
        try:
            measured=run_quality_case(
                tokens=row["tokens"],heads=cfg["heads"],
                mode=cfg["mode"],tiles=cfg["window_tiles"],
                rounds=cfg["rounds_per_case"],
                budget_mib=cfg["budget_mib"],
                reserve_mib=cfg["reserve_mib"],
                variants=tuple(cfg["variants"]),
                sample_queries=cfg["sample_queries"],
                seed=cfg["seed"],on_update=on_update,
            )
            row.update({
                "status":measured["status"],
                "variants":measured["results"],
                "quality_reference":measured["reference"],
                "quality_ranking":measured["quality_ranking"],
                "pairwise_diagnostics":measured["pairwise"],
                "sampled_query_positions":measured["sampled_query_positions"],
                "numerical_disagreement_aborts":False,
            })
        except BaseException as exc:
            row.update({
                "status":"RUNTIME_FAILED",
                "runtime_error":f"{type(exc).__name__}: {exc}"[:1800],
            })
            # A GPU process may need a restart after a fatal CUDA error,
            # but keep later rows visible rather than inventing a PASS.
        finally:
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except BaseException as clean:
                row["gpu_cleanup_error"]=str(clean)[:600]
            _save_once(output_path,report)
    report["gpu_execution"]="COMPLETED_WITH_RECORDED_FAILURES" if any(
        row["status"] in ("PARTIAL_RUNTIME_FAILURE","RUNTIME_FAILED")
        for row in report["results"]) else "COMPLETED"
    invalid_numeric=any(
        item.get("status")=="NUMERIC_NONFINITE_RECORDED"
        for row in report["results"] for item in row.get("variants",{}).values()
    )
    report["status"]=(
        "MEASURED_WITH_RUNTIME_FAILURES"
        if report["gpu_execution"]!="COMPLETED"
        else ("INVALID_NUMERIC_RECORDED" if invalid_numeric
              else "QUALITY_AND_TIMING_MEASURED")
    )
    report["speed_claim"]="NO_SOL_OR_VIDEO_SPEEDUP_PROVEN"
    report["numeric_disagreement_policy"]="RECORD_PERCENT_AND_CONTINUE"
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
    ap.add_argument("--sample-queries",type=int,default=64,
                    help="Dense reference evaluates all keys at this many queries")
    ap.add_argument("--seed",type=int,default=42026)
    ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    variants = tuple(v.strip() for v in args.variants.split(","))
    matrix = plan(
        variants,args.scope,heads=args.heads,tiles=args.tiles,
        mode=args.mode,rounds=args.rounds,
        budget_mib=args.budget_mib,reserve_mib=args.reserve_mib,
    )
    if args.sample_queries<1:
        ap.error("--sample-queries must be positive")
    matrix["sample_queries"]=args.sample_queries
    matrix["seed"]=args.seed
    report = {
        "experiment": "h3_flat_exact_dense_quality_vs_length_matrix_v2",
        "quality_reference":"FP32_FULL_KEYS_SAMPLED_QUERIES",
        "numeric_disagreement_policy":"RECORD_PERCENT_AND_CONTINUE",
        "status": "PLANNED_ONLY",
        "completed_gpu": False,
        "timestamp_kind": "tool_wall_time_not_H3_runtime",
        "plan": matrix,
        "results": [
            {"case": c["id"], "tokens": c["tokens"], "band": c["band"],
             "status":"UNRUN",
             "variants": {"materialized":{"status":"UNRUN"},
                          **{v:{"status":"UNRUN"} for v in variants}}}
            for c in matrix["cases"] if c["gpu_eligible"]
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
            "Quality = relative L2 percent to exact FP32 full-key attention "
            "at 64 or configured sampled query rows; NOT full-query dense output.",
            "Bitwise mismatch between Flat modes never cancels the experiment. "
            "Eager Flat agreement is diagnostic only, not the quality reference.",
            "Timings use non-interleaved warmed samples; no Sol speed claim.",
            "223605-token Full HD is analytical/preflight only, never a GPU case.",
            "Numerical discrepancies are recorded as percentages and do not "
            "cancel any next mode/length. Runtime errors are recorded separately.",
        ],
    }
    # The Full-HD row is a source-based original Flat estimate only.
    # Never claim admission or GPU feasibility for a new fused backend.
    for cap in report["capacity_cases"]:
        cap["source_model_scope"]="MATERIALIZED_FLAT_FORMULA_ONLY"
        cap["source_model"]=estimate_case(
            tokens=cap["tokens"],heads=56,mode=args.mode,
            budget_mib=args.budget_mib,reserve_mib=args.reserve_mib
        )
    started=time.perf_counter()
    try:
        if args.execute:
            execute(report,output_path=args.output)
            report["attempted_gpu"]=True
            report["completed_gpu"] = report["gpu_execution"]=="COMPLETED"
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
