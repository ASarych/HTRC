"""Read-only Nsight resource/time triage of the existing HTRC Materialized Graph.

Never run H3, create a CUDA kernel, make an ABBA timing claim, or produce more
than one JSON. An installed NCU metric preflight precedes all GPU work.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHILD = Path(__file__).with_name("profile_h3_flat_materialized_ncu.py")
NVTX_RANGE = "HTRC_MATERIALIZED_REPLAY/"
KERNEL_FILTER = "regex:_[srt][iot][a-z]*_kernel"
WINDOWS_BATCH_META = "|&<>^()%!"
REQUIRED = {
    "duration": "gpu__time_duration.sum",
    "active_warps_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
}
OPTIONAL = {
    "dram_saturation_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_saturation_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "local_load_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "local_store_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
}
LAUNCH_ATTRIBUTES = (
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__grid_size",
    "launch__block_size",
)
METRIC_NAME = re.compile(
    r"(?<![A-Za-z0-9_.])([A-Za-z][A-Za-z0-9_]*__"
    r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)(?![A-Za-z0-9_.])"
)
NUMERIC = re.compile(r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def decode_output(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("oem" if os.name == "nt" else "cp866", errors="replace")


def select_metrics(query_text: str) -> tuple[tuple[str, ...], list[str]]:
    available = set(METRIC_NAME.findall(query_text))
    missing = [k for k, v in REQUIRED.items() if v not in available]
    if missing:
        raise ValueError("Required NCU metrics missing: " + ", ".join(missing))
    selected = list(REQUIRED.values())
    missing_opt = []
    for label, metric in OPTIONAL.items():
        if metric in available:
            selected.append(metric)
        else:
            missing_opt.append(label)
    return tuple(selected), missing_opt


def ncu_command(ncu: str, py: str, child: str, metrics: tuple[str, ...],
                count: int = 12) -> list[str]:
    if not 3 <= count <= 16:
        raise ValueError("Require 3..16 bounded Graph kernel launches")
    if any(ch in KERNEL_FILTER for ch in WINDOWS_BATCH_META):
        raise ValueError("Unsafe cmd.exe meta-character in filter")
    if not set(REQUIRED.values()).issubset(metrics):
        raise ValueError("Time and active-warp metrics are mandatory")
    if any(not re.fullmatch(r"[A-Za-z0-9_.]+", metric) for metric in metrics):
        raise ValueError("Unsafe NCU metric argument")
    return [ncu, "--nvtx", "--nvtx-include", NVTX_RANGE,
            "--graph-profiling", "node", "--kernel-name-base", "function",
            "--kernel-name", KERNEL_FILTER, "--launch-count", str(count),
            "--metrics", ",".join(metrics), "--csv", "--page", "raw",
            "--print-units", "base", py, "-B", child, "--child"]


def numeric(text: str, *, integer: bool = False) -> float | None:
    v = str(text).strip()
    if v.lower() in ("", "n/a", "nan", "inf", "-inf"):
        return None
    # On the observed Russian Windows, NBSP decoded as cp866 Cyrillic 'a'.
    v = re.sub(r"(?<=\d)[\u00a0\u202f\u2009\u0430 ](?=\d)", "", v)
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", v):
        v = v.replace(",", "")
    elif "," in v and "." not in v:
        v = v.replace(",", ".")
    elif "," in v:
        return None
    if not NUMERIC.fullmatch(v):
        return None
    n = float(v)
    if not math.isfinite(n) or n < 0 or (integer and not n.is_integer()):
        return None
    return n


def classify_kernel(name: str) -> str | None:
    # Exact family only: no accidental _merge_tail_kernel match.
    for label in ("stats", "rowmax", "tile"):
        if "_" + label + "_kernel" in name:
            return label
    return None


def parse_resource_csv(output: str, metrics: tuple[str, ...]) -> dict:
    header = None
    units = None
    samples = []
    rejected = 0
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        row = [part.lstrip("\ufeff").strip() for part in row]
        if "Kernel Name" in row and "ID" in row:
            header = row
            units = None
            continue
        if header is None or len(row) != len(header):
            continue
        if units is None:
            units = dict(zip(header, row))
            continue
        record = dict(zip(header, row))
        stage = classify_kernel(record.get("Kernel Name", ""))
        if stage is None:
            continue
        numbers = {}
        for label, metric in {**REQUIRED, **OPTIONAL}.items():
            if metric in metrics:
                value = numeric(record.get(metric, ""), integer=label in (
                    "local_load_sectors", "local_store_sectors"))
                if value is not None:
                    numbers[label] = value
        if not all(label in numbers for label in REQUIRED):
            rejected += 1
            continue
        if not 0 <= numbers["active_warps_pct"] <= 100:
            rejected += 1
            continue
        launch = {}
        for attr in LAUNCH_ATTRIBUTES:
            if attr in record:
                value = numeric(record[attr], integer=True)
                if value is not None:
                    launch[attr] = int(value)
        samples.append({"stage": stage, "id": record.get("ID"),
                        "metrics": numbers, "launch": launch})
    best = header or []
    unit_str = (units or {}).get(REQUIRED["duration"], "")
    return {
        "samples": samples,
        "diagnostics": {
            "csv_header_found": bool(header), "csv_column_count": len(best),
            "missing_metric_columns": [m for m in metrics if m not in best],
            "duration_unit": unit_str,
            "rejected_required_metric_rows": rejected,
        },
    }


def summarize(parsed: dict) -> dict:
    samples = parsed["samples"]
    groups = {stage: [x for x in samples if x["stage"] == stage]
              for stage in ("stats", "rowmax", "tile")}
    duration_unit = parsed["diagnostics"]["duration_unit"].lower()
    scales_us = {"nsecond": .001, "ns": .001,
                 "usecond": 1., "us": 1., "msecond": 1000.,
                 "ms": 1000., "second": 1e6, "s": 1e6}
    scale = scales_us.get(duration_unit)
    out = {}
    for stage, data in groups.items():
        if not data:
            continue
        times = [x["metrics"]["duration"] for x in data]
        occupancies = [x["metrics"]["active_warps_pct"] for x in data]
        launch_cols = {}
        for key in LAUNCH_ATTRIBUTES:
            values = [x["launch"][key] for x in data if key in x["launch"]]
            if values:
                launch_cols[key] = sorted(set(values))
        optional_summary = {}
        for key in OPTIONAL:
            found = [x["metrics"][key] for x in data if key in x["metrics"]]
            if len(found) == len(data):
                optional_summary[key] = {
                    "median": statistics.median(found), "min": min(found),
                    "max": max(found),
                }
        out[stage] = {
            "samples": len(data),
            "duration_raw_median": statistics.median(times),
            "duration_raw_unit": parsed["diagnostics"]["duration_unit"],
            "duration_us_median": (statistics.median(times)*scale
                                   if scale is not None else None),
            "occupancy_pct_median": statistics.median(occupancies),
            "occupancy_pct_range": [min(occupancies), max(occupancies)],
            "launch_attributes_distinct_values": launch_cols,
            "optional_metrics": optional_summary,
        }
    return {"stages": out, "timing_unit_supported": scale is not None,
            "has_stats_and_tile": bool(groups["stats"] and groups["tile"]),
            "parsed_launches": len(samples)}


def verify_exact_head(expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("Expected exact Git commit SHA")
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=15, text=True)
    if p.returncode or p.stdout.strip() != expected:
        raise RuntimeError("SOURCE_HEAD_MISMATCH: refuse profiler")


def run_parent(ncu: str, count: int, expected_head: str) -> dict:
    report = {
        "schema": "htrc-materialized-resource-ncu-v1",
        "status": "FAIL_CLOSED", "completed": False,
        "hardware_profile": False, "source_head": expected_head,
        "tokens": 15275, "heads": 4, "mode": "balanced",
        "exact_leaves": 512, "window_tiles": 32, "stats_bt": 16,
        "candidate": "materialized_ONLY", "graph_node_count_cap": count,
        "h3_checkpoint": "UNRUN", "video": "UNRUN",
        "production_promotion": "PROHIBITED_RESOURCE_TRIAGE_ONLY",
        "ncu_timing_comparable_with_unprofiled_abba": False,
        "limitations": [
            "NCU node-replay durations are not normal full attention timing",
            "Local-memory sectors are not proof of compiler register spilling",
            "Throughput percentages are resource pressure, not proportional speedups",
            "Only a bounded subset of Stats/Rowmax/Tile Graph nodes is sampled",
        ],
    }
    try:
        verify_exact_head(expected_head)
        query = subprocess.run([ncu, "--query-metrics-mode", "all"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=90)
        if query.returncode:
            raise RuntimeError("NCU_QUERY_FAILED:" +
                               decode_output(query.stdout or b"")[-300:])
        metrics, unavailable = select_metrics(decode_output(query.stdout or b""))
        report["metric_preflight"] = "PASS_ADVERTISED_MANDATORY_METRICS"
        report["requested_metrics"] = list(metrics)
        report["optional_unavailable"] = unavailable
        p = subprocess.run(ncu_command(ncu, sys.executable, str(CHILD),
                                       metrics, count), cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=600)
        stdout = decode_output(p.stdout or b"")
        report["ncu_exit_code"] = p.returncode
        report["strict_child_parity_marker"] = (
            "HTRC_NCU_CHILD_STRICT_PARITY_PASS" in stdout)
        report["diagnostic_output_tail"] = stdout[-2500:]
        if "ERR_NVGPUCTRPERM" in stdout:
            report["status"] = "BLOCKED_GPU_COUNTER_PERMISSION"
            return report
        if p.returncode:
            report["status"] = "FAIL_CLOSED_NCU_NONZERO_EXIT"
            return report
        parsed = parse_resource_csv(stdout, metrics)
        report["csv_diagnostics"] = parsed["diagnostics"]
        report.update(summarize(parsed))
        report["kernel_samples"] = parsed["samples"]
        ok = (report["strict_child_parity_marker"] and
              report["has_stats_and_tile"] and
              report["parsed_launches"] > 0 and
              not parsed["diagnostics"]["missing_metric_columns"] and
              not parsed["diagnostics"]["rejected_required_metric_rows"] and
              report["timing_unit_supported"])
        if ok:
            report["status"] = "PROFILE_RESOURCE_COUNTERS_COLLECTED"
            report["completed"] = True
            report["hardware_profile"] = True
        else:
            report["status"] = "FAIL_CLOSED_RESOURCE_CSV_OR_PARITY"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"[:800]
        report["error"] = msg
        if "Required NCU metrics missing" in msg:
            report["status"] = "BLOCKED_UNSUPPORTED_REQUIRED_METRICS"
        elif "SOURCE_HEAD_MISMATCH" in msg:
            report["status"] = "FAIL_CLOSED_SOURCE_HEAD_MISMATCH"
        elif "NCU_QUERY_FAILED" in msg:
            report["status"] = "FAIL_CLOSED_METRIC_QUERY"
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ncu", required=True)
    ap.add_argument("--expected-head", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--count", type=int, default=12)
    args = ap.parse_args()
    report = run_parent(args.ncu, args.count, args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True)+"\n",
                           encoding="utf-8")
    print(f"one_json={args.output} status={report['status']}")
    return 0 if report["status"] == "PROFILE_RESOURCE_COUNTERS_COLLECTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
