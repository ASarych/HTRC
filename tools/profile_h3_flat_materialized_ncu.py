"""Fail-closed Nsight Compute sample of EXISTING materialized Tile32 CUDA Graph nodes.

One JSON only; profiler stdout is consumed in memory, not written as CSV or ncu-rep.
Neither GPU timing under ncu nor an L2 hit rate is a production speed claim.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NVTX_RANGE = "HTRC_MATERIALIZED_REPLAY/"
# ncu.BAT is parsed by cmd.exe on Windows even if subprocess shell=False.
# Do not pass metacharacters such as | or parentheses via this .BAT wrapper.
# The bracket class matches stats, rowmax and tile without shell alternation.
# Match only st*, ro*, ti* kernel families; this also excludes
# _merge_tail_kernel, which matched the previous suffix-only pattern.
KERNEL_FILTER = "regex:_[srt][iot][a-z]*_kernel"
WINDOWS_BATCH_META = "|&<>^()%!"
# For NVIDIA SM120, DRAM counter names may contain an op_read/op_write
# infix. Query this exact installed ncu version BEFORE executing any GPU work.
METRIC_ALTERNATIVES = {
    "dram_read_bytes": (
        "dram__bytes_op_read.sum", "dram__bytes_read.sum",
    ),
    "dram_write_bytes": (
        "dram__bytes_op_write.sum", "dram__bytes_write.sum",
    ),
    "l2_read_sectors": (
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "lts__t_sectors_op_read.sum",
    ),
}
# CPU parser fixture defaults, not an assumption of runtime availability.
METRICS = (
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "launch__registers_per_thread",
)


def select_available_metrics(query_output: str) -> tuple[str, ...]:
    """Select full, advertised metric names; no unsupported silent fallback."""
    # --query-metrics-mode all prints full names in the Metric Name column.
    names = set(re.findall(r"(?<![a-zA-Z0-9_.])"
                           r"([a-zA-Z][a-zA-Z0-9_]*__"
                           r"[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)+)"
                           r"(?![a-zA-Z0-9_.])", query_output))
    selected = []
    for family, options in METRIC_ALTERNATIVES.items():
        exact = next((name for name in options if name in names), None)
        if exact is None:
            raise ValueError(f"Unsupported NCU metric family: {family}")
        selected.append(exact)
    # Device attribute is optional; available in raw output on many GPUs.
    if re.search(r"(?<![a-zA-Z0-9_.])launch__registers_per_thread"
                 r"(?![a-zA-Z0-9_.])", query_output):
        selected.append("launch__registers_per_thread")
    return tuple(selected)


def query_metrics(ncu: str) -> tuple[tuple[str, ...], dict]:
    """CPU-only installed-NCU metric discovery, prior to CUDA launch."""
    proc = subprocess.run(
        [ncu, "--query-metrics-mode", "all"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90,
    )
    content = decode_profiler_output(proc.stdout or b"")
    if proc.returncode != 0:
        raise RuntimeError(
            "Nsight metric query failed (exit "
            + str(proc.returncode) + "): " + content[-350:]
        )
    chosen = select_available_metrics(content)
    return chosen, {
        "query_exit_code": proc.returncode,
        "query_preflight": "PASS_ADVERTISED_METRIC_NAMES",
        "verified_metric_names": list(chosen),
    }


def command(ncu: str, py: str, self_path: str, count: int = 12,
            metrics: tuple[str, ...] | None = None) -> list[str]:
    if not 1 <= count <= 16:
        raise ValueError("Profiled launches must be 1..16")
    if any(ch in KERNEL_FILTER for ch in WINDOWS_BATCH_META):
        raise ValueError("ncu.BAT filter contains unsafe Windows cmd metacharacters")
    selected = tuple(metrics or METRICS)
    if len(selected) < 3 or any(not re.fullmatch(
        r"[a-zA-Z0-9_.]+", metric) for metric in selected
    ):
        raise ValueError("Invalid selected metric names")
    return [
        ncu, "--nvtx", "--nvtx-include", NVTX_RANGE,
        "--graph-profiling", "node",
        "--kernel-name-base", "function", "--kernel-name", KERNEL_FILTER,
        "--launch-count", str(count), "--metrics", ",".join(selected),
        "--csv", "--page", "raw", "--print-units", "base",
        py, "-B", self_path, "--child",
    ]


def decode_profiler_output(raw: bytes) -> str:
    """Decode UTF-8 CLI output, falling back to Windows console OEM code page.

    cmd.exe error messages in Russian Windows are usually OEM cp866, not
    Python's ANSI locale encoding. Preserve diagnostics in our UTF-8 JSON.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("oem" if os.name == "nt" else "cp866", errors="replace")


def parse_counter(value: str) -> float | None:
    """Numeric NCU counters under en-US or ru-RU Windows formatting.

    In the observed NCU 2026.1.1 output, cp1251 NBSP (0xA0) was
    decoded using cp866 and appeared as Cyrillic 'а' between digits.
    Never invent a zero for absent, n/a, fraction, or malformed values.
    Requested DRAM bytes and L2 sector counts must be integers.
    """
    v = str(value).strip()
    if not v or v.lower() in ("n/a", "nan", "inf", "-inf"):
        return None
    v = re.sub(r"(?<=\d)[\u00a0\u202f\u2009\u0430 ](?=\d)", "", v)
    v = v.strip().replace("\u00a0", "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", v):
        v = v.replace(",", "")
    elif "," in v and "." not in v:
        v = v.replace(",", ".")
    elif "," in v:
        # Decimal comma and dot thousands separators; do not guess.
        return None
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", v):
        return None
    number = float(v)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return number


def inspect_csv(output: str, selected: tuple[str, ...]) -> dict:
    """Diagnostic-only CSV census, never claims unparsed values are valid."""
    headers = []
    by_kernel = {}
    malformed = 0
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        row = [x.lstrip("\ufeff").strip() for x in row]
        if "Kernel Name" in row and "ID" in row:
            headers.append(row)
        elif any("_stats_kernel" in x or "_tile_kernel" in x or
                 "_rowmax_kernel" in x for x in row):
            name = next((x for x in row if x in
                         ("_stats_kernel", "_rowmax_kernel",
                          "_tile_kernel", "_merge_tail_kernel")), None)
            if name:
                by_kernel[name] = by_kernel.get(name, 0) + 1
            else:
                malformed += 1
    best = max(headers, key=lambda h: sum(m in h for m in selected),
               default=[])
    return {
        "csv_headers_found": len(headers),
        "csv_total_columns": len(best),
        "csv_metric_columns": [m for m in best if m.startswith(
            ("dram__", "lts__", "launch__"))][:40],
        "csv_missing_requested_metric_columns": [
            m for m in selected[:3] if m not in best
        ],
        "csv_named_kernel_rows": by_kernel,
        "csv_other_kernel_like_rows": malformed,
        "csv_header_is_complete_for_required_metrics": (
            bool(best) and all(m in best for m in selected[:3])
        ),
    }


def parse_csv(output: str, metrics: tuple[str, ...] | None = None) -> list[dict]:
    """Nsight raw CSV: header row, units row, then launch rows."""
    selected = tuple(metrics or METRICS)
    header = None
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        row = [x.lstrip("\ufeff").strip() for x in row]
        if ("Kernel Name" in row and "ID" in row and
            all(m in row for m in selected[:3])):
            header = row
            continue
        if header is None or len(row) != len(header):
            continue
        item = dict(zip(header, row))
        name = item.get("Kernel Name", "")
        if not any(x in name for x in ("_stats_kernel", "_rowmax_kernel",
                                       "_tile_kernel")):
            continue
        metrics = {}
        for metric in selected:
            number = parse_counter(item.get(metric, ""))
            if number is not None:
                metrics[metric] = number
        if selected[0] in metrics and selected[1] in metrics and selected[2] in metrics:
            rows.append({
                "kernel": "stats" if "_stats_kernel" in name
                          else "rowmax" if "_rowmax_kernel" in name
                          else "tile",
                "metrics": metrics,
                "launch_index": item.get("ID"),
            })
    return rows


def summary(rows: list[dict]) -> dict:
    stages = {}
    for row in rows:
        name = row["kernel"]
        st = stages.setdefault(name, {"profiled_launches": 0, "metrics_sum": {}})
        st["profiled_launches"] += 1
        for key, value in row["metrics"].items():
            if key.startswith("launch__"):
                continue
            st["metrics_sum"][key] = st["metrics_sum"].get(key, 0) + value
    has_both = "stats" in stages and "tile" in stages
    return {
        "status": "PROFILE_COUNTERS_COLLECTED" if has_both else "PROFILE_PARTIAL",
        "stages": stages,
        "profiled_launches": len(rows),
        "has_stats_and_tile": has_both,
    }


def child() -> None:
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live

    # The short-lived child is exclusively a no-grad profiler process.
    # graph_attention_bthd rejects autograd-enabled entry.
    torch.set_grad_enabled(False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
        raise RuntimeError("Pinned SM120 profile only")
    if "RTX 5070 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Pinned RTX 5070 Ti only")
    torch.manual_seed(42026)
    shape = (1, 4, 15275, 128)
    banks = [tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3)) for _ in range(2)]
    with tempfile.TemporaryDirectory(prefix="htrc-ncu-gate-") as temp:
        os.environ["APL_HTRC_DIRECTFLOW_REPORT"] = str(Path(temp) / "gate.json")
        os.environ["APL_HTRC_DIRECTFLOW_GROUP"] = "11"
        os.environ["APL_HTRC_DIRECTFLOW_WARMUPS"] = "4"
        os.environ["APL_HTRC_DIRECTFLOW_VB_MODE"] = "materialized"
        flow.clear_session_cache()

        def call(bank):
            return flow.graph_attention_bthd(
                *bank, heads=4, mode="balanced", tiles=32,
                budget_mib=1024, reserve_mib=256, workspace=True,
            )

        first = call(banks[0])
        torch.cuda.synchronize()
        gate = json.loads((Path(temp) / "gate.json").read_text(encoding="utf-8"))
        if (gate.get("status") != "PASS_FIRST_REAL_CALL" or
            gate.get("all_groups_bit_exact") is not True):
            raise AssertionError("Materialized first-call eager/Graph gate FAIL")
        second = call(banks[1])
        torch.cuda.synchronize()
        eager = live.flat_attention_bthd(
            *banks[1], heads=4, mode="balanced", tiles=32,
            budget_mib=1024, reserve_mib=256,
            workspace=True, query_staging="native_torch_mu",
            kv_staging="native_direct", dispatch="eager",
        )
        if not torch.equal(second, eager) or torch.equal(first, second):
            raise AssertionError("Materialized bank-B eager/Graph gate FAIL")
        del eager
        for _ in range(5):
            warmed = call(banks[1])
        torch.cuda.synchronize()
        if not torch.equal(warmed, second):
            raise AssertionError("Post-capture materialized replay mismatch")
        torch.cuda.nvtx.range_push("HTRC_MATERIALIZED_REPLAY")
        try:
            check = call(banks[1])
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        if not torch.equal(check, second):
            raise AssertionError("Profiled replay output changed")
        flow.clear_session_cache()
    print("HTRC_NCU_CHILD_STRICT_PARITY_PASS", flush=True)


def run_parent(ncu: str, count: int) -> dict:
    result = {
        "schema": "htrc-materialized-sm120-ncu-node-v1",
        "status": "FAIL_CLOSED",
        "completed": False,
        "hardware_profile": False,
        "gpu": "NVIDIA GeForce RTX 5070 Ti (SM120, user inventory)",
        "mode": "balanced", "tokens": 15275, "heads": 4,
        "window_tiles": 32, "stats_bt": 16, "exact_leaves": 512,
        "candidate": "materialized_ONLY",
        "metric_names": list(METRICS),
        "max_profiled_kernel_nodes": count,
        "nvtx_filter": NVTX_RANGE,
        "kernel_filter": KERNEL_FILTER,
        "reference_parity": "MUST_PASS_OR_NO_REPORT_SUCCESS",
        "one_json_no_ncu_rep": True,
        "timing_comparable_to_unprofiled_abba": False,
        "dram_l2_note": (
            "DRAM and L2 counters cover different operations; "
            "do not divide them to claim an L2 hit rate. "
            "Nsight kernel replay may change cache residence."
        ),
        "real_h3_video": "UNRUN",
        "production_promotion": "PROHIBITED_PROFILER_ONLY",
    }
    try:
        chosen, preflight = query_metrics(ncu)
        result.update(preflight)
        result["metric_names"] = list(chosen)
        proc = subprocess.run(
            command(ncu, sys.executable, str(Path(__file__).resolve()),
                    count, metrics=chosen),
            cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=600,
        )
        output = decode_profiler_output(proc.stdout or b"")
        result["ncu_exit_code"] = proc.returncode
        result["strict_child_parity_marker"] = (
            "HTRC_NCU_CHILD_STRICT_PARITY_PASS" in output
        )
        if "ERR_NVGPUCTRPERM" in output:
            result["status"] = "BLOCKED_PERFORMANCE_COUNTER_PERMISSIONS"
            result["error"] = "Nsight GPU counter permission denied; see NVIDIA App developer performance counter setting or elevated shell"
        elif proc.returncode != 0:
            result["error"] = "Nsight run failed; see diagnostic_output_tail"
        else:
            parsed = parse_csv(output, chosen)
            result.update(summary(parsed))
            result["csv_diagnostics"] = inspect_csv(output, chosen)
            result["kernel_samples"] = parsed
            result["completed"] = bool(
                result["status"] == "PROFILE_COUNTERS_COLLECTED" and
                result["strict_child_parity_marker"]
            )
            result["hardware_profile"] = result["completed"]
            if not result["completed"]:
                result["status"] = "FAIL_CLOSED_NO_VALID_COUNTERS_OR_PARITY"
        result["diagnostic_output_tail"] = output[-6000:]
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:1000]
        result["error"] = message
        if "Unsupported NCU metric family" in message:
            result["status"] = "BLOCKED_UNSUPPORTED_NCU_COUNTERS"
        elif "Nsight metric query failed" in message:
            result["status"] = "FAIL_CLOSED_NCU_METRIC_QUERY"
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--ncu", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--count", type=int, default=12)
    args = ap.parse_args()
    if args.child:
        child()
        return 0
    if args.ncu is None or args.output is None:
        ap.error("parent requires --ncu and --output")
    result = run_parent(str(args.ncu), args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"one_json={args.output} status={result['status']}")
    return 0 if result["status"] == "PROFILE_COUNTERS_COLLECTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
