"""Fail-closed Nsight Compute sample of EXISTING materialized Tile32 CUDA Graph nodes.

One JSON only; profiler stdout is consumed in memory, not written as CSV or ncu-rep.
Neither GPU timing under ncu nor an L2 hit rate is a production speed claim.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NVTX_RANGE = "HTRC_MATERIALIZED_REPLAY/"
# ncu.BAT is parsed by cmd.exe on Windows even if subprocess shell=False.
# Do not pass metacharacters such as | or parentheses via this .BAT wrapper.
# The bracket class matches stats, rowmax and tile without shell alternation.
KERNEL_FILTER = "regex:_[srt][a-z]*_kernel"
WINDOWS_BATCH_META = "|&<>^()%!"
METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "launch__registers_per_thread",
)


def command(ncu: str, py: str, self_path: str, count: int = 12) -> list[str]:
    if not 1 <= count <= 16:
        raise ValueError("Profiled launches must be 1..16")
    if any(ch in KERNEL_FILTER for ch in WINDOWS_BATCH_META):
        raise ValueError("ncu.BAT filter contains unsafe Windows cmd metacharacters")
    return [
        ncu, "--nvtx", "--nvtx-include", NVTX_RANGE,
        "--graph-profiling", "node",
        "--kernel-name-base", "function", "--kernel-name", KERNEL_FILTER,
        "--launch-count", str(count), "--metrics", ",".join(METRICS),
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


def parse_csv(output: str) -> list[dict]:
    """Nsight raw CSV: header row, units row, then launch rows."""
    header = None
    rows = []
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        if "Kernel Name" in row and METRICS[0] in row:
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
        for metric in METRICS:
            v = item.get(metric, "").replace(",", "").strip()
            if not v or v.lower() in ("n/a", "nan", "inf", "-inf"):
                continue
            try:
                metrics[metric] = float(v)
            except ValueError:
                continue
        if METRICS[0] in metrics and METRICS[2] in metrics:
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
        proc = subprocess.run(
            command(ncu, sys.executable, str(Path(__file__).resolve()), count),
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
            parsed = parse_csv(output)
            result.update(summary(parsed))
            result["kernel_samples"] = parsed
            result["completed"] = bool(parsed) and result["strict_child_parity_marker"]
            result["hardware_profile"] = result["completed"]
            if not result["completed"]:
                result["status"] = "FAIL_CLOSED_NO_VALID_COUNTERS_OR_PARITY"
        result["diagnostic_output_tail"] = output[-6000:]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:1000]
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
