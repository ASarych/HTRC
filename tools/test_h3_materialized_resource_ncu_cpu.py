"""CPU-only contracts for HTRC materialized NCU resource triage.

No torch, CUDA, GPU kernel or NVIDIA tool is started by these tests.
"""
from __future__ import annotations

import csv
import importlib.util
import io
from pathlib import Path
import re
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("profile_h3_flat_resource_ncu.py")
spec = importlib.util.spec_from_file_location("htrc_resource_ncu", SOURCE)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fake_csv(*, with_tile=True, parity=True):
    cols = ["ID", "Kernel Name", "gpu__time_duration.sum",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
            "launch__registers_per_thread", "launch__shared_mem_per_block_allocated"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    writer.writerow(["", "", "nsecond", "%", "%", "sector", "register", "byte"])
    writer.writerow(["0", "_stats_kernel", "2а500а000", "49,5", "70,3", "1а024", "80", "15а360"])
    writer.writerow(["1", "_rowmax_kernel", "250000", "36,0", "30,1", "0", "32", "11264"])
    if with_tile:
        writer.writerow(["2", "_tile_kernel", "9а000а000", "16,67", "50,3", "2а048", "255", "50а688"])
    if parity:
        buf.write("HTRC_NCU_CHILD_STRICT_PARITY_PASS\n")
    return buf.getvalue()


METRICS = (
    m.REQUIRED["duration"], m.REQUIRED["active_warps_pct"],
    m.OPTIONAL["dram_saturation_pct"], m.OPTIONAL["local_load_sectors"],
)


class ResourceNCUContracts(unittest.TestCase):
    def test_only_advertised_metrics_are_used(self):
        chosen, missing = m.select_metrics("\n".join(METRICS))
        self.assertEqual(chosen, METRICS)
        self.assertIn("local_store_sectors", missing)
        self.assertNotIn("local_load_sectors", missing)
        self.assertNotIn("sm_saturation_pct", chosen)

    def test_required_metrics_fail_closed_without_fallback(self):
        for text in ("gpu__time_duration.sum", "sm__warps_active.avg.pct_of_peak_sustained_active", ""):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "Required NCU metrics missing"):
                m.select_metrics(text)

    def test_windows_batch_no_shell_meta_and_real_kernel_only(self):
        cmd = m.ncu_command("ncu.BAT", "python.exe", "child.py", METRICS)
        self.assertEqual(cmd[cmd.index("--graph-profiling")+1], "node")
        self.assertEqual(cmd[cmd.index("--launch-count")+1], "12")
        self.assertIn("--nvtx-include", cmd)
        self.assertEqual(cmd[cmd.index("--kernel-name")+1], m.KERNEL_FILTER)
        self.assertFalse(any(c in m.KERNEL_FILTER for c in m.WINDOWS_BATCH_META))
        pattern = re.compile(m.KERNEL_FILTER.removeprefix("regex:"))
        for name in ("_stats_kernel", "_rowmax_kernel", "_tile_kernel"):
            self.assertIsNotNone(pattern.search(name), name)
        for name in ("_merge_tail_kernel", "_merge_kernel", "empty_kernel"):
            self.assertIsNone(pattern.search(name), name)
        with self.assertRaises(ValueError):
            m.ncu_command("ncu", "python", "child", METRICS, count=17)

    def test_localized_float_and_integer(self):
        self.assertEqual(m.numeric("2а500а000", integer=True), 2500000)
        self.assertEqual(m.numeric("16,67"), 16.67)
        self.assertEqual(m.numeric("1\u202f024", integer=True), 1024)
        self.assertEqual(m.numeric("1,024", integer=True), 1024)
        for v in ("n/a", "nan", "", "abc", "-2", "12,5"): 
            with self.subTest(v=v):
                self.assertIsNone(m.numeric(v, integer=True))
        self.assertEqual(m.numeric("12,5"), 12.5)

    def test_wide_realistic_csv_and_stage_metadata(self):
        parsed = m.parse_resource_csv(fake_csv(), METRICS)
        self.assertEqual(len(parsed["samples"]), 3)
        self.assertFalse(parsed["diagnostics"]["missing_metric_columns"])
        self.assertEqual(parsed["diagnostics"]["duration_unit"], "nsecond")
        summary = m.summarize(parsed)
        self.assertTrue(summary["has_stats_and_tile"])
        self.assertEqual(summary["stages"]["tile"]["duration_us_median"],9000.0)
        self.assertEqual(summary["stages"]["tile"]["occupancy_pct_median"],16.67)
        self.assertEqual(summary["stages"]["tile"]["launch_attributes_distinct_values"]["launch__registers_per_thread"], [255])
        self.assertEqual(summary["stages"]["stats"]["optional_metrics"]["local_load_sectors"]["median"], 1024)

    def test_missing_tile_no_success(self):
        parsed = m.parse_resource_csv(fake_csv(with_tile=False), METRICS)
        self.assertFalse(m.summarize(parsed)["has_stats_and_tile"])
        self.assertEqual(len(parsed["samples"]),2)

    def test_nan_or_unavailable_mandatory_counter_rejected(self):
        raw = fake_csv().replace('9а000а000', 'n/a')
        parsed = m.parse_resource_csv(raw, METRICS)
        self.assertEqual(len(parsed["samples"]),2)
        self.assertEqual(parsed["diagnostics"]["rejected_required_metric_rows"],1)

    def test_source_head_rejects_mismatch_without_gpu(self):
        good = "a"*40
        with patch.object(m.subprocess, "run", return_value=types.SimpleNamespace(
            returncode=0,stdout="b"*40+"\n")):
            with self.assertRaisesRegex(RuntimeError,"SOURCE_HEAD_MISMATCH"):
                m.verify_exact_head(good)
        with self.assertRaises(ValueError):
            m.verify_exact_head("bad")

    def test_parent_accepts_only_existing_kernel_parity_and_metrics(self):
        query = types.SimpleNamespace(returncode=0,stdout=("\n".join(METRICS)).encode("ascii"))
        profile = types.SimpleNamespace(returncode=0,stdout=fake_csv().encode("utf-8"))
        with patch.object(m,"verify_exact_head"),patch.object(m.subprocess,"run",side_effect=[query,profile]) as spy:
            report = m.run_parent("ncu.BAT",12,"a"*40)
        self.assertEqual(spy.call_count,2)
        self.assertEqual(report["status"],"PROFILE_RESOURCE_COUNTERS_COLLECTED")
        self.assertTrue(report["strict_child_parity_marker"])
        self.assertTrue(report["hardware_profile"])
        self.assertTrue(report["completed"])
        self.assertEqual(report["parsed_launches"],3)

    def test_parent_fail_closed_when_child_marker_missing(self):
        query = types.SimpleNamespace(returncode=0,stdout=("\n".join(METRICS)).encode("ascii"))
        profile = types.SimpleNamespace(returncode=0,stdout=fake_csv(parity=False).encode("utf-8"))
        with patch.object(m,"verify_exact_head"),patch.object(m.subprocess,"run",side_effect=[query,profile]):
            report = m.run_parent("ncu.BAT",12,"a"*40)
        self.assertEqual(report["status"],"FAIL_CLOSED_RESOURCE_CSV_OR_PARITY")
        self.assertFalse(report["completed"])
        self.assertFalse(report["hardware_profile"])

    def test_parent_query_fails_before_profile(self):
        result = types.SimpleNamespace(returncode=0,stdout=b"gpu__time_duration.sum\n")
        with patch.object(m,"verify_exact_head"),patch.object(m.subprocess,"run",return_value=result) as spy:
            report = m.run_parent("ncu.BAT",12,"a"*40)
        self.assertEqual(spy.call_count,1)
        self.assertEqual(report["status"],"BLOCKED_UNSUPPORTED_REQUIRED_METRICS")
        self.assertFalse(report["completed"])
        self.assertNotIn("ncu_exit_code",report)


if __name__ == "__main__":
    unittest.main()
