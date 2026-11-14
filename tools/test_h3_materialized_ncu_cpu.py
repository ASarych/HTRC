"""CPU-only contracts for one-JSON Nsight materialized Graph-node profiler."""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
import re
import types
import unittest
from unittest import mock

SRC = Path(__file__).with_name("profile_h3_flat_materialized_ncu.py")
spec = importlib.util.spec_from_file_location("htrc_ncu_materialized", SRC)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class NcuMaterializedAdmission(unittest.TestCase):
    def test_ncu_command_bounds_and_graph_node_nvtx(self):
        c = model.command("ncu.BAT","python.exe","script.py", 12)
        self.assertEqual(c[0],"ncu.BAT")
        self.assertEqual(c[c.index("--graph-profiling")+1],"node")
        self.assertEqual(c[c.index("--nvtx-include")+1],
                         "HTRC_MATERIALIZED_REPLAY/")
        self.assertEqual(c[c.index("--launch-count")+1],"12")
        self.assertIn(model.KERNEL_FILTER,c)
        self.assertNotIn("--export", c)
        for count in (0,17):
            with self.assertRaises(ValueError):
                model.command("ncu","python","script",count)

    def test_windows_batch_kernel_filter_no_shell_metacharacters(self):
        # Regression: ncu.BAT via cmd.exe treated '|' as pipeline and
        # attempted to execute _rowmax_kernel as an external command.
        c = model.command("ncu.BAT","python.exe","script.py",12)
        k = c[c.index("--kernel-name") + 1]
        self.assertEqual(k,"regex:_[srt][iot][a-z]*_kernel")
        self.assertFalse(any(ch in k for ch in model.WINDOWS_BATCH_META))
        pattern = re.compile(k.removeprefix("regex:"))
        for kernel in ("_stats_kernel", "_rowmax_kernel", "_tile_kernel"):
            self.assertIsNotNone(pattern.search(kernel), kernel)
        for other in ("_merge_kernel", "_merge_tail_kernel", "empty_kernel", "_prep_kernel"):
            self.assertIsNone(pattern.search(other), other)

    def test_ncu_console_output_oem_decode(self):
        message = (
            '"_rowmax_kernel" ne yavlyaetsya komandoy '
            '- Windows cmd pipeline misuse'
        )
        self.assertEqual(model.decode_profiler_output(
            message.encode("utf-8")), message)
        cyrillic = '"_rowmax_kernel" не является внутренней или внешней командой'
        self.assertEqual(
            model.decode_profiler_output(cyrillic.encode("cp866")), cyrillic)

    def test_ncu_csv_parser_filters_noise_and_header_units(self):
        metrics = model.METRICS
        head = ["ID", "Kernel Name", *metrics]
        unit = ["","","bytes","bytes","sectors","registers"]
        a = ["0","_stats_kernel_optim", "4096","2048","240","64"]
        b = ["1","_tile_kernel_other", "12000","1000","400","90"]
        other = ["2","random_kernel", "10","10","10","5"]
        raw = "==PROF== Connected\n" + "\n".join(
            ",".join('"' + str(v) + '"' for v in row)
            for row in (head,unit,a,b,other)
        )
        rows = model.parse_csv(raw)
        self.assertEqual([x["kernel"] for x in rows],["stats","tile"])
        summary = model.summary(rows)
        self.assertEqual(summary["status"],"PROFILE_COUNTERS_COLLECTED")
        self.assertEqual(summary["stages"]["stats"]["metrics_sum"][metrics[0]],4096)
        self.assertEqual(summary["stages"]["tile"]["metrics_sum"][metrics[2]],400)
        self.assertNotIn("launch__registers_per_thread",
                         summary["stages"]["tile"]["metrics_sum"])

    def test_malformed_profiler_stdout_fails_closed(self):
        self.assertEqual(model.parse_csv("==PROF== ERR_NVGPUCTRPERM"),[])
        self.assertEqual(model.summary([])["status"],"PROFILE_PARTIAL")
        self.assertEqual(model.summary([])["profiled_launches"],0)

    def test_cuda_children_only_no_prod_switch(self):
        source=SRC.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('torch.set_grad_enabled(False)',source)
        self.assertIn('flow.graph_attention_bthd(',source)
        self.assertIn('"materialized"',source)
        self.assertIn('"PASS_FIRST_REAL_CALL"',source)
        self.assertIn('torch.equal(second, eager)',source)
        self.assertIn('torch.cuda.nvtx.range_push("HTRC_MATERIALIZED_REPLAY")',source)
        self.assertNotIn('compare_only=True',source)
        self.assertNotIn('fused_vb',source)
        self.assertIn('"ERR_NVGPUCTRPERM"',source)
        self.assertIn('"one_json_no_ncu_rep": True',source)

    def test_sm120_metric_query_selects_op_read_write_without_guessing(self):
        sample = """
Metric Name
dram__bytes_read.sum.pct_of_peak_sustained_elapsed
dram__bytes_op_read.sum
dram__bytes_op_write.sum
lts__t_sectors_srcunit_tex_op_read.sum
launch__registers_per_thread
"""
        chosen = model.select_available_metrics(sample)
        self.assertEqual(chosen, model.METRICS)
        self.assertEqual(chosen[0], "dram__bytes_op_read.sum")
        self.assertEqual(chosen[1], "dram__bytes_op_write.sum")
        self.assertIn("launch__registers_per_thread", chosen)
        command = model.command("ncu.BAT","python.exe","profile.py",12,
                                metrics=chosen)
        self.assertEqual(
            command[command.index("--metrics")+1],
            ",".join(chosen)
        )

    def test_old_dram_aliases_and_l2_fallback_admitted_only_if_listed(self):
        sample = """
dram__bytes_read.sum
dram__bytes_write.sum
lts__t_sectors_op_read.sum
"""
        chosen = model.select_available_metrics(sample)
        self.assertEqual(chosen,(
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "lts__t_sectors_op_read.sum"
        ))
        with self.assertRaisesRegex(ValueError,"Unsupported NCU metric family"):
            model.select_available_metrics(
                "dram__bytes_read.sum\nlts__t_sectors_op_read.sum"
            )

    def test_localized_counter_values_not_zero_filled(self):
        self.assertEqual(model.parse_counter("2\u0430572\u0430000"), 2572000)
        self.assertEqual(model.parse_counter("101\u00a0360"), 101360)
        self.assertEqual(model.parse_counter("1\u202f024"), 1024)
        self.assertEqual(model.parse_counter("1,024"), 1024)
        self.assertEqual(model.parse_counter("0"), 0)
        for bad in ("n/a", "", "4,55", "101\u0430360,25", "nan", "junk"):
            with self.subTest(bad=bad):
                self.assertIsNone(model.parse_counter(bad))

    def test_wide_sm120_csv_uses_queried_names_and_reports_missing(self):
        col = ["ID","Process ID","Kernel Name","Some Device Attribute",
               "dram__bytes_op_read.sum","dram__bytes_op_write.sum",
               "lts__t_sectors_srcunit_tex_op_read.sum"]
        values = [
            ["0","30576","_stats_kernel","12.0",
             "2\u0430572\u0430000","2\u0430147\u0430483\u0430647","123"],
            ["1","30576","_tile_kernel","12.0",
             "100\u0430000","1\u0430024","256"],
        ]
        import csv
        import io
        buf=io.StringIO()
        writer=csv.writer(buf)
        writer.writerow(col)
        writer.writerow(["","","","", "bytes","bytes","sectors"])
        writer.writerows(values)
        raw="==PROF== Connected\n" + buf.getvalue() + \
            "HTRC_NCU_CHILD_STRICT_PARITY_PASS\n"
        selected=model.METRICS[:3]
        parsed=model.parse_csv(raw, selected)
        self.assertEqual(len(parsed),2)
        self.assertEqual(model.summary(parsed)["status"],
                         "PROFILE_COUNTERS_COLLECTED")
        self.assertEqual(parsed[0]["metrics"][selected[0]],2572000)
        report=model.inspect_csv(raw, selected)
        self.assertEqual(report["csv_headers_found"],1)
        self.assertEqual(report["csv_total_columns"],len(col))
        self.assertEqual(report["csv_missing_requested_metric_columns"],[])
        self.assertEqual(report["csv_named_kernel_rows"]["_tile_kernel"],1)
        self.assertEqual(model.parse_csv(raw, (
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            selected[2],
        )),[])
        self.assertFalse(model.inspect_csv(raw, (
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            selected[2],
        ))["csv_header_is_complete_for_required_metrics"])

    def test_metric_query_runs_before_gpu_and_fails_closed_without_names(self):
        fake=types.SimpleNamespace(returncode=0,stdout=(
            "dram__bytes_op_read.sum\n"
            "dram__bytes_op_write.sum\n"
            "lts__t_sectors_srcunit_tex_op_read.sum\n"
        ).encode("ascii"))
        with mock.patch.object(model.subprocess,"run",return_value=fake) as run:
            names,info=model.query_metrics("ncu.BAT")
            self.assertEqual(names, model.METRICS[:3])
            self.assertEqual(info["query_preflight"],
                             "PASS_ADVERTISED_METRIC_NAMES")
            run.assert_called_once()
            self.assertEqual(run.call_args[0][0],
                             ["ncu.BAT","--query-metrics-mode","all"])
        with mock.patch.object(model,"query_metrics",side_effect=ValueError(
            "Unsupported NCU metric family: dram_read_bytes"
        )), mock.patch.object(model.subprocess,"run") as run:
            result=model.run_parent("ncu.BAT",12)
            self.assertEqual(result["status"],"BLOCKED_UNSUPPORTED_NCU_COUNTERS")
            self.assertFalse(result["completed"])
            self.assertFalse(result["hardware_profile"])
            run.assert_not_called()

    def test_one_output_and_no_fake_success(self):
        source=SRC.read_text(encoding="utf-8")
        self.assertIn('args.output.write_text(',source)
        self.assertIn('result["strict_child_parity_marker"]',source)
        self.assertIn('result["completed"] = bool(parsed) and result["strict_child_parity_marker"]',source)
        self.assertIn('return 0 if result["status"] == "PROFILE_COUNTERS_COLLECTED" else 2',source)


if __name__ == "__main__":
    unittest.main()
