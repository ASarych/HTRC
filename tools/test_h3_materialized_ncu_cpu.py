"""CPU-only contracts for one-JSON Nsight materialized Graph-node profiler."""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
import unittest

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
        self.assertIn("regex:(_stats_kernel|_rowmax_kernel|_tile_kernel)",c)
        self.assertNotIn("--export", c)
        for count in (0,17):
            with self.assertRaises(ValueError):
                model.command("ncu","python","script",count)

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

    def test_one_output_and_no_fake_success(self):
        source=SRC.read_text(encoding="utf-8")
        self.assertIn('args.output.write_text(',source)
        self.assertIn('result["strict_child_parity_marker"]',source)
        self.assertIn('result["completed"] = bool(parsed) and result["strict_child_parity_marker"]',source)
        self.assertIn('return 0 if result["status"] == "PROFILE_COUNTERS_COLLECTED" else 2',source)


if __name__ == "__main__":
    unittest.main()
