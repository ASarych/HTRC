"""CPU contracts: mandatory short A/B for every long-context HTRC claim.

No GPU is run, and no PASS_NUMERIC status is allowed from plan-only mode.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import h3_flat_scale_protocol as proto

CLI = ROOT / "tools" / "benchmark_h3_flat_scale_matrix.py"


def make(scope="short",variants=("split_stream","fused_vb")):
    return proto.plan(
        variants,scope,heads=4,tiles=16,mode="fast",
        rounds=3,budget_mib=1024,reserve_mib=256,
    )


class ScaleProtocolContracts(unittest.TestCase):
    def test_all_files_compile(self):
        paths=(
            ROOT / "tools" / "h3_flat_scale_protocol.py",
            CLI,
            Path(__file__),
        )
        for path in paths:
            with self.subTest(path=str(path)):
                compile(path.read_text(encoding="utf-8"),str(path),"exec")

    def test_all_scopes_have_both_short_controls_first(self):
        for scope in proto.SCOPES:
            with self.subTest(scope=scope):
                cases=proto.select_cases(scope)
                self.assertEqual(
                    [(c.band,c.tokens) for c in cases[:2]],
                    [("short",15275),("short",18723)]
                )

    def test_long_scope_adds_medium_and_long_not_only_long(self):
        cases=proto.select_cases("scaling")
        self.assertEqual(
            [c.tokens for c in cases],
            [15275,18723,40759,82000]
        )
        self.assertTrue(all(c.gpu_eligible for c in cases))

    def test_capacity_fullhd_is_analytical_only(self):
        cases=proto.select_cases("capacity")
        self.assertEqual(cases[-1].tokens,223605)
        self.assertEqual(cases[-1].band,"capacity")
        self.assertFalse(cases[-1].gpu_eligible)

    def test_each_variant_classified_from_source_strategy(self):
        x=make()
        self.assertEqual(
            x["approach_classes"],
            {"split_stream":"length_amplified_candidate",
             "fused_vb":"length_amplified_candidate"}
        )
        self.assertTrue(x["classification_is_hypothesis_not_evidence"])

    def test_all_length_group_is_separate_from_long_group(self):
        expected={
            "graph_dispatch":"universal_candidate",
            "native_qkv_staging":"universal_candidate",
            "stats_score_only":"universal_candidate",
            "graph_window_tiles":"length_amplified_candidate",
            "split_stream":"length_amplified_candidate",
            "fused_vb":"length_amplified_candidate",
            "multi_centre_kv_reuse":"length_amplified_candidate",
            "qkv_streaming":"capacity_enabler",
            "workspace_grouping":"capacity_enabler",
        }
        self.assertEqual(
            {name:record["class"] for name,record in proto.APPROACHES.items()},
            expected
        )

    def test_reject_unsupported_gpu_variants(self):
        bad=((),("materialized",),("split_stream","split_stream"),
             ("untested_future_mode",),("fused_vb","other"))
        for variant in bad:
            with self.subTest(variant=variant),self.assertRaises(ValueError):
                make(variants=variant)

    def test_bad_scopes_rejected(self):
        with self.assertRaises(ValueError):
            proto.select_cases("long_only")
        with self.assertRaises(ValueError):
            make(scope="long_only")

    def test_shape_and_budget_guard(self):
        for change in (
            {"heads":0}, {"heads":57}, {"tiles":17},
            {"mode":"approx"}, {"rounds":1},
            {"budget_mib":384}, {"reserve_mib":-1}
        ):
            opts={
                "heads":4,"tiles":16,"mode":"fast","rounds":3,
                "budget_mib":1024,"reserve_mib":256
            }
            opts.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                proto.plan(("split_stream",),"short",**opts)

    def test_single_json_and_fail_closed_gpu_path(self):
        txt=CLI.read_text(encoding="utf-8")
        self.assertIn("args.output",txt)
        self.assertIn("tmp.replace(path)",txt)
        self.assertIn('"status": "UNRUN"',txt)
        self.assertIn('report["status"] = "FAIL_CLOSED"',txt)
        self.assertIn('report["gpu_execution"] = "STOPPED_AT_FAILURE"',txt)
        self.assertIn('if args.execute:',txt)
        self.assertIn('default="short"',txt)
        self.assertIn('"no_fullhd_gpu_execution": True',
                      (ROOT/"tools"/"h3_flat_scale_protocol.py").read_text(
                          encoding="utf-8"))

    def test_every_long_gpu_case_follows_short_results(self):
        cases=make(scope="scaling")["cases"]
        self.assertEqual([x["band"] for x in cases],
                         ["short","short","medium","long"])
        self.assertTrue(all(x["required_short_regression"] for x in cases[:2]))
        self.assertFalse(any(x["required_short_regression"] for x in cases[2:]))

    def test_legacy_runner_is_not_replaced(self):
        text=CLI.read_text(encoding="utf-8")
        self.assertIn("benchmark_h3_flat_streamed_vb_ab.py",text)
        self.assertIn("benchmark_h3_flat_fused_vb_ab.py",text)
        self.assertIn('["materialized"]["sync_wall_ms_per_call"]',text)
        self.assertNotIn("run_h3_baseline.ps1",text)
        self.assertNotIn("run_hyperflow_t2v.ps1",text)


if __name__ == "__main__":
    unittest.main()
