"""CPU contracts: mandatory short A/B for every long-context HTRC claim.

No GPU is run; numerical disagreement is a measured quality result, not an abort.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import h3_flat_scale_protocol as proto
import h3_flat_quality_reference as exact

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
            ROOT / "tools" / "h3_flat_quality_ab.py",
            ROOT / "tools" / "h3_flat_quality_reference.py",
            ROOT / "h3" / "htrc_flat_directflow_live.py",
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
        self.assertEqual(x["disagreement_policy"],"record_and_continue")
        self.assertEqual(
            x["reference"],"exact_FP32_softmax_over_all_keys_sampled_queries")
        self.assertFalse(x["bit_exact_between_flat_modes_required"])

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

    def test_single_json_numeric_nonabort_gpu_path(self):
        txt=CLI.read_text(encoding="utf-8")
        self.assertIn("args.output",txt)
        self.assertIn("tmp.replace(path)",txt)
        self.assertIn('"status":"UNRUN"',txt)
        self.assertIn('run_quality_case(',txt)
        self.assertIn('"RECORD_PERCENT_AND_CONTINUE"',txt)
        self.assertIn('"numerical_disagreement_aborts":False',txt)
        self.assertNotIn("STOPPED_AT_FAILURE",txt)
        self.assertIn('if args.execute:',txt)
        self.assertIn('default="short"',txt)
        self.assertIn('cap["source_model"]=estimate_case(',txt)
        self.assertIn('tokens=cap["tokens"],heads=56',txt)
        self.assertIn('"no_fullhd_gpu_execution": True',
                      (ROOT/"tools"/"h3_flat_scale_protocol.py").read_text(
                          encoding="utf-8"))

    def test_every_long_gpu_case_follows_short_results(self):
        cases=make(scope="scaling")["cases"]
        self.assertEqual([x["band"] for x in cases],
                         ["short","short","medium","long"])
        self.assertTrue(all(x["required_short_regression"] for x in cases[:2]))
        self.assertFalse(any(x["required_short_regression"] for x in cases[2:]))

    def test_quality_runner_has_same_dense_reference_for_every_mode(self):
        txt=CLI.read_text(encoding="utf-8")
        gpu=(ROOT/"tools"/"h3_flat_quality_ab.py").read_text(encoding="utf-8")
        self.assertIn("from h3_flat_quality_ab import run_case",txt)
        self.assertIn('order=("materialized",)+tuple(variants)',gpu)
        self.assertIn('exact_dense_sampled_queries(*bank,positions',gpu)
        self.assertIn("quality_against_exact(",gpu)
        self.assertIn("normalized_difference_percent(",gpu)
        self.assertIn('outcome["bit_exact_to_materialized_is_not_required"]=True',gpu)
        self.assertIn('"status":"RUNTIME_FAILED"',gpu)
        self.assertNotIn("run_h3_baseline.ps1",txt)
        self.assertNotIn("run_hyperflow_t2v.ps1",txt)

    def test_exact_reference_reads_all_keys_but_samples_queries(self):
        src=(ROOT/"tools"/"h3_flat_quality_reference.py").read_text(
            encoding="utf-8")
        self.assertIn("logits=(qh@kh.transpose(0,1))*scaling",src)
        self.assertIn("probabilities=torch.softmax(logits,dim=-1)",src)
        self.assertIn("out[start:start+len(ix),head,:]=probabilities@vh",src)
        self.assertIn("FP32_softmax_all_KV_for_sampled_query_positions",src)
        self.assertNotIn("torch.topk",src)

    def test_sampled_query_positions_stable_and_inclusive(self):
        self.assertEqual(exact.query_positions(1,64),(0,))
        self.assertEqual(exact.query_positions(5,5),(0,1,2,3,4))
        self.assertEqual(exact.query_positions(10,3),(0,4,9))
        for n in (15275,18723,40759,82000,223605):
            pos=exact.query_positions(n,64)
            self.assertEqual(len(pos),64)
            self.assertEqual(pos[0],0)
            self.assertEqual(pos[-1],n-1)
            self.assertEqual(len(set(pos)),len(pos))
        with self.assertRaises(ValueError):
            exact.query_positions(0,10)

    def test_graph_research_only_relaxes_eager_exact_gate(self):
        graph=(ROOT/"h3"/"htrc_flat_directflow_live.py").read_text(
            encoding="utf-8")
        self.assertIn('compare_only=False):',graph)
        self.assertIn('if not exact and not compare_only:',graph)
        self.assertIn('"relative_l2_to_eager_flat_percent":rel_l2_pct',graph)
        self.assertIn('"strict_parity_required":not compare_only',graph)
        self.assertIn('not compare_only',graph)
        self.assertIn('torch.equal(group_out, eager_out[0])',graph)

    def test_numeric_error_is_percent_not_bit_exact_flag(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable on CPU")
        reference=torch.ones((3,2,128),dtype=torch.float32)
        whole=torch.ones((1,3,2,128),dtype=torch.float32)
        got=exact.quality_against_exact(whole,reference,(0,1,2))
        self.assertAlmostEqual(got["relative_l2_percent"],0.0)
        whole *= 0.5
        got=exact.quality_against_exact(whole,reference,(0,1,2))
        self.assertAlmostEqual(got["relative_l2_percent"],50.0,places=5)
        whole *= 0
        got=exact.quality_against_exact(whole,reference,(0,1,2))
        self.assertAlmostEqual(got["relative_l2_percent"],100.0,places=5)


if __name__ == "__main__":
    unittest.main()
