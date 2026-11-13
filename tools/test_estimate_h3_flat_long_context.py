"""No-GPU contracts for source-based long-context HTRC scaling model."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

MODULE=Path(__file__).with_name("estimate_h3_flat_long_context.py")
spec=importlib.util.spec_from_file_location("htrc_long_context_model",MODULE)
assert spec is not None and spec.loader is not None
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LongContextModelContracts(unittest.TestCase):
    def test_source_18723_replay_count(self):
        x=m.estimate_case(tokens=18723)
        self.assertEqual(x["head_dim"],128)
        self.assertEqual(x["exact_leaves"],256)
        self.assertEqual(x["centre_leaf_pairs"],56*293*1170)
        self.assertEqual(x["graph_tile_candidates"][0]["group_cap"],11)
        self.assertEqual(x["graph_tile_candidates"][0]["graph_replays"],114)
        self.assertGreater(x["state_write_gib_per_attention"],7.0)

    def test_82k_budget_replay_shape(self):
        r=m.estimate_case(tokens=82000)
        self.assertEqual([x["tiles"] for x in r["graph_tile_candidates"]],[16,32,64])
        self.assertEqual([x["group_cap"] for x in r["graph_tile_candidates"]],[3,2,1])
        self.assertEqual([x["graph_replays"] for x in r["graph_tile_candidates"]],
                         [1539,1148,1176])
        self.assertEqual(r["least_replay_tiles_under_configured_budget"],32)

    def test_82k_budget_2048(self):
        r=m.estimate_case(tokens=82000,budget_mib=2048)
        self.assertEqual([x["group_cap"] for x in r["graph_tile_candidates"]],[8,5,3])
        self.assertEqual([x["graph_replays"] for x in r["graph_tile_candidates"]],
                         [567,492,399])
        self.assertEqual(r["least_replay_tiles_under_configured_budget"],64)

    def test_quadratic_state_scaling(self):
        a=m.estimate_case(tokens=40759)
        b=m.estimate_case(tokens=82000)
        self.assertGreater(b["state_write_gib_per_attention"],
                           3.9*a["state_write_gib_per_attention"])
        self.assertLess(b["state_write_gib_per_attention"],
                        4.2*a["state_write_gib_per_attention"])
        self.assertGreater(b["state_write_gib_per_attention"],130.0)
        self.assertLess(b["two_pass_score_only_write_gib_per_attention"],
                        b["state_write_gib_per_attention"]/40)

    def test_not_virtual_global_vram_measurement(self):
        r=m.estimate_case(tokens=223605)
        self.assertGreater(r["full_bf16_qkv_and_output_gib"],11.9)
        self.assertTrue(r["capacity_not_a_real_vram_admission"])
        self.assertGreater(r["state_write_gib_per_attention"],995)
        self.assertEqual(r["graph_tile_candidates"][0]["group_cap"],1)

    def test_balanced_same_geometry_but_more_near_leaves(self):
        fast=m.estimate_case(tokens=40759,mode="fast")
        balanced=m.estimate_case(tokens=40759,mode="balanced")
        self.assertEqual(balanced["exact_leaves"],512)
        self.assertEqual(fast["centre_leaf_pairs"],balanced["centre_leaf_pairs"])
        self.assertAlmostEqual(2*fast["selected_key_fraction"],
                               balanced["selected_key_fraction"],places=6)

    def test_score_only_is_explicitly_two_pass(self):
        r=m.estimate_case(tokens=82000)
        self.assertTrue(r["candidate_must_recompute_key_stats"])
        self.assertGreater(r["stage_write_reduction_theoretical"],0.98)
        self.assertEqual(m.STATE_BYTES,392)

    def test_validate_inputs(self):
        bad=(
            dict(tokens=1),
            dict(tokens=82000,heads=0),
            dict(tokens=82000,mode="approx"),
            dict(tokens=82000,budget_mib=300,reserve_mib=256),
            dict(tokens=82000,candidates=(17,)),
            dict(tokens=82000,mem_bandwidth_gbps=0),
        )
        for kw in bad:
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                m.estimate_case(**kw)

    def test_graph_tiles_are_optin_with_parity_gate(self):
        source=(MODULE.parent.parent / "h3" / "htrc_flat_directflow_live.py").read_text(
            encoding="utf-8")
        ast.parse(source)
        self.assertIn("tiles not in (16, 32, 64)",source)
        self.assertIn("torch.equal(group_out, eager_out[0])",source)
        self.assertIn('"window_tiles":tiles',source)
        self.assertIn('"graph_replays_per_attention"',source)
        self.assertIn("max_extra*_MIB",source)

    def test_selection_stats_kernel_has_unchanged_default(self):
        source=(MODULE.parent.parent / "h3" / "htrc_flat_triton.py").read_text(
            encoding="utf-8")
        tree=ast.parse(source)
        methods=[f for f in tree.body if isinstance(f,ast.FunctionDef)
                 and f.name == "_stats_kernel"]
        self.assertEqual(len(methods),1)
        section=ast.get_source_segment(source,methods[0])
        self.assertIsNotNone(section)
        self.assertIn("SCORE_ONLY: tl.constexpr = 0",section)
        self.assertGreaterEqual(section.count("if not SCORE_ONLY:"),8)
        self.assertIn("vb = tl.dot(w, vj, input_precision=PREC)",section)
        self.assertIn("tl.store(LZ + t * nb + l",section)

    def test_score_only_gpu_benchmark_is_isolated(self):
        source=(MODULE.parent / "benchmark_h3_flat_score_only.py").read_text(
            encoding="utf-8")
        ast.parse(source)
        self.assertIn("SCORE_ONLY=int(score_only)",source)
        self.assertIn("VB_sentinel_unchanged",source)
        self.assertIn('far_tile_and_full_output_parity":"UNRUN_NOT_IMPLEMENTED"',source)

    def test_memory_scarcity_fails_closed(self):
        r=m.estimate_case(tokens=223605,budget_mib=385,reserve_mib=256)
        self.assertIsNone(r["least_replay_tiles_under_configured_budget"])
        self.assertTrue(all(not c["feasible_under_budget"] for c in
                            r["graph_tile_candidates"]))


if __name__=="__main__":
    unittest.main()
