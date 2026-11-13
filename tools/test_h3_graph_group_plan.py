"""CPU-only contracts for opt-in Direct KV memory-to-speed Graph planner.

Source/AST checks are not GPU compilation, numeric parity or speed PASS.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/"h3"/"htrc_flat_graph_group_plan.py"
FLOW=ROOT/"h3"/"htrc_flat_directflow_live.py"
LIVE=ROOT/"h3"/"htrc_flat_live_backend.py"
RUNNER=ROOT/"tools"/"benchmark_h3_graph_group_policy.py"

spec=importlib.util.spec_from_file_location("graph_group_plan_test",HELPER)
assert spec is not None and spec.loader is not None
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def plan(tokens, **kwargs):
    config=dict(tokens=tokens,heads=56,tiles=16,mode="balanced",
                vb_mode="materialized",max_group=32,budget_mib=1024,
                reserve_mib=256,max_extra_mib=768,free_mib=5200,
                min_free_mib=256)
    config.update(kwargs)
    return m.choose_group(**config)


class TestGraphMemoryToSpeed(unittest.TestCase):
    def test_syntax(self):
        for file in (HELPER,FLOW,LIVE,RUNNER):
            with self.subTest(file=file.name):
                compile(file.read_text(encoding="utf8"),str(file),"exec")

    def test_native_kv_views_not_double_counted(self):
        x=m.graph_stage_head_bytes(tokens=18723,tiles=16,mode="balanced")
        self.assertEqual(x["duplicated_kbf_vbf_bytes_per_head"],0)
        self.assertEqual(x["stage_bf16_kv_bytes_per_head"],2*18723*128*2)
        self.assertEqual(
            x["native_kv_cache_k8_ks_bytes_per_head"],
            (18723//16*16)*(128+4))
        self.assertEqual(x["exact_leaves"],512)

    def test_same_tile_count_different_valid_rows_need_two_graphs(self):
        x=m.graph_stage_head_bytes(tokens=2024,tiles=16,mode="balanced")
        self.assertEqual(x["unique_graph_geometries"],2)
        self.assertEqual([w["tiles"] for w in x["windows"]],[16,16])
        self.assertEqual([w["valid_rows"] for w in x["windows"]],[1024,1000])

    def test_graph_split_and_fused_only_change_vb_workspace(self):
        args=dict(tokens=82000,tiles=16,mode="balanced")
        original=m.graph_stage_head_bytes(vb_mode="materialized",**args)
        streamed=m.graph_stage_head_bytes(vb_mode="split_stream",**args)
        fused=m.graph_stage_head_bytes(vb_mode="fused_vb",**args)
        self.assertGreater(original["estimated_persisted_bytes_per_head"],
                           streamed["estimated_persisted_bytes_per_head"])
        self.assertGreater(streamed["estimated_persisted_bytes_per_head"],
                           fused["estimated_persisted_bytes_per_head"])

    def test_balanced_short_group14_avoids_ragged_group1(self):
        for tokens in (15275,18723):
            with self.subTest(tokens=tokens):
                x=plan(tokens)
                self.assertEqual(x["group_size"],14)
                self.assertEqual(x["group_count"],4)
                self.assertEqual(x["unique_graph_session_sizes"],[14])
                self.assertLess(x["estimated_total_extra_mib"],
                                x["available_extra_mib"])
                self.assertEqual(x["speedup_status"],"UNRUN")
                self.assertTrue(x["estimate_is_not_actual_cuda_graph_pool"])

    def test_long_model_chooses_4_with_capacity_not_speed_claim(self):
        x=plan(82000)
        self.assertEqual(x["group_size"],4)
        self.assertEqual(x["group_count"],14)
        self.assertEqual(x["unique_graph_session_sizes"],[4])
        self.assertEqual(x["quality_status"],"UNRUN")

    def test_medium_may_choose_legacy_size(self):
        x=plan(40759)
        self.assertEqual(x["group_size"],5)
        self.assertEqual(x["group_count"],12)
        self.assertEqual(x["unique_graph_session_sizes"],[5,1])

    def test_budget_or_free_limit_fails_closed(self):
        for x in (dict(budget_mib=400),
                  dict(free_mib=320,min_free_mib=256),
                  dict(max_extra_mib=40)):
            with self.subTest(config=x),self.assertRaises(MemoryError):
                plan(82000,**x)

    def test_invalid_config_rejected(self):
        bad=(dict(tokens=3),dict(max_group=0),dict(tiles=8),
             dict(mode="unknown"),dict(vb_mode="auto"),
             dict(free_mib=float("nan")),dict(reserve_mib=-1),
             dict(heads=0))
        for changes in bad:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                plan(15275,**changes)

    def test_legacy_group_is_default_even_after_optin(self):
        text=FLOW.read_text(encoding="utf8")
        self.assertIn('"APL_HTRC_DIRECTFLOW_GROUP_POLICY","legacy"',text)
        self.assertIn('32 if policy=="reuse_aware" else 11',text)
        self.assertIn('policy not in ("legacy","reuse_aware")',text)
        self.assertIn("shared._head_plan(",text)
        self.assertIn("group_plan.choose_group(",text)
        self.assertIn("extra_reserved > max_extra*_MIB",text)
        self.assertIn("free < min_free*_MIB",text)
        self.assertIn("torch.equal(group_out, eager_out[0])",text)

    def test_graph_sessions_key_on_policy_and_group(self):
        text=FLOW.read_text(encoding="utf8")
        self.assertIn('compare_only, policy)',text)
        self.assertIn('key = (h,)',text)
        self.assertIn('group_policy":policy',text)
        self.assertIn('"head_group_cap":group_cap',text)
        self.assertIn('if not exact and not compare_only:',text)

    def test_live_flat_quality_default_balanced(self):
        src=LIVE.read_text(encoding="utf8")
        self.assertIn('heads, mode="balanced", tiles=16',src)
        self.assertIn('os.environ.get("APL_HTRC_FLAT_DISPATCH","eager")',src)

    def test_quality_ab_is_nonfatal(self):
        src=RUNNER.read_text(encoding="utf8")
        self.assertIn("quality_against_exact(",src)
        self.assertIn("relative_l2_percent",src)
        self.assertIn("graph_attention_bthd(",src)
        self.assertIn("compare_only=True",src)
        self.assertIn("materialized",src)
        self.assertIn("reuse_aware",src)
        self.assertIn("flat_graph_group_balanced_ab.json",src)
        self.assertIn("RUNTIME_FAILED",src)
        self.assertNotIn("raise AssertionError(",src)


if __name__=="__main__":
    unittest.main()
