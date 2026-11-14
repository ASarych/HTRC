"""CPU-only numerical checks of the HTRC cooperative-KV admission model."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest

SRC = Path(__file__).with_name("estimate_h3_cooperative_kv.py")
spec = importlib.util.spec_from_file_location("htrc_cooperative", SRC)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class CooperativeKVAdmission(unittest.TestCase):
    def test_materialized_16_centre_reuse_and_pair_bytes(self):
        self.assertEqual(model.STATS_BT, 16)
        self.assertEqual(model.K_OR_V_LEAF_BYTES, 4096)
        self.assertEqual(model.REFERENCE_LOGICAL_BYTES_PER_PAIR, 1024)

    def test_long_context_b8_not_enough_and_heavy_accumulator(self):
        case = model.estimate(tokens=82000, centres_per_cta=8)
        self.assertEqual(case["full_key_leaves"], 5125)
        self.assertEqual(case["far_leaves"], 4613)
        self.assertEqual(case["logical_accumulator_kib_per_cta"], 256)
        self.assertEqual(case["split_ctas_per_head_window"], 16)
        self.assertFalse(case["independent_traffic_below_reference"]["direct_recompute"])
        self.assertTrue(case["independent_traffic_below_reference"]["fp32_weight_staging"])
        self.assertFalse(case["single_cta_64_kib_accumulator_review_cap_met"])

    def test_b16_source_advantage_has_512kib_accumulator(self):
        c = model.estimate(tokens=82000, centres_per_cta=16)
        self.assertEqual(c["logical_accumulator_kib_per_cta"], 512)
        self.assertLess(c["direct_recompute_bytes_per_pair"]["worst_union_all_leaves"], 1024)
        self.assertFalse(c["single_cta_64_kib_accumulator_review_cap_met"])

    def test_query_row_splitting_repeats_source_reads(self):
        a = model.estimate(tokens=82000, centres_per_cta=8, query_row_groups=1)
        b = model.estimate(tokens=82000, centres_per_cta=8, query_row_groups=4)
        self.assertEqual(b["logical_accumulator_kib_per_cta"], 64)
        self.assertGreater(b["fp32_weight_staging_bytes_per_pair"]["worst_union_all_leaves"],
                           a["fp32_weight_staging_bytes_per_pair"]["worst_union_all_leaves"])

    def test_w_staging_is_four_times_smaller_than_vb(self):
        c = model.estimate(tokens=15275, heads=4, tiles=32)
        self.assertAlmostEqual(4 * c["fp32_weights_window_mib"],
                               c["reference_vb_window_mib"], delta=0.005)
        self.assertEqual(c["full_key_leaves"],954)
        self.assertEqual(c["exact_near_leaves"],512)

    def test_no_optimistic_gpu_or_vram_claim(self):
        c = model.estimate(tokens=40759)
        self.assertEqual(c["status"], "ANALYTIC_ONLY_NO_GPU")
        self.assertIn("not a measured",c["note"])
        self.assertGreater(c["far_fraction"], 0.79)

    def test_invalid_shapes_rejected(self):
        for kwargs in (
            {"tokens": 15}, {"tokens": 82000, "heads": 0},
            {"tokens": 82000, "tiles": 31},
            {"tokens": 82000, "centres_per_cta": 3},
            {"tokens": 82000, "query_row_groups": 3},
            {"tokens": 82000, "mode": "approx"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.estimate(**kwargs)


if __name__ == "__main__":
    unittest.main()
