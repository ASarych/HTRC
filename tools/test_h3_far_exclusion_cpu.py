"""CPU contract tests; no torch or GPU required."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_h3_far_exclusion import aligned_split, summarize_exclusions


class FarExclusionCPU(unittest.TestCase):
    def test_split_matches_pinned_triton(self):
        self.assertEqual(aligned_split(954, 4, 64), 256)
        self.assertEqual(aligned_split(128, 4, 64), 64)
        self.assertEqual(aligned_split(10, 4, 64), 64)

    def test_full_exclusion_blocks_and_packing(self):
        masks = [[[1] * 64 + [0] * 64]]
        idx = [[[j for j in range(64)]]]
        s = summarize_exclusions(masks, idx, splits=2, bn=64, expected_near=64)
        self.assertEqual(s["current_far_dot_blocks"], 2)
        self.assertEqual(s["fully_excluded_blocks"], 1)
        self.assertEqual(s["optimistic_packed_blocks_per_split_lower_bound"], 1)
        self.assertEqual(s["eligible_far_leaves_all_centers"], 64)
        self.assertEqual(s["block_far_eligible_histogram"]["0"], 1)
        self.assertEqual(s["block_far_eligible_histogram"]["64"], 1)

    def test_no_fully_excluded_block_even_when_majority_near(self):
        mask = [1 if j % 2 == 0 else 0 for j in range(128)]
        ids = [j for j in range(128) if j % 2 == 0]
        s = summarize_exclusions([[mask]], [[ids]], splits=2, bn=64)
        self.assertEqual(s["fully_excluded_blocks"], 0)
        self.assertEqual(s["current_far_dot_blocks"], 2)
        self.assertEqual(s["optimistic_packed_blocks_per_split_lower_bound"], 2)
        self.assertEqual(s["eligible_far_leaves_per_center"], 64)

    def test_two_heads_and_tiles_reconcile(self):
        row = [1, 1, 0, 0, 0, 0, 0, 0]
        s = summarize_exclusions([[row, row], [row, row]],
                                 [[[0, 1], [0, 1]], [[0, 1], [0, 1]]],
                                 splits=2, bn=4)
        self.assertEqual(s["verified_tile_centers"], 4)
        self.assertEqual(s["current_far_dot_blocks"], 8)
        self.assertEqual(s["eligible_far_leaves_all_centers"], 24)
        self.assertEqual(sum(x["current_blocks"] for x in s["per_head"]), 8)

    def test_exclusion_shape_and_topk_validation(self):
        good = [[[1, 0, 1, 0]]]
        bad_cases = [
            ([], []),
            (good, []),
            (good, [[[0, 0]]]),
            (good, [[[0, 4]]]),
            (good, [[[0, 1]]]),
            ([[[1, 0, 2, 0]]], [[[0, 2]]]),
            (good, [[[0]]]),
        ]
        for mask, ids in bad_cases:
            with self.subTest(mask=mask, ids=ids), self.assertRaises(ValueError):
                summarize_exclusions(mask, ids, splits=2, bn=2, expected_near=2)

    def test_pinned_shape_bounds(self):
        # 954 leaves, 512 selected, 442 eligible far leaves.
        mask = [1] * 512 + [0] * 442
        ids = list(range(512))
        s = summarize_exclusions([[mask]], [[ids]], splits=4, bn=64,
                                 expected_near=512)
        self.assertEqual(s["current_far_dot_blocks"], 15)
        self.assertEqual(s["fully_excluded_blocks"], 8)
        self.assertEqual(s["optimistic_packed_blocks_per_split_lower_bound"], 7)
        self.assertEqual(s["eligible_far_leaves_per_center"], 442)
        self.assertAlmostEqual(s["excluded_fraction_of_real_far_leaves"], 512/954)



class ProbeSourceContracts(unittest.TestCase):
    def test_fail_closed_single_json_and_no_micro_timing(self):
        s = Path(__file__).with_name('profile_h3_far_exclusion.py').read_text(encoding='utf-8')
        import ast
        ast.parse(s)
        for x in ('subprocess.check_output', 'if actual != args.expect_head:',
                  'summarize_exclusions(', 'selection_mask_sha256',
                  'PASS_FAR_EXCLUSION_GEOMETRY', 'FAIL_CLOSED',
                  'torch.equal(eager_b.cpu(), outputs[1])',
                  'args.output.write_text(json.dumps(report',
                  'No timing of any new candidate'):
            self.assertIn(x, s)
        self.assertNotIn('torch.cuda.CUDAGraph()', s)
        self.assertNotIn('graph.replay()', s)

    def test_exact_repo_exclusion_semantics(self):
        root = Path(__file__).parents[1]
        tr = root/'h3'/'htrc_flat_triton.py'
        ba = root/'h3'/'htrc_flat_head_batch.py'
        if not tr.exists() or not ba.exists():
            self.skipTest('Requires full exact-head HTRC checkout')
        a, b = tr.read_text(encoding='utf-8'), ba.read_text(encoding='utf-8')
        self.assertIn('sc = tl.where(((ex == 0) & cm)[None, :], sc, float("-inf"))', a)
        self.assertIn('for n0 in range(lo, hi, BN):', a)
        self.assertIn('excl.scatter_(1,ids,1)', b)
        self.assertIn('"EXCL":excl.view(h,centers,nb)', b)
        self.assertIn('"IDX":idx', b)

class RunnerContracts(unittest.TestCase):
    def test_runner_pins_head_and_json(self):
        s=(Path(__file__).with_name('run_h3_far_exclusion_safe.ps1')
           .read_text(encoding='utf-8'))
        for x in ("[string]$ExactHead", "git clone --no-checkout --no-hardlinks",
                  "git -C $tmp switch --detach $ExactHead",
                  "test_h3_far_exclusion_cpu.py", "profile_h3_far_exclusion.py",
                  "flat_far_exclusion_probe.json", "PASS_FAR_EXCLUSION_GEOMETRY",
                  "current_far_dot_blocks -ne 1920",
                  "no_candidate_tile_kernel_timing -ne $true"):
            self.assertIn(x,s)


if __name__ == "__main__":
    unittest.main()
