"""Fast CPU-only contracts for HTRC's research-only far index-packing probe."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_h3_far_pack import reference_pack, verify_pack_outputs


def one_mask(mask, selected, splits=2, bn=2):
    return reference_pack([[mask]], [[selected]], splits=splits, bn=bn,
                          expected_near=len(selected))


def padded(ref, fill=-333):
    indices, counts = [], []
    for h in ref["indices"]:
        rh, ch = [], []
        for t in h:
            rt, ct = [], []
            for leaf_list in t:
                rt.append(leaf_list + [fill] * (ref["split_capacity"] - len(leaf_list)))
                ct.append(len(leaf_list))
            rh.append(rt)
            ch.append(ct)
        indices.append(rh)
        counts.append(ch)
    return indices, counts


class PureCPUReferenceTests(unittest.TestCase):
    def test_simple_split_order(self):
        r = one_mask([0, 1, 0, 1], [1, 3])
        self.assertEqual(r["indices"], [[[[0], [2]]]])
        self.assertEqual(r["packed_indices"], 2)
        self.assertEqual(r["packed_blocks"], 2)
        self.assertEqual(r["split_capacity"], 2)
        p, c = padded(r)
        self.assertTrue(verify_pack_outputs(p, c, r)["valid_prefixes_bit_exact"])

    def test_interleaved_near_and_far_keeps_original_order(self):
        near = [i for i in range(128) if i % 3 == 0]
        mask = [int(i in near) for i in range(128)]
        r = one_mask(mask, near, splits=2, bn=64)
        self.assertEqual(r["indices"][0][0][0], [i for i in range(64) if i % 3])
        self.assertEqual(r["indices"][0][0][1], [i for i in range(64, 128) if i % 3])
        p, c = padded(r)
        audit = verify_pack_outputs(p, c, r)
        self.assertEqual(audit["packed_valid_indices"], 128 - len(near))
        self.assertTrue(audit["padded_suffix_not_read"])

    def test_pinned_tail_split_capacity_and_blocks(self):
        near = list(range(512))
        mask = [int(i < 512) for i in range(954)]
        r = one_mask(mask, near, splits=4, bn=64)
        self.assertEqual(r["split_capacity"], 256)
        self.assertEqual([len(x) for x in r["indices"][0][0]], [0, 0, 256, 186])
        self.assertEqual(r["packed_indices"], 442)
        self.assertEqual(r["packed_blocks"], 7)
        p, c = padded(r, fill=32767)
        self.assertEqual(verify_pack_outputs(p, c, r)["packed_block_lower_bound"], 7)

    def test_nontrivial_two_heads_two_tiles(self):
        row_a = [1, 0, 0, 1, 0, 0, 0, 0]
        row_b = [0, 1, 0, 0, 0, 0, 1, 0]
        r = reference_pack([[row_a, row_b], [row_b, row_a]],
                           [[[0, 3], [1, 6]], [[1, 6], [0, 3]]],
                           splits=2, bn=4)
        self.assertEqual(r["packed_indices"], 24)
        p, c = padded(r)
        self.assertEqual(verify_pack_outputs(p, c, r)["packed_valid_indices"], 24)

    def test_stale_tail_ignored_but_wrong_prefix_rejected(self):
        r = one_mask([1, 0, 1, 0, 1, 0, 1, 0], [0, 2, 4, 6], splits=2, bn=4)
        p, c = padded(r, fill=-2147483648)
        verify_pack_outputs(p, c, r)  # Deliberately undefined suffix is allowed.
        p[0][0][0][0] = 999
        with self.assertRaisesRegex(AssertionError, "order/content"):
            verify_pack_outputs(p, c, r)

    def test_wrong_counts_and_shape_fail_closed(self):
        r = one_mask([1, 0, 1, 0], [0, 2])
        p, c = padded(r)
        c[0][0][0] = 0
        with self.assertRaisesRegex(AssertionError, "count mismatch"):
            verify_pack_outputs(p, c, r)
        p, c = padded(r)
        c[0][0][0] = -1
        with self.assertRaisesRegex(AssertionError, "count invalid"):
            verify_pack_outputs(p, c, r)
        p, c = padded(r)
        with self.assertRaisesRegex(AssertionError, "head count"):
            verify_pack_outputs([], c, r)
        with self.assertRaisesRegex(AssertionError, "tile count"):
            verify_pack_outputs([[]], c, r)

    def test_wrong_excl_idx_not_tolerated(self):
        with self.assertRaises(ValueError):
            one_mask([1, 0, 0, 0], [1])
        with self.assertRaises(ValueError):
            one_mask([1, 1, 0, 0], [0, 0])
        with self.assertRaises(ValueError):
            one_mask([1, 0, 0, 2], [0, 3])


class SourceContracts(unittest.TestCase):
    def test_gpu_script_ast_exact_head_and_fail_closed(self):
        path = HERE / "benchmark_h3_far_pack_cost.py"
        src = path.read_text(encoding="utf-8")
        ast.parse(src)
        for token in (
            'if actual != args.expect_head:',
            '"status": "FAIL_CLOSED"',
            '"status": "PASS_PACK_COST_ONLY"',
            'args.output.write_text(json.dumps(report',
            '"production_promotion": "PROHIBITED_RESEARCH_ONLY"',
            'torch.equal(eager.cpu(), full_outputs[1])',
        ):
            self.assertIn(token, src)
        self.assertIn('"no_packed_attention_kernel": True', src)
        self.assertIn('"no_attention_candidate_timing": True', src)
        self.assertNotIn('launch_prepared_tile_batch(', src)
        self.assertNotIn('torch.nn.functional.scaled_dot_product_attention', src)

    def test_exclusion_and_two_bank_pack_verified_before_graph_timing(self):
        src = (HERE / "benchmark_h3_far_pack_cost.py").read_text(encoding="utf-8")
        self.assertIn('reference_pack(excl.detach().cpu().tolist()', src)
        self.assertIn('verify_pack_outputs(', src)
        self.assertIn('"bank_b_complete_graph_eager_bit_exact": True', src)
        self.assertLess(src.index('for bi, bank in enumerate(banks):'),
                        src.index('for block in range(1, 5):'))
        self.assertLess(src.index('torch.equal(eager.cpu(), full_outputs[1])'),
                        src.index('for block in range(1, 5):'))
        self.assertLess(src.index('_check(torch, packed, counts, ref, f"bank{bi}_eager")'),
                        src.index('for block in range(1, 5):'))
        self.assertIn('"replays_per_sample": args.replays_per_sample', src)
        self.assertIn('"packed_suffix_is_uninitialized_and_must_be_masked": True', src)

    def test_research_kernel_no_extra_scatter_or_stale_tail_reads(self):
        src = (HERE / "htrc_far_pack_triton.py").read_text(encoding="utf-8")
        self.assertIn('keep = valid & (excluded == 0)', src)
        self.assertIn('dest = tl.cumsum(indicator, 0) - 1', src)
        self.assertIn('tl.store(PACKED + (tile * SPLITS + split) * LPS + dest, leaf,', src)
        self.assertIn('mask=keep)', src)
        self.assertIn('tl.store(COUNTS + tile * SPLITS + split, tl.sum(indicator, 0))', src)
        self.assertIn('@triton.jit', src)
        self.assertEqual(src.count('tl.store('), 2)
        launcher = (HERE / "benchmark_h3_far_pack_cost.py").read_text(encoding="utf-8")
        self.assertIn('pack_kernel[(HEADS * TILES, SPLITS)](', launcher)
        self.assertEqual(launcher.count('torch.cuda.CUDAGraph()'), 1)

    def test_runner_disposable_checkout_and_one_json(self):
        src = (HERE / "run_h3_far_pack_safe.ps1").read_text(encoding="utf-8")
        for x in ("git clone --no-checkout --no-hardlinks", "git -C $tmp switch --detach $ExactHead",
                  "test_h3_far_pack_cpu.py", "benchmark_h3_far_pack_cost.py",
                  "flat_far_pack_cost.json", "PASS_PACK_COST_ONLY",
                  "$report.capture_blocks.Count -ne 4", "$report.banks.Count -ne 2",
                  "no_packed_attention_kernel -ne $true"):
            self.assertIn(x, src)

    def test_source_exact_triton_far_split_geometry(self):
        root = HERE.parent
        p_tr = root / "h3" / "htrc_flat_triton.py"
        p_batch = root / "h3" / "htrc_flat_head_batch.py"
        if not p_tr.exists() or not p_batch.exists():
            self.skipTest("Requires full pinned HTRC source checkout")
        tr = p_tr.read_text(encoding="utf-8")
        ba = p_batch.read_text(encoding="utf-8")
        self.assertIn('for n0 in range(lo, hi, BN):', tr)
        self.assertIn('sc = tl.where(((ex == 0) & cm)[None, :], sc, float("-inf"))', tr)
        self.assertIn('excl.scatter_(1,ids,1)', ba)
        self.assertIn('"EXCL":excl.view(h,centers,nb)', ba)
        self.assertIn('"IDX":idx', ba)
        self.assertIn('triton_module._split(leaves,splits,bn)', ba)


if __name__ == "__main__":
    unittest.main()
