"""Strict CPU/source contracts for HTRC's experimental dense-vs-packed far gather."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_h3_far_pack import reference_pack
from analyze_h3_far_gather import gather_geometry,gather_scalar_reference


def reference_for(mask,idx,splits=2,bn=4):
    return reference_pack([[mask]],[[idx]],splits=splits,bn=bn,expected_near=len(idx))


class GatherReferenceTests(unittest.TestCase):
    def test_order_preserved_and_valid_subsequence(self):
        mask=[1,0,1,0,0,1,0,1]
        idx=[0,2,5,7]
        ref=reference_for(mask,idx)
        dense=gather_scalar_reference([[list(range(8))]],ref,dense=True)
        packed=gather_scalar_reference([[list(range(8))]],ref,dense=False)
        self.assertEqual(dense,[[[[0,1,2,3],[4,5,6,7]]]])
        self.assertEqual(packed,[[[[1,3],[4,6]]]])
        self.assertEqual(gather_geometry(ref)['packed_valid_source_leaves'],4)

    def test_last_split_clips_source_without_reading_padding(self):
        mask=[1,0,0,0,0,0,0,0,0,0]
        ref=reference_for(mask,[0],splits=2,bn=4)
        self.assertEqual(gather_scalar_reference([[list(range(10))]],ref,dense=True),
                         [[[[0,1,2,3,4,5,6,7],[8,9]]]])
        self.assertEqual(gather_scalar_reference([[list(range(10))]],ref,dense=False),
                         [[[[1,2,3,4,5,6,7],[8,9]]]])

    def test_fake_values_are_not_reordered_across_splits(self):
        mask=[0 if i%3 else 1 for i in range(16)]
        ref=reference_for(mask,[0,3,6,9,12,15],splits=2,bn=4)
        vals=[[(i*13)%256 for i in range(16)]]
        packed=gather_scalar_reference([vals],ref,dense=False)
        self.assertEqual(packed[0][0][0],[vals[0][i] for i in (1,2,4,5,7)])
        self.assertEqual(packed[0][0][1],[vals[0][i] for i in (8,10,11,13,14)])

    def test_two_head_geometries(self):
        mask=[1,0,0,0,0,0,0,0]
        ref=reference_pack([[mask,mask],[mask,mask]],
                           [[[0],[0]],[[0],[0]]],splits=2,bn=4)
        geo=gather_geometry(ref)
        self.assertEqual(geo['tile_centers'],4)
        self.assertEqual(geo['dense_valid_source_leaves'],32)
        self.assertEqual(geo['packed_valid_source_leaves'],28)
        self.assertAlmostEqual(geo['source_leaf_load_ratio_packed_over_dense'],0.875)

    def test_reject_shape_drift(self):
        ref=reference_for([1,0,0,0],[0])
        for invalid in ([],[[]],[[[0]]],[[[0,1,2]]]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                gather_scalar_reference(invalid,ref,dense=True)

    def test_bad_counts_fail_closed(self):
        ref=reference_for([1,0,0,0],[0])
        ref['packed_indices']+=1
        with self.assertRaises(AssertionError):gather_geometry(ref)

    def test_reference_does_not_need_torch(self):
        src=(HERE/'analyze_h3_far_gather.py').read_text(encoding='utf-8')
        ast.parse(src)
        self.assertNotIn('import torch',src)
        self.assertIn('source_leaf_load_ratio_packed_over_dense',src)


class SourceContracts(unittest.TestCase):
    def test_triton_guarded_prefix_without_padded_read(self):
        src=(HERE/'htrc_far_gather_triton.py').read_text(encoding='utf-8')
        ast.parse(src)
        self.assertIn('for offset in range(0, count, BN):',src)
        self.assertIn('valid = slot < count',src)
        self.assertIn('tl.load(PACKED + pos * LPS + slot, mask=valid, other=0)',src)
        self.assertIn('for offset in range(0, LPS, BN):',src)
        self.assertNotIn('tl.load(PACKED + pos * LPS + slot)',src)
        self.assertNotIn('torch',src)

    def test_benchmark_stays_not_attention_and_fail_closed(self):
        src=(HERE/'benchmark_h3_far_gather_cost.py').read_text(encoding='utf-8')
        ast.parse(src)
        for expected in (
            '"no_packed_attention_kernel":True',
            '"no_far_softmax_parity_claim":True',
            '"attention_candidate":"UNRUN"',
            'if actual!=args.expect_head:',
            '"status":"FAIL_CLOSED"',
            'args.output.write_text(json.dumps(report',
            'for capture_index,arm in enumerate(("control_dense","candidate_packed",',
            '"candidate_packed","control_dense"), 1)',
            'torch.cuda.CUDAGraph()',
            'stop.synchronize()',
            'torch.equal(got, want)',
            '"packing_cost_excluded":True',
        ):
            self.assertIn(expected,src)

    def test_candidate_oracle_never_reads_uninitialized_pack_suffix(self):
        src=(HERE/'benchmark_h3_far_gather_cost.py').read_text(encoding='utf-8')
        self.assertIn('safe_row.extend(split + [0] * (LPS - len(split)))',src)
        self.assertNotIn('torch.where(valid.view(HEADS,TILES,SPLITS,LPS), packed, 0)',src)
        self.assertIn('torch.equal(counts.detach().cpu()',src)

    def test_our_benchmark_does_not_call_partial_or_full_tile_candidate(self):
        src=(HERE/'benchmark_h3_far_gather_cost.py').read_text(encoding='utf-8')
        self.assertNotIn('launch_prepared_tile_batch(',src)
        self.assertNotIn('_tile_kernel[',src)
        self.assertNotIn('pack_cost_from_previous_run',src)

    def test_existing_htrc_source_untouched_contract(self):
        root=HERE.parent
        t=root/'h3'/'htrc_flat_triton.py'
        b=root/'h3'/'htrc_flat_head_batch.py'
        if not t.exists() or not b.exists():
            self.skipTest('Requires complete exact-head HTRC checkout')
        source=t.read_text(encoding='utf-8')
        self.assertIn('for n0 in range(lo, hi, BN):',source)
        self.assertIn('sc = tl.where(((ex == 0) & cm)[None, :], sc, float("-inf"))',source)
        self.assertIn('excl.scatter_(1,ids,1)',b.read_text(encoding='utf-8'))

    def test_runner_exact_head_one_json(self):
        src=(HERE/'run_h3_far_gather_safe.ps1').read_text(encoding='utf-8')
        for expected in ('[string]$ExactHead','git clone --no-checkout --no-hardlinks',
                         'switch --detach $ExactHead','test_h3_far_gather_cpu.py',
                         'benchmark_h3_far_gather_cost.py','flat_far_gather_cost.json',
                         'PASS_GATHER_COST_ONLY','Close ComfyUI','Close Blender'):
            self.assertIn(expected,src)


if __name__=='__main__':
    unittest.main()
