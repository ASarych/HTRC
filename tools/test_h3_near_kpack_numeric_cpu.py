"""CPU-only fail-closed contracts for pinned near K8+scale packing research."""
from __future__ import annotations
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TRITON = ROOT / 'tools' / 'htrc_near_kpack_triton.py'
BENCH = ROOT / 'tools' / 'benchmark_h3_near_kpack_numeric.py'
RUNNER = ROOT / 'tools' / 'run_h3_near_kpack_numeric_safe.ps1'
CANON = ROOT / 'h3' / 'htrc_flat_triton.py'

CHANGES = (
    ('def _tile_kernel(', 'def near_kpacked_tile_kernel('),
    ('kk = tl.load(K + (key_base + tok)[:, None] * D + d[None, :], mask=jm[:, None], other=0.0)',
     'kk = tl.load(K + (key_base + j)[:, None] * D + d[None, :], mask=jm[:, None], other=0.0)'),
    ('ksc = tl.load(KS + key_base + tok, mask=jm, other=0.0)',
     'ksc = tl.load(KS + key_base + j, mask=jm, other=0.0)'),
    ('vv=tl.load(V+head*V_SH+tok[:,None]*V_SS+d[None,:]*V_SD,',
     'vv=tl.load(V+(t // 32)*V_SH+tok[:,None]*V_SS+d[None,:]*V_SD,'),
)


class NearKPackCPU(unittest.TestCase):
    def test_syntax_only_no_cuda(self):
        ast.parse(TRITON.read_text('utf-8'),filename=str(TRITON))
        ast.parse(BENCH.read_text('utf-8'),filename=str(BENCH))

    def test_original_tile_clone_has_only_four_authorized_changes(self):
        if not CANON.exists():
            self.skipTest('Canonical HTRC source missing')
        src = CANON.read_text('utf-8')
        self.assertEqual(src.count('@triton.jit\ndef _tile_kernel('),1)
        self.assertEqual(src.count('@triton.jit\ndef _rowmax_kernel('),1)
        original = src[src.index('@triton.jit\ndef _tile_kernel('):
                       src.index('@triton.jit\ndef _rowmax_kernel(')]
        clone = TRITON.read_text('utf-8')
        research = clone[clone.index('@triton.jit\ndef near_kpacked_tile_kernel('):]
        for before,after in CHANGES:
            self.assertEqual(original.count(before),1,before)
            self.assertEqual(original.count(after),0,after)
            original = original.replace(before,after)
        self.assertEqual(research,original)

    def test_near_pack_preserves_unsorted_original_leaf_order(self):
        # Four different leaves, keeping input order instead of sorting.
        idx=[9,2,6,4]
        expected=[leaf*16+i for leaf in idx for i in range(16)]
        actual=[idx[j//16]*16+j%16 for j in range(64)]
        self.assertEqual(actual,expected)
        self.assertNotEqual(actual,sorted(actual))
        self.assertEqual([idx[j//16] for j in range(64)][::16],idx)

    def test_pack_kernel_has_original_byte_exact_indices(self):
        s=TRITON.read_text('utf-8')
        for fragment in ('leaf = tl.load(IDX + centre * KX + j // 16,',
                         'source_token = leaf * 16 + j % 16',
                         'source_row = real_head * SRC_KEYS + source_token',
                         'destination_row = centre * NEAR_TOKENS + j',
                         'tl.store(PACKED_K + destination_row',
                         'tl.store(PACKED_KS + destination_row, scale'):
            self.assertIn(fragment,s)

    def test_memory_formula_within_optimistic_graph_cap(self):
        h,t,k,d=4,32,512,128
        scratch=h*t*k*16*(d+4)
        self.assertEqual(scratch,138412032) # exactly 132 MiB
        self.assertEqual(scratch//1048576,132)
        # 56x16 group8 uses exactly 8x16 = 128 pseudo-centres.
        self.assertEqual(8*16*k*16*(d+4),scratch)
        self.assertLess(590+scratch/1048576,768)
        self.assertGreater(768-(590+scratch/1048576),0)

    def test_candidate_keeps_native_v_and_same_idx(self):
        s=BENCH.read_text('utf-8')
        self.assertIn('states["IDX"].view(HEADS*TILES,KX)',s)
        self.assertIn('pk.view(HEADS*TILES*NEAR_TOKENS,D),v,',s)
        self.assertIn('V_DIRECT=1,V_SH=vs[0],V_SS=vs[1],V_SD=vs[2]',s)
        self.assertIn('BATCH_TILES_PER_HEAD=1,BATCH_KEY_ROWS=NEAR_TOKENS',s)
        self.assertIn('num_warps=4,num_stages=2',s)
        self.assertIn('PASS=0,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0',s)

    def test_strict_two_bank_numeric_gate_before_speed(self):
        s=BENCH.read_text('utf-8')
        self.assertIn('for bi,bank in enumerate(banks):',s)
        self.assertIn('compare_far_partials(torch,ref,cand)',s)
        self.assertIn('candidate_full_window_bit_exact',s)
        self.assertIn('second_complete_graph_eager_bit_exact',s)
        self.assertIn('candidate_speed_measurement":"UNRUN_NUMERIC_GATE_ONLY"',s)
        self.assertIn('speed_capture_blocks":[]',s)
        self.assertNotIn('graph.replay()',s)
        self.assertNotIn('torch.cuda.Event(',s)

    def test_exact_sha_before_any_gpu_import(self):
        s=BENCH.read_text('utf-8')
        self.assertLess(s.index('if actual!=args.expect_head:'),
                        s.index('measured=run_gpu()'))
        self.assertIn('    import torch\n',s)
        self.assertNotIn('\nimport torch\n',s[:s.index('def run_gpu():')])

    def test_source_cpu_preflight_runner_and_one_json(self):
        self.assertTrue(RUNNER.exists())
        p=RUNNER.read_text('utf-8')
        self.assertIn('git -C $tmp switch --detach $ExactHead',p)
        self.assertIn('test_h3_near_kpack_numeric_cpu.py',p)
        self.assertIn('flat_near_kpack_numeric.json',p)
        self.assertIn('-IClosedOtherGpuWorkloads',p)
        self.assertNotIn('@traceArgs',p)
        self.assertIn('PASS_NEAR_KPACK_NUMERIC_ONLY',p)
        self.assertIn('NO_GO_NEAR_KPACK_BIT_EXACT',p)
        self.assertIn('PROHIBITED_RESEARCH_ONLY',p)


if __name__ == '__main__':
    unittest.main()
