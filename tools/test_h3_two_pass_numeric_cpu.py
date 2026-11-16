"""CPU-only contracts for original Tile PASS1+PASS2 split research."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from htrc_two_pass_tile_adapter import (
    PINNED_GRID, TwoPassTileProxy, launch_two_pass_tile,
    validate_original_pass_source,
)

ROOT=Path(__file__).resolve().parents[1]
ADAPTER=ROOT/'tools'/'htrc_two_pass_tile_adapter.py'
BENCH=ROOT/'tools'/'benchmark_h3_two_pass_numeric.py'
RUNNER=ROOT/'tools'/'run_h3_two_pass_numeric_safe.ps1'
ORIGINAL=ROOT/'h3'/'htrc_flat_triton.py'


class FakeKernel:
    def __init__(self, fail_on=None):
        self.calls=[]
        self.fail_on=fail_on
    def __getitem__(self, grid):
        def launch(*args,**kw):
            self.calls.append((tuple(grid), args, dict(kw)))
            if kw['PASS']==self.fail_on:
                raise RuntimeError('synthetic pass failure')
            return object()
        return launch


def pinned_kwargs():
    return dict(PASS=0,TAIL=0,DOT=1,I8=1,RB=1,GR=1,
                FAR_PV_ACC=0,NEAR_PV_ACC=0,LATE_NEAR_Q=0,
                BN=64,BN_NEAR=64,BATCH_TILES_PER_HEAD=32,
                T=64,D=128)


def fake_states():
    return {'Q':SimpleNamespace(shape=(4,32,64,128)),
            '_vb_mode':'materialized','_direct_q':True}


def fake_config():
    return {'splits':4,'exact_leaves':512,
            'kern':{'tile_bn':64,'tile_warps':4,'tile_stages':2}}


class TwoPassCPU(unittest.TestCase):
    def test_python_source_parse_without_cuda(self):
        for file in (ADAPTER,BENCH):
            ast.parse(file.read_text('utf-8'),filename=str(file))

    def test_original_source_pass_handoff_contract(self):
        original=ORIGINAL.read_text('utf-8')
        self.assertTrue(validate_original_pass_source(original))
        self.assertIn('if PASS != 2:',original)
        self.assertIn('if PASS != 1:',original)

    def test_order_and_identical_partial_operand_pointers(self):
        kernel=FakeKernel()
        proxy=TwoPassTileProxy(kernel)
        grid=PINNED_GRID
        a=object()
        ptr=[object() for _ in range(22)]
        ptr[9],ptr[10],ptr[11]=a,object(),object()
        result=proxy[grid](*ptr,**pinned_kwargs())
        self.assertIs(result,proxy.compiled[1])
        self.assertEqual(len(kernel.calls),2)
        self.assertEqual([c[2]['PASS'] for c in kernel.calls],[1,2])
        self.assertEqual([c[0] for c in kernel.calls],[grid,grid])
        for position in (9,10,11):
            self.assertIs(kernel.calls[0][1][position],
                          kernel.calls[1][1][position])
        self.assertEqual(proxy.calls,1)

    def test_modified_pass_or_geometry_refused_before_launch(self):
        kernel=FakeKernel()
        proxy=TwoPassTileProxy(kernel)
        args=(object(),)*22
        with self.assertRaises(AssertionError):
            proxy[PINNED_GRID](*args,**dict(pinned_kwargs(),GR=2))
        with self.assertRaises(AssertionError):
            proxy[(32,4,1)]
        self.assertEqual(len(kernel.calls),0)

    def test_patch_restores_original_kernel_after_success(self):
        kernel=FakeKernel()
        tr=SimpleNamespace(_tile_kernel=kernel)
        states=fake_states();cfg=fake_config();buf=object()
        class Batch:
            def launch_prepared_tile_batch(self, states, cfg, tr, *, buffers):
                compiled=tr._tile_kernel[PINNED_GRID](
                    *([object()]*22),**pinned_kwargs())
                return buffers, compiled
        got, compiled=launch_two_pass_tile(Batch(),states,cfg,tr,buffers=buf)
        self.assertIs(got,buf)
        self.assertIs(tr._tile_kernel,kernel)
        self.assertEqual(len(kernel.calls),2)
        self.assertIsNot(compiled[0],compiled[1])

    def test_patch_restores_original_kernel_after_failure(self):
        kernel=FakeKernel(fail_on=2)
        tr=SimpleNamespace(_tile_kernel=kernel)
        class Batch:
            def launch_prepared_tile_batch(self, states, cfg, tr, *, buffers):
                return buffers,tr._tile_kernel[PINNED_GRID](
                    *([object()]*22),**pinned_kwargs())
        with self.assertRaisesRegex(RuntimeError,'synthetic pass failure'):
            launch_two_pass_tile(Batch(),fake_states(),fake_config(),
                                 tr,buffers=object())
        self.assertIs(tr._tile_kernel,kernel)
        self.assertEqual([call[2]['PASS'] for call in kernel.calls],[1,2])

    def test_source_head_gate_and_raw_bit_checks(self):
        s=BENCH.read_text('utf-8')
        self.assertIn("if actual!=args.expect_head:",s)
        self.assertLess(s.index('if actual!=args.expect_head:'),
                        s.index('data=run_gpu()'))
        self.assertIn("    import torch  # imported ONLY after exact source HEAD admission",s)
        self.assertNotIn('\nimport torch\n',s[:s.index('def run_gpu():')])
        self.assertIn('compare_far_partials(torch,control,candidate)',s)
        self.assertIn('_compare_bitwise(torch,ctrl_merged,merge_out)',s)
        self.assertIn('torch.equal(eager_b.cpu(),full_outputs[1])',s)
        self.assertIn('EXPECTED_MASKS[bi]',s)
        self.assertIn('candidate_both_banks_bit_exact',s)
        self.assertIn("'candidate_complete_56head_graph':'UNRUN'",s)
        self.assertIn("'candidate_speed_measurement':'UNRUN_NUMERIC_ONLY'",s)
        self.assertNotIn('torch.cuda.Event(',s)

    def test_scratch_reused_no_new_kv_pack(self):
        s=ADAPTER.read_text('utf-8')
        self.assertIn("with patch.object(tr, '_tile_kernel', proxy):",s)
        self.assertIn("**dict(kwargs, PASS=1)",s)
        self.assertIn("**dict(kwargs, PASS=2)",s)
        self.assertNotIn('torch.empty',s)
        self.assertNotIn('triton.jit',s)

    def test_runner_pins_cpu_gate_and_one_json(self):
        s=RUNNER.read_text('utf-8')
        for phrase in ("git -C $tmp switch --detach $ExactHead",
                       "test_h3_two_pass_numeric_cpu.py",
                       "test_h3_flat_directflow_live_contract.py",
                       "flat_tile_far_near_pass12_numeric.json",
                       "PASS_TWO_PASS_NUMERIC_ONLY",
                       "UNRUN_NUMERIC_ONLY",
                       "PROHIBITED_RESEARCH_ONLY"):
            self.assertIn(phrase,s)
        self.assertLess(s.index('test_h3_two_pass_numeric_cpu.py'),
                        s.index('& $py -B $benchmark --expect-head'))
        self.assertNotIn('@traceArgs',s)

    def test_numeric_extra_partial_traffic_formula(self):
        # 8 native heads, 16 centres, 4 splits, 64 Q rows,
        # FP32 PM+PL+PACC = (1+1+128) * 4 B per row.
        per_window=8*16*4*64*130*4
        self.assertEqual(per_window,17039360)
        self.assertEqual(per_window/1048576,16.25)
        self.assertEqual(PINNED_GRID,(128,4,1))


if __name__=='__main__':
    unittest.main()
