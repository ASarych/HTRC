"""CPU/source contract tests for research-only 56-head near-K8 full Graph ABBA."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from htrc_near_kpack_graph_adapter import (
    NearKGraphAdapter, SCRATCH_BYTES, GROUP_HEADS, MAX_TILES,
)
from benchmark_h3_near_kpack_graph_abba import decide_abba, CAPTURE_ORDER

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/'tools'/'benchmark_h3_near_kpack_graph_abba.py'
ADAPTER=ROOT/'tools'/'htrc_near_kpack_graph_adapter.py'
RUNNER=ROOT/'tools'/'run_h3_near_kpack_graph_abba_safe.ps1'
TILE=ROOT/'tools'/'htrc_near_kpack_triton.py'


def blocks(control1,packed1,packed2,control2):
    return [
        {'arm':arm,'gpu_event_median_ms':float(v),
         'samples':[{'gpu_ms':float(v),'sync_wall_ms':float(v)+.5}
                    for _ in range(7)]}
        for arm,v in zip(CAPTURE_ORDER,(control1,packed1,packed2,control2))]


class NearKGraphABBAContract(unittest.TestCase):
    def test_syntax_parse(self):
        ast.parse(BENCH.read_text('utf-8'))
        ast.parse(ADAPTER.read_text('utf-8'))

    def test_abba_positive_strict_median_and_worst(self):
        d=decide_abba(blocks(67.0,59.0,59.5,67.2))
        self.assertEqual(d['decision'],'PROMISING_RESEARCH_ONLY')
        self.assertTrue(d['promising_full_graph_signal'])
        self.assertTrue(d['candidate_wins_both_independent_captures'])
        self.assertTrue(d['strict_capture_stability'])
        self.assertGreaterEqual(d['worst_pair_full_graph_reduction_pct'],5)

    def test_abba_small_gain_no_go(self):
        d=decide_abba(blocks(66.7,66.0,66.1,66.8))
        self.assertEqual(d['decision'],'NO_GO_FULL_GRAPH')
        self.assertFalse(d['promising_full_graph_signal'])

    def test_worst_pair_blocks_spurious_median_gain(self):
        d=decide_abba(blocks(67.0,62.0,64.3,68.5))
        self.assertTrue(d['strict_capture_stability'])
        self.assertGreater(d['median_full_graph_reduction_pct'],5)
        self.assertLess(d['worst_pair_full_graph_reduction_pct'],5)
        self.assertEqual(d['decision'],'NO_GO_FULL_GRAPH')

    def test_unstable_controls_inconclusive(self):
        d=decide_abba(blocks(66.0,60.0,60.1,76.0))
        self.assertEqual(d['decision'],'INCONCLUSIVE_CAPTURE_DRIFT')
        self.assertFalse(d['promising_full_graph_signal'])

    def test_broken_abba_order_or_timing_rejected(self):
        with self.assertRaises(ValueError):
            decide_abba(blocks(70,60,60,70)[:3])
        items=blocks(70,60,60,70)
        items[0]['arm']='packed'
        with self.assertRaises(ValueError):
            decide_abba(items)
        items=blocks(70,60,60,70)
        items[0]['gpu_event_median_ms']=float('nan')
        with self.assertRaises(ValueError):
            decide_abba(items)

    def test_one_132mib_scratch_shared_across_two_geometry(self):
        self.assertEqual(GROUP_HEADS,8)
        self.assertEqual(MAX_TILES,16)
        self.assertEqual(SCRATCH_BYTES,138412032)
        self.assertEqual(SCRATCH_BYTES / (1<<20),132)
        s=ADAPTER.read_text('utf-8')
        self.assertIn('self._packed_k = None',s)
        self.assertIn('if self._packed_k is not None:',s)
        self.assertIn("if not self._windows_seen.issuperset({15, 16}):",s)
        self.assertIn('if self.torch.cuda.is_current_stream_capturing():',s)
        self.assertIn('REAL_TILES_PER_HEAD=tiles',s)

    def test_patch_scoped_to_window_not_original_eager(self):
        orig=lambda *args,**kwargs:'original batch'
        batch=SimpleNamespace(launch_prepared_tile_batch=orig)
        flow=SimpleNamespace(_one_window=lambda **kw:batch.launch_prepared_tile_batch())
        adapter=NearKGraphAdapter(None,flow,batch,None,None)
        adapter.launch=lambda *args,**kw:'packed batch'
        old_window=flow._one_window
        with adapter.capturing():
            self.assertEqual(flow._one_window(),'packed batch')
            self.assertIs(batch.launch_prepared_tile_batch,orig)
        self.assertIs(flow._one_window,old_window)
        self.assertIs(batch.launch_prepared_tile_batch,orig)

    def test_original_eager_first_call_not_mutated(self):
        text=BENCH.read_text('utf-8')
        self.assertIn('with adapter.capturing():',text)
        self.assertIn("first=json.loads(gate_path.read_text('utf-8'))",text)
        self.assertIn('verify_first_call(first,0)',text)
        self.assertIn("query_staging='native_torch_mu',kv_staging='native_direct'",text)
        self.assertIn('compare_far_partials(torch,ref,packed)',text)
        self.assertIn("if not all(row['bit_exact'] for row in fields.values()):",text)

    def test_numeric_preflight_before_any_timed_capture(self):
        s=BENCH.read_text('utf-8')
        self.assertLess(
            s.index("result['numeric_gate']='PASS_TWO_BANK_FULL_56HEAD_AND_RAW_PM_PL_PACC_BIT_EXACT'"),
            s.index('timed=_measure(torch,measured_call,args.rounds)'))
        self.assertIn('len(parity)==2',s)
        self.assertIn('adapter.launch(states,cfg,tr,buffers=packed)',s)
        self.assertIn('runtime_graph_capture_hard_768mib_gate',s)
        self.assertIn("'independent_model_resident_vram':'UNRUN'",s)
        self.assertEqual(list(CAPTURE_ORDER),['control','packed','packed','control'])

    def test_modified_kernel_uses_dynamic_original_v_head(self):
        s=TILE.read_text('utf-8')
        self.assertIn('REAL_TILES_PER_HEAD: tl.constexpr = 32',s)
        self.assertIn('vv=tl.load(V+(t // REAL_TILES_PER_HEAD)*V_SH',s)
        self.assertIn('pack_near_key_and_scale',s)

    def test_exact_head_guard_before_cuda_import(self):
        s=BENCH.read_text('utf-8')
        self.assertIn('    import torch  # CUDA only AFTER exact-source verification in main()',s)
        self.assertNotIn('\nimport torch\n',s[:s.index('def run_gpu(')])
        self.assertLess(s.index('if sha!=args.expect_head:'),s.index('report=run_gpu(args,report)'))

    def test_one_json_ps_runner_preflights_cpu_and_exact_head(self):
        p=RUNNER.read_text('utf-8')
        for guard in ("git -C $tmp switch --detach $ExactHead",
                      "test_h3_near_kpack_numeric_cpu.py",
                      "test_h3_near_kpack_graph_abba_cpu.py",
                      "flat_56head_near_kpack_graph_abba.json",
                      "PASS_NUMERIC_NO_GO_FULL_GRAPH",
                      "PASS_NUMERIC_PROMISING_FULL_GRAPH",
                      "PROHIBITED_RESEARCH_ONLY"):
            self.assertIn(guard,p)
        self.assertNotIn('@traceArgs',p)
        self.assertIn('--quiet-confirmed --rounds 7 --warmups 5',p)


if __name__=='__main__':
    unittest.main()
