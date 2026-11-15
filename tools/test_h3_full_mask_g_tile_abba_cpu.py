"""CPU decisions, source and runner contracts for research full mask-G Tile A/B/B/A."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark_h3_full_mask_g_tile_abba import (
    ARMS, MAX_INTER_CAPTURE_SPREAD_PCT, MIN_REDUCTION_PCT,
    decide_abba, resources_admitted, validate_request,
)


class Decisions(unittest.TestCase):
    def test_order_and_gate_numbers(self):
        self.assertEqual(ARMS, ('control', 'mask_g', 'mask_g', 'control'))
        self.assertEqual(MIN_REDUCTION_PCT, 5.0)
        self.assertEqual(MAX_INTER_CAPTURE_SPREAD_PCT, 3.0)

    def test_stable_strong_win(self):
        d=decide_abba([0.385,0.386],[0.351,0.3515])
        self.assertEqual(d['status'],'PASS_TILE_ONLY_SPEED_GATE')
        self.assertTrue(d['inter_capture_stable'])
        self.assertTrue(d['all_mask_g_captures_at_least_five_pct_faster_than_all_controls'])
        self.assertGreater(d['mask_g_time_reduction_pct'],5)

    def test_stable_weak_gain_is_no_go(self):
        d=decide_abba([0.385,0.386],[0.373,0.374])
        self.assertEqual(d['status'],'NO_GO_TILE_ONLY')
        self.assertFalse(d['all_mask_g_captures_at_least_five_pct_faster_than_all_controls'])

    def test_slower_is_no_go(self):
        d=decide_abba([0.385,0.3852],[0.391,0.3913])
        self.assertEqual(d['status'],'NO_GO_TILE_ONLY')
        self.assertLess(d['mask_g_time_reduction_pct'],0)

    def test_unstable_control_or_candidate_inconclusive(self):
        self.assertEqual(decide_abba([0.385,0.411],[0.320,0.321])['status'],
                         'INCONCLUSIVE_TILE_ONLY')
        self.assertEqual(decide_abba([0.385,0.3851],[0.320,0.345])['status'],
                         'INCONCLUSIVE_TILE_ONLY')

    def test_median_good_but_worst_capture_inadequate(self):
        d=decide_abba([0.385,0.395],[0.373,0.3731])
        self.assertEqual(d['status'],'NO_GO_TILE_ONLY')
        self.assertGreater(d['mask_g_time_reduction_pct'],4.0)
        self.assertLess(d['worst_cross_capture_time_reduction_pct'],5.0)

    def test_non_finite_missing_and_zero_medians_fail(self):
        bad=([0.1],[0.2,0.21,0.22],[0.0,0.1],[-0.1,0.2],[float('nan'),0.2],
             [float('inf'),0.2])
        for a in bad:
            with self.subTest(a=a),self.assertRaises(ValueError):
                decide_abba(a,[0.1,0.11])

    def test_resource_fail_closed(self):
        control={'registers_per_thread':255,'shared_bytes_per_cta':49664,'compiler_spills':0}
        self.assertTrue(resources_admitted(control,dict(control)))
        for patch in ({'compiler_spills':1},{'registers_per_thread':256},
                      {'shared_bytes_per_cta':49665},
                      {'registers_per_thread':None}):
            cand={**control,**patch}
            self.assertFalse(resources_admitted(control,cand))
        self.assertFalse(resources_admitted({**control,'compiler_spills':None},control))
        self.assertFalse(resources_admitted({**control,'registers_per_thread':254},control))

    def test_cli_minimum_samples(self):
        class Args:
            rounds=7;warmups=5;replays_per_sample=16
        validate_request(Args())
        for field,value in [('rounds',6),('rounds',13),('warmups',4),
                            ('warmups',13),('replays_per_sample',1)]:
            obj=Args();setattr(obj,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_request(obj)


class CodeAndRunner(unittest.TestCase):
    def test_benchmark_no_new_math_and_strict_source_gates(self):
        s=(HERE/'benchmark_h3_full_mask_g_tile_abba.py').read_text(encoding='utf-8')
        ast.parse(s)
        for token in (
            "_launch_full_candidate", "_compare_bitwise", "_merge",
            "batch.launch_prepared_tile_batch", "full_mask_g_partial",
            "summarize_exclusions", "EXPECTED_MASKS[bi]",
            "'PASS_FIRST_REAL_CALL'", "dispatch='eager'",
            "'mirror', 'mask_g'", "_verify_partials_and_merge(",
            "torch.cuda.CUDAGraph()", "with torch.cuda.graph(graph):",
            "graph.replay()", "start.elapsed_time(stop)",
            "(1.0 - gm / cm) * 100.0", "resources_admitted(",
            "'NO_GO_RESOURCE_GATE'", "'INCONCLUSIVE_TILE_ONLY'",
            "'PASS_TILE_ONLY_SPEED_GATE'", "'NO_GO_TILE_ONLY'",
            "'FAIL_CLOSED'", "'UNRUN_NUMERIC_GATE'",
            "'ONE_WINDOW_TILE_KERNEL_ONLY_NO_MERGE'",
        ):
            self.assertIn(token,s)
        self.assertLess(s.index('actual != args.expect_head:'),s.index('report.update(run_gpu(args))'))
        self.assertLess(s.index("report['numeric_gate'] ="),s.index('for index, arm in enumerate(ARMS'))
        self.assertNotIn('from htrc_packed_far_numeric_triton',s)
        self.assertNotIn('import torch\n',s.split('def run_gpu(args):')[0])

    def test_no_production_edits_contract(self):
        doc=HERE.parent/'research'/'H3_FULL_MASK_G_TILE_ABBA_ADMISSION_2026-11-15.md'
        self.assertTrue(doc.is_file())
        text=doc.read_text(encoding='utf-8')
        self.assertIn('no changes to h3/',text)
        self.assertIn('Tile-only',text)
        self.assertIn('5%',text)

    def test_runner_single_json_and_exact_checkout(self):
        s=(HERE/'run_h3_full_mask_g_tile_abba_safe.ps1').read_text(encoding='utf-8')
        for token in ('git clone --no-checkout --no-hardlinks',
                      'git -C $tmp switch --detach $ExactHead',
                      'test_h3_full_mask_g_numeric_cpu.py',
                      'test_h3_full_mask_g_tile_abba_cpu.py',
                      'flat_full_mask_g_tile_abba.json',
                      'Get-NetTCPConnection -LocalPort 8188',
                      'Get-Process -Name blender',
                      "'PASS_TILE_ONLY_SPEED_GATE'", "'NO_GO_TILE_ONLY'",
                      "'INCONCLUSIVE_TILE_ONLY'", "'NO_GO_RESOURCE_GATE'",
                      'pre_capture_validation.full_window_bit_exact',
                      'post_timing_validation.full_window_bit_exact',
                      'exit 0'):
            self.assertIn(token,s)

    def test_wrong_head_fail_closed_without_cuda_import(self):
        with tempfile.TemporaryDirectory(prefix='htrc-mask-g-abba-test-') as temp:
            root=Path(temp);tools=root/'tools';tools.mkdir()
            for name in ('benchmark_h3_full_mask_g_tile_abba.py',
                         'benchmark_h3_full_mask_g_numeric.py',
                         'benchmark_h3_far_masked_loads_numeric.py',
                         'analyze_h3_far_exclusion.py',
                         'profile_h3_far_exclusion.py'):
                shutil.copy2(HERE/name,tools/name)
            subprocess.run(['git','init','-q',str(root)],capture_output=True,check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=HTRC-test',
                            '-c','user.email=test@example.invalid',
                            'commit','--allow-empty','-q','-m','init'],capture_output=True,check=True)
            import json
            output=root/'result.json'
            proc=subprocess.run([sys.executable,'-B',str(tools/'benchmark_h3_full_mask_g_tile_abba.py'),
                                '--expect-head','0'*40,'--output',str(output)],
                                text=True,capture_output=True)
            r=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(proc.returncode,2)
            self.assertEqual(r['status'],'FAIL_CLOSED')
            self.assertEqual(r['speed_measurement'],'UNRUN_NUMERIC_GATE')
            self.assertEqual(r['capture_blocks'],[])
            self.assertFalse(r['completed'])
            self.assertIn('Exact HEAD mismatch',r['error'])

    def test_exact_repo_unchanged_tile_and_mask_g(self):
        tr=HERE.parent/'h3'/'htrc_flat_triton.py'
        batch=HERE.parent/'h3'/'htrc_flat_head_batch.py'
        if not tr.is_file() or not batch.is_file():
            self.skipTest('Full exact-head HTRC checkout required in Windows runner')
        a,b=tr.read_text(encoding='utf-8'),batch.read_text(encoding='utf-8')
        self.assertIn('for n0 in range(lo, hi, BN):',a)
        self.assertIn('sc = tl.where(((ex == 0) & cm)[None, :], sc, float("-inf"))',a)
        self.assertIn('def launch_prepared_tile_batch(',b)
        self.assertIn('def launch_prepared_merge_batch(',b)
        self.assertIn('FAR_PV_ACC=0,NEAR_PV_ACC=resource_near_pv_acc',b)


if __name__ == '__main__':
    unittest.main()
