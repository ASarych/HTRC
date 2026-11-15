"""CPU-only source and decision gates: original Tile fresh vs sentinel stability."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark_h3_tile_control_stability import (
    ARMS, REPLAYS, MAX_INTER_CAPTURE_SPREAD_PCT, _spread_pct,
    decide_control_stability, validate_request,
)


class Decisions(unittest.TestCase):
    def test_pinned_sentinels_and_threshold(self):
        self.assertEqual(ARMS, ('control',) * 4)
        self.assertEqual(REPLAYS, 16)
        self.assertEqual(MAX_INTER_CAPTURE_SPREAD_PCT, 3.0)

    def test_clean_controls_pass_without_a_speed_claim(self):
        c = [0.3832, 0.3840, 0.3836, 0.3833]
        a = [0.3829, 0.3835, 0.3836, 0.3835]
        b = [0.3832, 0.3834, 0.3832, 0.3837]
        d = decide_control_stability(c, a, b)
        self.assertEqual(d['status'], 'PASS_CONTROL_STABILITY_ONLY')
        self.assertEqual(d['classification'], 'STABLE_CONTROL_ONLY')
        self.assertTrue(d['control_stable'])
        self.assertTrue(d['sentinel_stable'])
        self.assertTrue(d['not_candidate_speed_gate'])

    def test_shared_drift_is_not_attributed_to_a_cause(self):
        c = [0.383, 0.50, 0.383, 0.50]
        a = [0.384, 0.51, 0.385, 0.51]
        b = [0.385, 0.51, 0.385, 0.51]
        d = decide_control_stability(c, a, b)
        self.assertEqual(d['status'], 'UNSTABLE_CONTROL_ONLY')
        self.assertEqual(d['classification'], 'SHARED_CONTROL_AND_SENTINEL_VARIATION')

    def test_fresh_capture_drift_with_stable_sentinel(self):
        c = [0.507, 0.384, 0.384, 0.383]
        a = [0.383, 0.384, 0.383, 0.383]
        b = [0.383, 0.383, 0.384, 0.383]
        d = decide_control_stability(c, a, b)
        self.assertEqual(d['classification'], 'FRESH_CAPTURE_VARIATION_SENTINEL_STABLE')
        self.assertFalse(d['control_stable'])
        self.assertTrue(d['sentinel_stable'])
        self.assertGreater(d['fresh_capture_spread_pct'], 30)

    def test_sentinel_drift_with_stable_fresh_controls(self):
        c = [0.383, 0.384, 0.3835, 0.3831]
        a = [0.383, 0.383, 0.50, 0.383]
        b = [0.383, 0.383, 0.383, 0.383]
        d = decide_control_stability(c, a, b)
        self.assertEqual(d['classification'], 'SENTINEL_VARIATION_FRESH_CAPTURE_STABLE')
        self.assertFalse(d['sentinel_stable'])
        self.assertTrue(d['control_stable'])

    def test_strict_invalid_sample_refusal(self):
        for bad in ([0.1], [0.0, 0.2], [float('nan'), 0.2],
                    [float('inf'), 0.2], [-0.2, 0.2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _spread_pct(bad)
        with self.assertRaises(ValueError):
            decide_control_stability([0.3] * 3, [0.3] * 4, [0.3] * 4)

    def test_pinned_request(self):
        class Args:
            rounds = 7
            warmups = 5
            replays_per_sample = 16
        validate_request(Args())
        for key, val in [('rounds',6),('rounds',13),('warmups',4),
                         ('warmups',13),('replays_per_sample',8)]:
            a = Args()
            setattr(a, key, val)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_request(a)


class SourceAndRunner(unittest.TestCase):
    def test_original_graph_only_and_timing_brackets(self):
        s = (HERE/'benchmark_h3_tile_control_stability.py').read_text('utf-8')
        ast.parse(s)
        for token in (
            "ARMS = ('control', 'control', 'control', 'control')",
            'def decide_control_stability(', 'def _measure_graph(',
            'def _nvidia_smi_snapshot(', 'def _launch_control(',
            'batch.launch_prepared_tile_batch(',
            'torch.cuda.CUDAGraph()', 'with torch.cuda.graph(sentinel):',
            'with torch.cuda.graph(fresh):',
            "block['sentinel_before'] = _measure_graph(",
            "block['fresh_control'] = _measure_graph(",
            "block['sentinel_after'] = _measure_graph(",
            "'sentinel_post_timing_validation'",
            'start.elapsed_time(stop)', 'EXCL', 'IDX', 'EXPECTED_MASKS[bi]',
            "'PASS_FIRST_REAL_CALL'", "dispatch='eager'",
            'PASS_TWO_BANK_ORIGINAL_TILE_MERGE_BIT_EXACT',
            'PASS_CONTROL_STABILITY_ONLY', 'UNSTABLE_CONTROL_ONLY',
            "'candidate_timed': False", "'candidate_timed':False",
            'CONTROL_ONLY_SENTINEL_BRACKET_EVENTS',
            "'PROHIBITED_RESEARCH_ONLY'",
        ):
            self.assertIn(token, s)
        self.assertNotIn('full_far_vb_cg_partial', s)
        self.assertNotIn('MASK_G=', s)
        self.assertNotIn('CACHE_VB=', s)
        self.assertLess(s.index('actual != args.expect_head:'), s.index('report.update(run_gpu(args))'))
        self.assertNotIn('import torch\n', s.split('def run_gpu(args):')[0])

    def test_runner_and_document(self):
        s = (HERE/'run_h3_tile_control_stability_safe.ps1').read_text('utf-8')
        for token in (
            'git clone --no-checkout --no-hardlinks',
            'git -C $tmp switch --detach $ExactHead',
            'test_h3_full_mask_g_numeric_cpu.py',
            'test_h3_tile_control_stability_cpu.py',
            'benchmark_h3_tile_control_stability.py',
            'flat_tile_control_stability.json',
            'Get-NetTCPConnection -LocalPort 8188',
            'Get-Process -Name blender',
            "'PASS_CONTROL_STABILITY_ONLY'", "'UNSTABLE_CONTROL_ONLY'",
            'candidate_timed -ne $false', 'capture_blocks.Count -ne 4',
        ):
            self.assertIn(token, s)
        doc = (HERE.parent/'research'/'H3_TILE_CONTROL_STABILITY_ADMISSION_2026-11-15.md')
        self.assertTrue(doc.is_file())
        d = doc.read_text('utf-8')
        self.assertIn('no changes to h3/',d)
        self.assertIn('control-only',d)
        self.assertIn('UNRUN',d)

    def test_wrong_head_fail_closed_before_cuda(self):
        with tempfile.TemporaryDirectory(prefix='htrc-control-stability-') as temp:
            root = Path(temp)
            tools = root/'tools'
            tools.mkdir()
            for f in (
                'benchmark_h3_tile_control_stability.py',
                'benchmark_h3_full_mask_g_numeric.py',
                'benchmark_h3_far_masked_loads_numeric.py',
                'analyze_h3_far_exclusion.py',
                'profile_h3_far_exclusion.py',
            ):
                shutil.copy2(HERE/f,tools/f)
            subprocess.run(['git','init','-q',str(root)],capture_output=True,check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=HTRC-test',
                            '-c','user.email=test@example.invalid',
                            'commit','--allow-empty','-q','-m','init'],
                           capture_output=True,check=True)
            report=root/'result.json'
            p = subprocess.run([sys.executable,'-B',
                                str(tools/'benchmark_h3_tile_control_stability.py'),
                                '--expect-head','0'*40,'--output',str(report)],
                               capture_output=True,text=True)
            data=json.loads(report.read_text('utf-8'))
            self.assertEqual(p.returncode,2)
            self.assertEqual(data['status'],'FAIL_CLOSED')
            self.assertFalse(data['completed'])
            self.assertEqual(data['capture_blocks'],[])
            self.assertFalse(data['candidate_timed'])
            self.assertIn('Exact HEAD mismatch',data['error'])

    def test_exact_original_production_source(self):
        tr = HERE.parent/'h3'/'htrc_flat_triton.py'
        b = HERE.parent/'h3'/'htrc_flat_head_batch.py'
        if not tr.is_file() or not b.is_file():
            self.skipTest('Full exact-head HTRC source required by Windows runner')
        src=tr.read_text('utf-8')
        self.assertIn('for n0 in range(lo, hi, BN):',src)
        self.assertIn('PASS: tl.constexpr = 0',src)
        self.assertIn('def launch_prepared_tile_batch(',b.read_text('utf-8'))
        self.assertIn('def launch_prepared_merge_batch(',b.read_text('utf-8'))


if __name__ == '__main__':
    unittest.main()
