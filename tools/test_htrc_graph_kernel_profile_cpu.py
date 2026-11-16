"""CPU-only tests for conservative CUDA Graph kernel activity aggregation."""
from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from htrc_graph_kernel_profile import (
    classify_gpu_activity, summarize_gpu_activities,
    profile_one_verified_graph_call,
)


def ev(name, us, *, gpu=True):
    return SimpleNamespace(name=name, self_device_time_total=us,
                           device_type=SimpleNamespace(name='CUDA' if gpu else 'CPU'))


def summary(events, **kw):
    return summarize_gpu_activities(events, profiled_gpu_event_ms=70,
                                    profiled_wall_ms=77,
                                    unprofiled_gpu_median_ms=66,
                                    output_exact=True, **kw)


class GraphKernelProfileCPU(unittest.TestCase):
    def test_source_parse(self):
        path = Path(__file__).with_name('htrc_graph_kernel_profile.py')
        ast.parse(path.read_text('utf-8'))

    def test_classification_avoids_generic_kernel_guess(self):
        self.assertEqual(classify_gpu_activity('void _stats_kernel<int>(...)'), 'leaf_stats')
        self.assertEqual(classify_gpu_activity('_rowmax_kernel'), 'rowmax_selection')
        self.assertEqual(classify_gpu_activity('void some_topk_kernel()'), 'topk_named')
        self.assertEqual(classify_gpu_activity('_tile_kernel<...>'), 'tile')
        self.assertEqual(classify_gpu_activity('_merge_tail_kernel'), 'merge')
        self.assertEqual(classify_gpu_activity('Memcpy DtoD async'), 'device_copy')
        self.assertEqual(classify_gpu_activity('vectorized_elementwise_kernel'), 'unclassified')
        self.assertEqual(classify_gpu_activity('compute_kernel'), 'unclassified')

    def test_cuda_only_counts_no_cpu_double_count(self):
        data = [ev('_tile_kernel', 500), ev('_tile_kernel', 250),
                ev('_stats_kernel', 100), ev('cudaGraphLaunch', 10000, gpu=False),
                ev('copy generic', 50)]
        p = summary(data)
        self.assertEqual(p['cuda_activity_count'], 4)
        self.assertAlmostEqual(p['gpu_device_activity_total_ms'], 0.9)
        self.assertAlmostEqual(p['category_gpu_activity_ms']['tile']['gpu_device_total_ms'], .75)
        self.assertAlmostEqual(p['category_gpu_activity_ms']['leaf_stats']['gpu_device_total_ms'], .1)
        self.assertAlmostEqual(p['category_gpu_activity_ms']['unclassified']['gpu_device_total_ms'], .05)
        self.assertTrue(p['all_cpu_events_excluded'])
        self.assertTrue(p['not_production_speedup_evidence'])

    def test_gpu_typed_enclosing_record_function_does_not_double_count(self):
        # Captured on the actual RTX profiler: the HTRC range was incorrectly
        # represented as a CUDA-typed 67.47-ms event enclosing other kernels.
        data = [ev('HTRC_56HEAD_FULL_GRAPH_KERNEL_PROFILE', 67469.985),
                ev('_tile_kernel', 41456.084),
                ev('_stats_kernel', 15082.391),
                ev('generic_aten_kernel', 2420.234)]
        p = summary(data)
        self.assertEqual(p['cuda_activity_count'], 3)
        self.assertAlmostEqual(p['gpu_device_activity_total_ms'], 58.958709, places=6)
        self.assertAlmostEqual(
            p['excluded_non_kernel_profile_annotation']['duration_ms_not_added_to_kernel_total'],
            67.469985, places=6)
        self.assertEqual(p['excluded_non_kernel_profile_annotation']['count'], 1)
        self.assertTrue(p['non_kernel_profile_annotation_excluded_from_device_total'])
        self.assertNotIn('HTRC_56HEAD_FULL_GRAPH_KERNEL_PROFILE',
                         [row['name'] for row in p['all_gpu_activity_names_by_duration']])

    def test_only_parent_marker_has_no_gpu_kernel_evidence(self):
        with self.assertRaisesRegex(RuntimeError, 'no positive CUDA'):
            summary([ev('HTRC_56HEAD_FULL_GRAPH_KERNEL_PROFILE', 67469.985)])

    def test_no_device_activities_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'no positive CUDA'):
            summary([ev('cudaGraphLaunch', 2500, gpu=False)])

    def test_missing_and_invalid_duration_does_not_invent_gpu_time(self):
        good = ev('_tile_kernel', 10)
        invalid = ev('_stats_kernel', float('nan'))
        zero = ev('_merge_kernel', 0)
        p = summary([good, invalid, zero])
        self.assertEqual(p['cuda_activity_count'], 1)
        self.assertEqual(p['cuda_activities_missing_positive_duration'], 2)
        self.assertEqual(p['category_gpu_activity_ms']['tile']['count'], 1)

    def test_full_output_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            summarize_gpu_activities([ev('_tile_kernel', 300)],
                                     profiled_gpu_event_ms=66,
                                     profiled_wall_ms=70,
                                     unprofiled_gpu_median_ms=66,
                                     output_exact=False)

    def test_profiler_overhead_flag_preserves_raw_ratio(self):
        x = summarize_gpu_activities([ev('_tile_kernel', 23000)],
                                    profiled_gpu_event_ms=101,
                                    profiled_wall_ms=105,
                                    unprofiled_gpu_median_ms=66,
                                    output_exact=True)
        self.assertTrue(x['profiled_execution_more_than_20pct_slowdown'])
        self.assertAlmostEqual(x['profiled_over_unprofiled_gpu_ratio'],101/66)

    def test_negative_reference_rejected_before_cuda(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaises(ValueError):
            profile_one_verified_graph_call(torch, None, None,
                                            unprofiled_gpu_median_ms=-1)

    def test_windows_runner_passes_profile_flag_as_one_argument(self):
        runner = Path(__file__).with_name('run_h3_56head_stage_attribution_safe.ps1')
        if not runner.exists():
            self.skipTest('Windows runner requires full HTRC checkout')
        p = runner.read_text('utf-8')
        self.assertIn('[switch]$GraphKernelProfile', p)
        self.assertIn('flat_56head_graph_kernel_profile.json', p)
        self.assertIn('test_htrc_graph_kernel_profile_cpu.py', p)
        self.assertIn('if ($GraphStageTrace -and $GraphKernelProfile)', p)
        self.assertNotIn('@traceArgs', p)
        self.assertNotIn('@kernelArgs', p)
        lines = [line.strip() for line in p.splitlines()]
        full = ('& $py -B $benchmark --expect-head $ExactHead --output $output '
                '--quiet-confirmed --rounds 7 --warmups 5 --graph-kernel-profile')
        self.assertEqual(lines.count(full), 1)
        self.assertEqual(lines[lines.index(full)-1], 'if ($GraphKernelProfile) {')
        self.assertIn('PASS_GRAPH_KERNEL_PROFILE_DIAGNOSTIC_ONLY', p)
        self.assertIn('profile.full_output_bit_exact -ne $true', p)

    def test_benchmark_is_post_parity_opt_in(self):
        target = Path(__file__).with_name('benchmark_h3_56head_stage_attribution.py')
        if not target.exists():
            self.skipTest('Benchmark contract requires full HTRC checkout')
        s = target.read_text('utf-8')
        ast.parse(s)
        self.assertIn("ap.add_argument('--graph-kernel-profile', action='store_true'", s)
        self.assertIn("'graph_kernel_profile_requested': bool(args.graph_kernel_profile)", s)
        self.assertIn("'PASS_GRAPH_KERNEL_PROFILE_DIAGNOSTIC_ONLY'", s)
        self.assertLess(s.index('actual != args.expect_head:'),
                        s.index('result = run_gpu(args, result)'))
        self.assertLess(s.index("report['numeric_gate'] = 'PASS_TWO_BANK_FULL_OUTPUT_GRAPH_EAGER_BIT_EXACT'"),
                        s.index("report['graph_kernel_profile'] = profile_one_verified_graph_call("))
        self.assertLess(s.index("report['warmed_graph'] = verify_graph_timing(samples)"),
                        s.index("report['graph_kernel_profile'] = profile_one_verified_graph_call("))
        self.assertIn('from htrc_graph_kernel_profile import profile_one_verified_graph_call', s)
        self.assertNotIn('import torch\n', s.split('def run_gpu(args, report):')[0])



if __name__ == '__main__':
    unittest.main()
