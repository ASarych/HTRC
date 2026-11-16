"""Self-contained CPU regression tests for the opt-in Graph boundary tracer."""
from __future__ import annotations
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from htrc_graph_stage_trace import (
    STAGES, _require_canonical_replay, summarize_records, trace_graph_call,
)


def fake_rows():
    result = []
    for group in range(7):
        for stage in STAGES:
            for index in range(1 if stage == 'kv_refresh_quant' else 15):
                result.append({'stage': stage, 'group_index': group,
                               'window_index': None if stage == 'kv_refresh_quant' else index,
                               'cpu_enqueue_ms': 0.1, 'gpu_event_ms': 0.2})
    return result


class TraceCPU(unittest.TestCase):
    def test_exact_shape_and_diagnostic_flags(self):
        r = summarize_records(fake_rows(), groups=7, windows_per_group=15,
                              full_gpu_ms=77, full_wall_ms=82,
                              unprofiled_gpu_median_ms=66, output_exact=True)
        self.assertEqual(r['graph_replay_count'], 105)
        self.assertEqual(len(r['raw_intervals']), 7*(1+15*3))
        self.assertEqual(r['stage_gpu_stream_intervals_ms']['graph_replay'], 21.0)
        self.assertTrue(r['not_graph_internal_kernel_profile'])
        self.assertFalse(r['measurement_distorted_over_20pct'])

    def test_distorted_only_descriptive(self):
        d = summarize_records(fake_rows(), groups=7, windows_per_group=15,
                              full_gpu_ms=100, full_wall_ms=105,
                              unprofiled_gpu_median_ms=66, output_exact=True)
        self.assertTrue(d['measurement_distorted_over_20pct'])
        self.assertTrue(d['not_production_speedup_evidence'])

    def test_numeric_failure_fails_closed(self):
        with self.assertRaises(AssertionError):
            summarize_records(fake_rows(), groups=7, windows_per_group=15,
                              full_gpu_ms=77, full_wall_ms=82,
                              unprofiled_gpu_median_ms=66, output_exact=False)

    def test_no_missing_events_or_nans(self):
        rows = fake_rows()
        rows.pop()
        with self.assertRaises(ValueError):
            summarize_records(rows, groups=7, windows_per_group=15,
                              full_gpu_ms=77, full_wall_ms=82,
                              unprofiled_gpu_median_ms=66, output_exact=True)
        rows = fake_rows()
        rows[10]['gpu_event_ms'] = float('nan')
        with self.assertRaises(ValueError):
            summarize_records(rows, groups=7, windows_per_group=15,
                              full_gpu_ms=77, full_wall_ms=82,
                              unprofiled_gpu_median_ms=66, output_exact=True)

    def test_source_fingerprint_fails_closed(self):
        flow = SimpleNamespace(_replay=lambda stage: stage)
        with self.assertRaises(RuntimeError):
            _require_canonical_replay(flow)

    def test_instrumented_replay_restores_original_on_exception(self):
        # No CUDA needed: source guard is monkeypatched for a deliberately
        # failing fake call, ensuring the global runtime helper is restored.
        original = lambda *args: None
        flow = SimpleNamespace(_replay=original)
        class Event:
            def record(self): pass
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: Event()))
        with patch('htrc_graph_stage_trace._require_canonical_replay'):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                trace_graph_call(torch, flow,
                                 lambda: (_ for _ in ()).throw(RuntimeError('synthetic failure')),
                                 reference=object(), unprofiled_gpu_median_ms=66)
        self.assertIs(flow._replay, original)

    def test_trace_source_parse(self):
        path = Path(__file__).with_name('htrc_graph_stage_trace.py')
        ast.parse(path.read_text('utf-8'))


if __name__ == '__main__':
    unittest.main()
