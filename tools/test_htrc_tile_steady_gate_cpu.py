"""CPU-only gates for fixed 24-block control steady-state conditioning."""
import math
import statistics
import unittest
from unittest.mock import patch

from htrc_tile_steady_gate import (
    BLOCKS, SAMPLES_PER_BLOCK, REPLAYS_PER_SAMPLE, SETTLING_BLOCKS,
    MAX_LATE_SPREAD_PCT, MAX_LATE_HALVES_DRIFT_PCT,
    summarize_steady_gate, run_steady_precondition,
)


def blocks(early=0.400, late=0.408, late_last=None):
    result = []
    for i in range(BLOCKS):
        value = early if i < SETTLING_BLOCKS else (
            late if i < SETTLING_BLOCKS + 6 else late if late_last is None else late_last)
        samples = [{'wall_ms_per_replay': value} for _ in range(SAMPLES_PER_BLOCK)]
        result.append({'block_index': i + 1, 'samples': samples,
                       'median_ms_per_replay': value})
    return result


class SteadyGateCPU(unittest.TestCase):
    def test_fixed_protocol_and_stable_late_all_early_retained(self):
        self.assertEqual((BLOCKS, SAMPLES_PER_BLOCK, REPLAYS_PER_SAMPLE), (24, 5, 256))
        self.assertEqual((MAX_LATE_SPREAD_PCT, MAX_LATE_HALVES_DRIFT_PCT), (1.0, 0.5))
        d = summarize_steady_gate(blocks(early=0.38, late=0.408))
        self.assertTrue(d['admitted'])
        self.assertGreater(d['early_12_block_spread_pct'], -1)
        self.assertEqual(len(d['block_medians_ms_per_replay']), 24)
        self.assertTrue(d['no_candidate_timed_by_gate'])

    def test_late_half_drift_rejects_steady(self):
        d = summarize_steady_gate(blocks(late=0.408, late_last=0.411))
        self.assertFalse(d['admitted'])
        self.assertEqual(d['status'], 'INCONCLUSIVE_UNSTABLE_CONTROL')
        self.assertGreater(d['late_half_median_drift_pct'], 0.5)

    def test_spread_rejects_regardless_of_average_drift(self):
        items = blocks()
        items[13]['median_ms_per_replay'] = 0.416
        for sample in items[13]['samples']:
            sample['wall_ms_per_replay'] = 0.416
        d = summarize_steady_gate(items)
        self.assertFalse(d['admitted'])
        self.assertGreater(d['late_12_block_spread_pct'], 1)

    def test_incomplete_and_corrupt_raw_rejected(self):
        with self.assertRaises(ValueError):
            summarize_steady_gate(blocks()[:23])
        x = blocks()
        x[1]['samples'][0]['wall_ms_per_replay'] = math.nan
        with self.assertRaises(ValueError):
            summarize_steady_gate(x)
        x = blocks()
        x[1]['median_ms_per_replay'] = 123
        with self.assertRaises(ValueError):
            summarize_steady_gate(x)
        x = blocks()
        x[0]['samples'].pop()
        with self.assertRaises(ValueError):
            summarize_steady_gate(x)

    def test_original_graph_only_exact_replay_count(self):
        class Graph:
            def __init__(self): self.n = 0
            def replay(self): self.n += 1
        class Torch:
            class cuda:
                calls = 0
                @staticmethod
                def synchronize(device):
                    Torch.cuda.calls += 1
        g = Graph()
        ticks = iter(i * 100_000_000 for i in range(2 * BLOCKS * SAMPLES_PER_BLOCK))
        with patch('htrc_tile_steady_gate.time.perf_counter_ns', side_effect=ticks):
            result = run_steady_precondition(Torch, g, 'cuda:0')
        self.assertEqual(g.n, 24 * 5 * 256)
        self.assertEqual(Torch.cuda.calls, 24 * 5)
        self.assertTrue(result['decision']['admitted'])
        self.assertFalse(result['cuda_events_used'])


if __name__ == '__main__':
    unittest.main()
