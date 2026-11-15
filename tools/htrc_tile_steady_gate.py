"""Pre-registered original-only steady-state preconditioning for Tile A/B.

No candidate is timed until a fixed 24x5x256 replay series has completed.
This gate does not alter the A/B 3% spread or >=5% speed criteria.
"""
from __future__ import annotations

import math
import statistics
import time

BLOCKS = 24
SAMPLES_PER_BLOCK = 5
REPLAYS_PER_SAMPLE = 256
SETTLING_BLOCKS = 12
MAX_LATE_SPREAD_PCT = 1.0
MAX_LATE_HALVES_DRIFT_PCT = 0.5


def _spread_pct(values):
    values = [float(v) for v in values]
    if len(values) < 2 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Expected at least two finite, positive values')
    return (max(values) / min(values) - 1.0) * 100.0


def summarize_steady_gate(blocks):
    """Never discard early blocks or outliers; no adaptive/selective stopping."""
    if len(blocks) != BLOCKS:
        raise ValueError('Wrong fixed number of preconditioning blocks')
    medians = []
    for block in blocks:
        samples = block.get('samples', [])
        if len(samples) != SAMPLES_PER_BLOCK:
            raise ValueError('Incomplete preconditioning samples')
        ms = [s['wall_ms_per_replay'] for s in samples]
        _spread_pct(ms)
        calculated = statistics.median(ms)
        if abs(calculated - block['median_ms_per_replay']) > 1e-12:
            raise ValueError('Block median does not match raw samples')
        medians.append(calculated)
    early = medians[:SETTLING_BLOCKS]
    late = medians[SETTLING_BLOCKS:]
    first_late_half = statistics.median(late[:6])
    last_late_half = statistics.median(late[6:])
    drift = abs(first_late_half - last_late_half) / min(first_late_half, last_late_half) * 100
    spread = _spread_pct(late)
    admitted = spread <= MAX_LATE_SPREAD_PCT and drift <= MAX_LATE_HALVES_DRIFT_PCT
    return {
        'status': 'ADMIT_STEADY_A_B_ONLY' if admitted else 'INCONCLUSIVE_UNSTABLE_CONTROL',
        'admitted': admitted,
        'protocol': 'FIXED_ORIGINAL_CONTROL_24_BLOCKS_5_SAMPLES_256_REPLAYS',
        'block_medians_ms_per_replay': medians,
        'early_12_block_spread_pct': _spread_pct(early),
        'late_12_block_spread_pct': spread,
        'late_first_6_median_ms_per_replay': first_late_half,
        'late_last_6_median_ms_per_replay': last_late_half,
        'late_half_median_drift_pct': drift,
        'max_late_spread_pct': MAX_LATE_SPREAD_PCT,
        'max_late_halves_drift_pct': MAX_LATE_HALVES_DRIFT_PCT,
        'raw_samples_preserved': True,
        'no_candidate_timed_by_gate': True,
        'not_gpu_event_speed_admission': True,
    }


def run_steady_precondition(torch, original_graph, device):
    """Run only the unchanged original Tile Graph, before any candidate capture."""
    blocks = []
    for block_index in range(BLOCKS):
        samples = []
        for _ in range(SAMPLES_PER_BLOCK):
            start_ns = time.perf_counter_ns()
            for _ in range(REPLAYS_PER_SAMPLE):
                original_graph.replay()
            torch.cuda.synchronize(device)
            end_ns = time.perf_counter_ns()
            elapsed = (end_ns - start_ns) / (1e6 * REPLAYS_PER_SAMPLE)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Bad unprofiled control wall time')
            samples.append({'start_monotonic_ns': start_ns,
                            'end_monotonic_ns': end_ns,
                            'wall_ms_per_replay': elapsed})
        median = statistics.median(s['wall_ms_per_replay'] for s in samples)
        blocks.append({'block_index': block_index + 1, 'samples': samples,
                       'median_ms_per_replay': median})
    return {'blocks': blocks, 'decision': summarize_steady_gate(blocks),
            'replays_per_sample': REPLAYS_PER_SAMPLE,
            'samples_per_block': SAMPLES_PER_BLOCK,
            'number_of_blocks': BLOCKS,
            'control_only': True, 'cuda_events_used': False}
