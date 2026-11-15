"""Isolate startup original Tile variation: host enqueue vs CUDA Event GPU duration.

Research-only, single original persistent Graph from the existing exact two-bank gate.
CUDA Event timing is diagnostic instrumentation, NOT a candidate speed comparison.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools'))
from benchmark_h3_tile_quiet_environment import run_gpu

PHASES = ('startup_wall', 'paired_host_gpu', 'settled_wall')
BLOCKS = {'startup_wall': 12, 'paired_host_gpu': 12, 'settled_wall': 6}
SAMPLES_PER_BLOCK = 5
REPLAYS_PER_SAMPLE = 256
WARMUPS = 5
SPREAD_PCT = 3.0
STATUS = 'RECORDED_ORIGINAL_TILE_HOST_DEVICE_DIAGNOSTIC_ONLY'


def validate_request(args):
    if not args.quiet_confirmed:
        raise ValueError('Explicit quiet environment confirmation required')
    if args.warmups != WARMUPS:
        raise ValueError('Pinned warmup requires exactly five initial replays')
    if args.measurement_protocol != 'host_device':
        raise ValueError('Unexpected research-only measurement protocol')


def spread_pct(values):
    values = [float(v) for v in values]
    if len(values) < 2 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('At least two strictly positive finite samples required')
    return 100.0 * (max(values) / min(values) - 1.0)


def _time_sample(torch, graph, device, start_event=None, stop_event=None):
    instrumented = start_event is not None
    if instrumented != (stop_event is not None):
        raise ValueError('CUDA Event pair must be complete')
    # Host timestamps explicitly split enqueue and synchronization.
    t0 = time.perf_counter_ns()
    if instrumented:
        start_event.record()
    t1 = time.perf_counter_ns()
    for _ in range(REPLAYS_PER_SAMPLE):
        graph.replay()
    t2 = time.perf_counter_ns()
    if instrumented:
        stop_event.record()
        stop_event.synchronize()
        gpu_ms = float(start_event.elapsed_time(stop_event)) / REPLAYS_PER_SAMPLE
    else:
        torch.cuda.synchronize(device)
        gpu_ms = None
    t3 = time.perf_counter_ns()
    values = {
        'start_monotonic_ns': t0,
        'end_monotonic_ns': t3,
        'wall_ms_per_replay': (t3 - t0) / 1e6 / REPLAYS_PER_SAMPLE,
        'enqueue_ms_per_replay': (t2 - t1) / 1e6 / REPLAYS_PER_SAMPLE,
        'completion_wait_ms_per_replay': (t3 - t2) / 1e6 / REPLAYS_PER_SAMPLE,
        'gpu_event_ms_per_replay': gpu_ms,
        'event_instrumented': instrumented,
    }
    required = [values['wall_ms_per_replay'], values['enqueue_ms_per_replay']]
    if instrumented:
        required.append(gpu_ms)
    if not all(math.isfinite(v) and v > 0 for v in required):
        raise ValueError('Invalid host or CUDA Event timing')
    if values['completion_wait_ms_per_replay'] < 0:
        raise ValueError('Host clock regressed during synchronization')
    return values


def _phase(torch, graph, device, name):
    instrumented = name == 'paired_host_gpu'
    # Reuse the SAME events for all paired samples to avoid per-sample allocations.
    start = torch.cuda.Event(enable_timing=True) if instrumented else None
    stop = torch.cuda.Event(enable_timing=True) if instrumented else None
    blocks = []
    for index in range(BLOCKS[name]):
        samples = [_time_sample(torch, graph, device, start, stop)
                   for _ in range(SAMPLES_PER_BLOCK)]
        wall = [s['wall_ms_per_replay'] for s in samples]
        enqueue = [s['enqueue_ms_per_replay'] for s in samples]
        gpu = [s['gpu_event_ms_per_replay'] for s in samples] if instrumented else []
        blocks.append({
            'block_index': index + 1, 'samples': samples,
            'wall_median_ms_per_replay': statistics.median(wall),
            'wall_raw_spread_pct': spread_pct(wall),
            'enqueue_median_ms_per_replay': statistics.median(enqueue),
            'gpu_event_median_ms_per_replay': statistics.median(gpu) if instrumented else None,
            'gpu_event_raw_spread_pct': spread_pct(gpu) if instrumented else None,
        })
    return {'name': name, 'blocks': blocks, 'instrumented': instrumented,
            'replays_per_sample': REPLAYS_PER_SAMPLE,
            'samples_per_block': SAMPLES_PER_BLOCK}


def summarize_protocol(phases):
    if set(phases) != set(PHASES):
        raise ValueError('Incomplete or unexpected diagnostic phases')
    summary = {}
    for name in PHASES:
        phase = phases[name]
        blocks = phase['blocks']
        if len(blocks) != BLOCKS[name]:
            raise ValueError('Missing diagnostic block')
        if any(len(b['samples']) != SAMPLES_PER_BLOCK for b in blocks):
            raise ValueError('Missing raw samples')
        wall = [b['wall_median_ms_per_replay'] for b in blocks]
        enqueue = [b['enqueue_median_ms_per_replay'] for b in blocks]
        paired = name == 'paired_host_gpu'
        gpu = [b['gpu_event_median_ms_per_replay'] for b in blocks] if paired else None
        if any(b['instrumented'] != paired for b in
               (sample for block in blocks for sample in block['samples'])):
            raise ValueError('Instrumented sample mismatch')
        summary[name] = {
            'wall_block_medians_ms_per_replay': wall,
            'wall_block_median_spread_pct': spread_pct(wall),
            'wall_median_ms_per_replay': statistics.median(wall),
            'enqueue_block_median_spread_pct': spread_pct(enqueue),
            'enqueue_median_ms_per_replay': statistics.median(enqueue),
            'gpu_event_block_medians_ms_per_replay': gpu,
            'gpu_event_block_median_spread_pct': spread_pct(gpu) if paired else None,
            'gpu_event_median_ms_per_replay': statistics.median(gpu) if paired else None,
        }
    startup = summary['startup_wall']
    # These first/last-six cohorts are pre-registered, not a filtered speed claim.
    startup['first_six_block_spread_pct'] = spread_pct(
        startup['wall_block_medians_ms_per_replay'][:6])
    startup['last_six_block_spread_pct'] = spread_pct(
        startup['wall_block_medians_ms_per_replay'][6:])
    startup['first_six_median_ms'] = statistics.median(
        startup['wall_block_medians_ms_per_replay'][:6])
    startup['last_six_median_ms'] = statistics.median(
        startup['wall_block_medians_ms_per_replay'][6:])
    steady = summary['settled_wall']['wall_block_median_spread_pct']
    initial = startup['wall_block_median_spread_pct']
    if steady > SPREAD_PCT:
        classification = 'VARIATION_PERSISTS_IN_FINAL_UNPROFILED_PHASE'
    elif initial > SPREAD_PCT:
        classification = 'INITIAL_VARIATION_FINAL_UNPROFILED_STABLE'
    else:
        classification = 'INITIAL_AND_FINAL_UNPROFILED_WITHIN_THRESHOLD'
    return {
        'status': STATUS,
        'classification': classification,
        'phases': summary,
        'original_wall_initial_spread_pct': initial,
        'original_wall_final_spread_pct': steady,
        'paired_host_wall_spread_pct':
            summary['paired_host_gpu']['wall_block_median_spread_pct'],
        'paired_gpu_event_spread_pct':
            summary['paired_host_gpu']['gpu_event_block_median_spread_pct'],
        'max_descriptive_spread_pct': SPREAD_PCT,
        'candidate_timed': False, 'not_candidate_speed_gate': True,
        'no_proven_root_cause': True, 'not_full_attention_timing': True,
    }


def collect_control_probe(torch, graph, device, report, verify, snapshot, clients):
    """Called by the baseline run_gpu only after ALL original numerical/resource gates."""
    report['process_identity'] = {
        'current_pid': os.getpid(), 'sys_executable': sys.executable,
        'compute_clients_including_this_process': True,
    }
    report['quiet_declaration'] = 'OPERATOR_CONFIRMED_NOT_INDEPENDENTLY_PROVEN'
    report['compute_clients'] = {'before': clients()}
    phases = {}
    for name in PHASES:
        phase = _phase(torch, graph, device, name)
        phase['post_phase_parity'] = verify(name)
        phases[name] = phase
        report['phase_blocks'][name] = phase
        if not phase['post_phase_parity']['full_window_bit_exact']:
            raise AssertionError('Original Tile/Merge raw-bit drift after ' + name)
    report['compute_clients']['after'] = clients()
    report['environment_snapshots']['after'] = snapshot()
    summary = summarize_protocol(phases)
    report.update({
        'status': STATUS, 'completed': True,
        'host_device_diagnostic': summary,
        'speed_measurement': 'ORIGINAL_ONLY_HOST_WALL_AND_INSTRUMENTED_GPU_EVENT',
        'gpu_event_stability': 'DESCRIPTIVE_ONLY_NOT_PREVIOUS_ADMISSION',
        'timed_full_attention': False, 'candidate_timed': False,
        'no_model_runtime_change': True,
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-head', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quiet-confirmed', action='store_true')
    args = parser.parse_args()
    args.warmups = WARMUPS
    args.measurement_protocol = 'host_device'
    result = {
        'schema': 'htrc-tile-host-device-v1',
        'status': 'FAIL_CLOSED', 'completed': False, 'source_head': None,
        'probe': 'PERSISTENT_ORIGINAL_TILE_STARTUP_WALL_PAIRED_HOST_GPU_SETTLED_WALL',
        'phase_order': list(PHASES), 'phase_blocks': {},
        'numeric_gate': 'UNRUN', 'h3_model_or_video': 'UNRUN',
        'production_promotion': 'PROHIBITED_RESEARCH_ONLY',
        'candidate_timed': False, 'not_a_full_operator_speedup': True,
        'timing_scope': 'ONE_ORIGINAL_2048_ROW_TILE_ONLY_NO_MERGE_TIMED',
    }
    try:
        validate_request(args)
        actual = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != args.expect_head:
            raise RuntimeError('Exact HEAD mismatch ' + actual + ' != ' + args.expect_head)
        result['source_head'] = actual
        result.update(run_gpu(args))
    except BaseException as exc:
        result.update({
            'status': 'FAIL_CLOSED', 'completed': False,
            'error': (type(exc).__name__ + ': ' + str(exc))[:1600],
            'traceback_tail': traceback.format_exc()[-3500:],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('one_json=' + str(args.output) + ' status=' + result['status'] +
          ' numeric=' + result.get('numeric_gate', 'UNKNOWN'))
    return 0 if result['status'] == STATUS else 2


if __name__ == '__main__':
    raise SystemExit(main())
