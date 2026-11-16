"""Opt-in one-call CUDA Graph boundary attribution. Research-only, no kernel edits.

Records external K/V refresh+quant, Q stage, captured Graph replay, and output
write-back on the EXISTING runtime. CUDA Events include host submission gaps;
this cannot separate stats/TopK/Tile/Merge INSIDE each captured Graph.
"""
from __future__ import annotations

import inspect
import math
import time
from unittest.mock import patch

STAGES = ('kv_refresh_quant', 'q_stage', 'graph_replay', 'output_write')


def _require_canonical_replay(flow):
    source = inspect.getsource(flow._replay)
    required = (
        '_refresh_kv(stage, bank, tr, kv_provider=kv_provider)',
        '_copy_q(bank[0], geom, item["q"])',
        'item["graph"].replay()',
        'output[geom.start:geom.end, :, :].copy_(',
    )
    offsets = [source.find(token) for token in required]
    if any(offset < 0 for offset in offsets) or offsets != sorted(offsets):
        raise RuntimeError('Canonical DirectFlow _replay source changed: trace refused')


def summarize_records(records, *, groups, windows_per_group, full_gpu_ms,
                      full_wall_ms, unprofiled_gpu_median_ms, output_exact):
    if not output_exact:
        raise AssertionError('Instrumented 56-head Graph output not bit-exact')
    if groups != 7 or windows_per_group != 15:
        raise ValueError('Only pinned 56-head x 15-window geometry is accepted')
    expected = groups * (1 + 3 * windows_per_group)
    if len(records) != expected:
        raise ValueError(f'Missing Graph boundary records: {len(records)} != {expected}')
    if any(row['stage'] not in STAGES or row['group_index'] not in range(groups)
           or not math.isfinite(row['gpu_event_ms']) or row['gpu_event_ms'] <= 0
           or not math.isfinite(row['cpu_enqueue_ms']) or row['cpu_enqueue_ms'] <= 0
           for row in records):
        raise ValueError('Invalid Graph stage timing record')
    for group in range(groups):
        subset = [r for r in records if r['group_index'] == group]
        for stage in STAGES:
            expected_count = 1 if stage == 'kv_refresh_quant' else windows_per_group
            if sum(r['stage'] == stage for r in subset) != expected_count:
                raise ValueError(f'Incomplete {stage} group {group}')
    if any(not math.isfinite(v) or v <= 0 for v in
           (full_gpu_ms, full_wall_ms, unprofiled_gpu_median_ms)):
        raise ValueError('Invalid full-call GPU/host timing')
    sums = {stage: sum(r['gpu_event_ms'] for r in records if r['stage'] == stage)
            for stage in STAGES}
    cpu = {stage: sum(r['cpu_enqueue_ms'] for r in records if r['stage'] == stage)
           for stage in STAGES}
    ratio = full_gpu_ms / unprofiled_gpu_median_ms
    return {
        'status': 'PASS_GRAPH_BOUNDARY_TRACE_BIT_EXACT',
        'scope': 'ONE_INSTRUMENTED_FULL_GRAPH_CALL__NOT_UNPROFILED_SPEED',
        'heads': 56, 'group_size': 8, 'groups': groups,
        'windows_per_group': windows_per_group,
        'graph_replay_count': groups * windows_per_group,
        'stage_order': list(STAGES),
        'stage_gpu_stream_intervals_ms': sums,
        'stage_cpu_enqueue_intervals_ms': cpu,
        'instrumented_full_gpu_event_ms': full_gpu_ms,
        'instrumented_full_wall_ms': full_wall_ms,
        'unprofiled_full_gpu_median_ms': unprofiled_gpu_median_ms,
        'instrumented_over_unprofiled_gpu_ratio': ratio,
        'measurement_distorted_over_20pct': ratio > 1.2,
        'raw_intervals': records,
        'full_output_bit_exact': True,
        'events_include_cpu_queue_gaps': True,
        'gpu_and_cpu_intervals_overlap': True,
        'graph_replay_includes_stats_topk_tile_merge': True,
        'not_graph_internal_kernel_profile': True,
        'not_production_speedup_evidence': True,
    }


def trace_graph_call(torch, flow, call, reference, *, unprofiled_gpu_median_ms):
    """One strictly post-parity, warmed run. Restore original helper on ALL exits."""
    _require_canonical_replay(flow)
    events = []
    group_count = 0

    def record(stage, group_index, window_index, operation):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        begin = time.perf_counter_ns()
        operation()
        elapsed_cpu = (time.perf_counter_ns() - begin) / 1e6
        stop.record()
        events.append((start, stop, {
            'stage': stage, 'group_index': group_index,
            'window_index': window_index,
            'cpu_enqueue_ms': elapsed_cpu,
        }))

    def traced_replay(stage, bank, geometries, tr, output,
                      query_provider=None, kv_provider=None):
        nonlocal group_count
        if query_provider is not None or kv_provider is not None:
            raise ValueError('Graph boundary trace forbids research Q/K/V producers')
        gi = group_count
        group_count += 1
        if gi >= 7 or len(geometries) != 15 or int(output.shape[1]) != 8:
            raise AssertionError('Unexpected Graph group/window geometry')
        record('kv_refresh_quant', gi, None,
               lambda: flow._refresh_kv(stage, bank, tr, kv_provider=None))
        for wi, geom in enumerate(geometries):
            item = stage['graphs'][flow._geometry_key(geom)]
            record('q_stage', gi, wi,
                   lambda: flow._copy_q(bank[0], geom, item['q']))
            record('graph_replay', gi, wi, item['graph'].replay)
            merged = item['output']
            nhead = int(output.shape[1])
            selected = merged.reshape(nhead, geom.padded_rows, 128)
            selected = selected[:, :geom.valid_rows, :]
            def write():
                output[geom.start:geom.end, :, :].copy_(
                    selected.permute(1, 0, 2))
            record('output_write', gi, wi, write)

    begin = time.perf_counter_ns()
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    # patch.object restores _replay even if the instrumented call fails.
    with patch.object(flow, '_replay', traced_replay):
        start_event.record()
        traced = call()
        stop_event.record()
        stop_event.synchronize()
    full_wall_ms = (time.perf_counter_ns() - begin) / 1e6
    exact = bool(torch.equal(traced, reference))
    del traced
    rows = []
    for first, last, row in events:
        elapsed = float(first.elapsed_time(last))
        rows.append({**row, 'gpu_event_ms': elapsed})
    return summarize_records(
        rows, groups=group_count, windows_per_group=15,
        full_gpu_ms=float(start_event.elapsed_time(stop_event)),
        full_wall_ms=full_wall_ms,
        unprofiled_gpu_median_ms=float(unprofiled_gpu_median_ms),
        output_exact=exact)
