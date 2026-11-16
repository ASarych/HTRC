"""One-call PyTorch CUDA activity profile for verified full 56-head HTRC Graph.

Research-only. No CUDA counters or changes to graph capture or kernel math.
GPU activity names are heuristics; unknown kernels remain unclassified. GPU
kernel durations are NOT percentages of elapsed full-Graph wall time.
"""
from __future__ import annotations

from collections import defaultdict
import math
import statistics
import time

CATEGORIES = (
    'leaf_stats', 'rowmax_selection', 'topk_named',
    'tile', 'merge', 'device_copy', 'unclassified',
)


def classify_gpu_activity(name):
    """Conservative source-name matches; do NOT guess generic ATen kernels."""
    label = name.lower()
    if '_merge_tail_kernel' in label or '_merge_kernel' in label:
        return 'merge'
    if '_tile_kernel' in label:
        return 'tile'
    if '_stats_kernel' in label:
        return 'leaf_stats'
    if '_rowmax_kernel' in label:
        return 'rowmax_selection'
    if 'topk' in label or 'top_k' in label:
        return 'topk_named'
    if 'memcpy' in label or 'memset' in label or 'copy device-to-device' in label:
        return 'device_copy'
    return 'unclassified'


def _get_gpu_us(event):
    # Torch FunctionEvent's .self_device_time_total denotes microseconds.
    # CUDA FunctionEvents can differ by torch version; fall back to its
    # inclusive device time only when the exclusive value is unavailable.
    value = getattr(event, 'self_device_time_total', None)
    if value is None:
        value = getattr(event, 'device_time_total', None)
    if value is None:
        return None
    try:
        us = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return us if math.isfinite(us) and us > 0 else None


def summarize_gpu_activities(events, *, profiled_gpu_event_ms,
                             profiled_wall_ms, unprofiled_gpu_median_ms,
                             output_exact):
    """Use only actual CUDA-device FunctionEvents, never CPU launch timings."""
    if not output_exact:
        raise AssertionError('Profiled full 56-head output is not bit-exact')
    if any(not math.isfinite(v) or v <= 0 for v in
           (profiled_gpu_event_ms, profiled_wall_ms,
            unprofiled_gpu_median_ms)):
        raise ValueError('Invalid profiled/unprofiled full-call time')

    by_name = defaultdict(lambda: {'count': 0, 'total_us': 0.0,
                                   'min_us': float('inf'), 'max_us': 0.0})
    count = 0
    nonpositive = 0
    for event in events:
        device_type = getattr(event, 'device_type', None)
        if getattr(device_type, 'name', None) != 'CUDA':
            continue
        name = str(getattr(event, 'name', '')).strip()
        duration = _get_gpu_us(event)
        if not name or duration is None:
            nonpositive += 1
            continue
        record = by_name[name]
        record['count'] += 1
        record['total_us'] += duration
        record['min_us'] = min(record['min_us'], duration)
        record['max_us'] = max(record['max_us'], duration)
        count += 1
    if not count:
        raise RuntimeError(
            'Profiler recorded no positive CUDA kernel/device activities; '
            'no internal Graph attribution can be claimed')

    category = {name: {'count': 0, 'gpu_device_total_ms': 0.0}
                for name in CATEGORIES}
    details = []
    for name, record in by_name.items():
        cls = classify_gpu_activity(name)
        duration_ms = record['total_us'] / 1000
        category[cls]['count'] += record['count']
        category[cls]['gpu_device_total_ms'] += duration_ms
        details.append({
            'name': name, 'category': cls,
            'count': record['count'],
            'gpu_device_total_ms': duration_ms,
            'mean_us': record['total_us'] / record['count'],
            'min_us': record['min_us'], 'max_us': record['max_us'],
        })
    details.sort(key=lambda row: (-row['gpu_device_total_ms'], row['name']))
    total_ms = sum(row['gpu_device_total_ms'] for row in details)
    for record in category.values():
        record['share_of_profiled_device_duration_pct'] = (
            record['gpu_device_total_ms'] / total_ms * 100)
    known_ms = total_ms - category['unclassified']['gpu_device_total_ms']
    return {
        'status': 'PASS_PROFILED_CUDA_ACTIVITIES_NOT_SPEED_ADMISSION',
        'scope': 'ONE_PROFILED_VERIFIED_FULL_56HEAD_GRAPH_CALL',
        'cuda_activity_count': count,
        'unique_cuda_activity_names': len(details),
        'cuda_activities_missing_positive_duration': nonpositive,
        'gpu_device_activity_total_ms': total_ms,
        'category_gpu_activity_ms': category,
        'classified_device_activity_pct': known_ms / total_ms * 100,
        'unclassified_device_activity_pct':
            category['unclassified']['gpu_device_total_ms'] / total_ms * 100,
        'all_gpu_activity_names_by_duration': details,
        'profiled_full_gpu_event_ms': profiled_gpu_event_ms,
        'profiled_full_wall_ms': profiled_wall_ms,
        'unprofiled_gpu_median_ms': unprofiled_gpu_median_ms,
        'profiled_over_unprofiled_gpu_ratio':
            profiled_gpu_event_ms / unprofiled_gpu_median_ms,
        'profiled_execution_more_than_20pct_slowdown':
            profiled_gpu_event_ms / unprofiled_gpu_median_ms > 1.20,
        'device_durations_not_full_call_wall': True,
        'all_cpu_events_excluded': True,
        'kernel_name_classification_heuristic_not_proven_code_identity': True,
        'no_assumed_internal_attribution_for_unclassified': True,
        'full_output_bit_exact': True,
        'not_production_speedup_evidence': True,
        'hardware_counters_unrun': True,
        'warning': ('CUDA Graph replay activity visibility depends on PyTorch/'
                    'CUPTI. Kernel labels are source-name heuristics; '
                    'profiler overhead and queue overlap prevent direct '
                    'speed claims from these percentages.'),
    }


def profile_one_verified_graph_call(torch, graph_call, reference, *,
                                    unprofiled_gpu_median_ms):
    """Runs one warmed Graph under PyTorch Kineto, after strict numeric gates.

    Profile only the call, not Graph construction or Torch startup. The result
    is discarded unless it agrees bit-for-bit with the verified reference.
    """
    if not math.isfinite(unprofiled_gpu_median_ms) or unprofiled_gpu_median_ms <= 0:
        raise ValueError('Missing previously measured unprofiled baseline')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for Graph kernel profile')
    from torch.profiler import ProfilerActivity, profile, record_function
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False,
                 with_stack=False) as profiler:
        with record_function('HTRC_56HEAD_FULL_GRAPH_KERNEL_PROFILE'):
            start_event.record()
            start_ns = time.perf_counter_ns()
            output = graph_call()
            stop_event.record()
            stop_event.synchronize()
            profiled_wall_ms = (time.perf_counter_ns() - start_ns) / 1e6
    full_gpu_ms = float(start_event.elapsed_time(stop_event))
    exact = bool(torch.equal(output, reference))
    del output
    return summarize_gpu_activities(
        profiler.events(),
        profiled_gpu_event_ms=full_gpu_ms,
        profiled_wall_ms=profiled_wall_ms,
        unprofiled_gpu_median_ms=float(unprofiled_gpu_median_ms),
        output_exact=exact)
