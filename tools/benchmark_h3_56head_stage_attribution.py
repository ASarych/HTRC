"""Two-bank full-output HTRC Graph parity, warmed timing, eager stage attribution.

Research-only, synthetic DISTINCT 56-head BF16 input, balanced/512, 15,275 tokens.
The existing eager observer is NOT a profiling breakdown of CUDA Graph internals.
One atomic JSON; no production dispatch, kernels or model weights are modified.
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
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'htrc-56head-graph-eager-stage-attribution-v1'
TOKENS, HEADS, TILES, DIM = 15275, 56, 16, 128
GROUP_CAP, ROUNDS, WARMUPS = 8, 7, 5
STAGES = ('kv_workspace_setup', 'query_stats_selection', 'tile',
          'merge_tail', 'output_copy')
OUTPUT = Path(r'D:\ComfyOutput\H3\math_probe\flat_56head_stage_attribution.json')
ENV = {
    'APL_HTRC_DIRECTFLOW_GROUP_POLICY': 'reuse_aware',
    'APL_HTRC_DIRECTFLOW_GROUP': str(GROUP_CAP),
    'APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB': '768',
    'APL_HTRC_DIRECTFLOW_VB_MODE': 'materialized',
    'APL_HTRC_GRAPH_CAPTURE_SAFE_MU': '0',
}


def finite_positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def validate_request(args):
    if args.rounds != ROUNDS or args.warmups != WARMUPS:
        raise ValueError('Pinned 56-head attribution requires 7 rounds, 5 warmups')
    if not args.quiet_confirmed:
        raise ValueError('Explicit quiet-GPU confirmation required')
    if len(args.expect_head) != 40 or any(c not in '0123456789abcdef' for c in args.expect_head):
        raise ValueError('Expected lowercase 40-hex exact HEAD')


def verify_first_call(gate, bank_index):
    if not (gate.get('status') == 'PASS_FIRST_REAL_CALL'
            and gate.get('completed') is True
            and gate.get('all_groups_bit_exact') is True
            and gate.get('strict_parity_required') is True
            and gate.get('eager_first_call_diagnostic') == 'EXECUTED'
            and gate.get('heads') == HEADS
            and gate.get('tokens') == TOKENS
            and gate.get('mode') == 'balanced'
            and gate.get('vb_mode') == 'materialized'
            and gate.get('group_policy') == 'reuse_aware'
            and gate.get('head_group_cap') == GROUP_CAP
            and gate.get('window_tiles') == TILES
            and gate.get('fixed_exact_leaves') == 512):
        raise AssertionError(f'Full-output original first-call gate bank{bank_index} invalid')
    groups = gate.get('head_groups')
    if not isinstance(groups, list) or not groups:
        raise AssertionError('Missing strict per-group Graph/eager checks')
    ranges = [g.get('head_range') for g in groups]
    expected = [[i, min(i + GROUP_CAP, HEADS)] for i in range(0, HEADS, GROUP_CAP)]
    if ranges != expected or any(g.get('exact_flat_parity') is not True for g in groups):
        raise AssertionError('Not all 56 distinct heads passed full exact Graph/eager parity')
    return {'status': 'PASS_FULL_OUTPUT_BIT_EXACT', 'bank_index': bank_index,
            'source': 'synthetic_distinct_native_bf16_not_model_weights',
            'head_ranges': ranges, 'group_size': GROUP_CAP,
            'full_output_exact': True,
            'first_call_extra_reserved_mib': gate.get('extra_reserved_mib'),
            'graph_session_sizes': gate.get('graph_cache_group_sizes')}


def verify_observer(events):
    if len(events) != 1:
        raise AssertionError('Expected exactly one complete eager observer callback')
    name, row = events[0]
    if name != 'htrc_flat_gpu_dispatch' or row.get('heads') != HEADS or row.get('tokens') != TOKENS:
        raise AssertionError('Wrong existing eager observer result')
    gpu = row.get('profile_gpu_stream_elapsed_ms')
    cpu = row.get('profile_cpu_enqueue_ms')
    if not isinstance(gpu, dict) or not isinstance(cpu, dict):
        raise AssertionError('Missing real eager observer stage timings')
    if set(gpu) != set(STAGES) or set(cpu) != set(STAGES):
        raise AssertionError('Unexpected or missing eager stage names')
    if any(not finite_positive(gpu[k]) or not finite_positive(cpu[k]) for k in STAGES):
        raise AssertionError('Non-finite or missing eager stage measurement')
    if row.get('query_staging') != 'native_torch_mu' or row.get('kv_staging') != 'native_direct':
        raise AssertionError('Unexpected eager staging path')
    if row.get('exact_leaves') != 512:
        raise AssertionError('Unexpected exact-leaf budget')
    return {'head_group_capacity': row.get('group_capacity'),
            'query_windows': row.get('query_windows'),
            'stage_gpu_stream_intervals_ms': {k: float(gpu[k]) for k in STAGES},
            'stage_cpu_enqueue_intervals_ms': {k: float(cpu[k]) for k in STAGES},
            'scope': 'EAGER_ONLY__NOT_GRAPH_KERNEL_TIMING',
            'gpu_event_intervals_include_stream_gaps': True,
            'gpu_and_cpu_intervals_overlap_not_additive': True,
            'source_observer': 'h3.htrc_flat_live_backend.flat_attention_bthd'}


def verify_graph_timing(samples):
    if len(samples) != ROUNDS:
        raise AssertionError('Incomplete warmed full-Graph samples')
    for row in samples:
        if not finite_positive(row.get('gpu_ms')) or not finite_positive(row.get('sync_wall_ms')):
            raise AssertionError('Invalid full-Graph event/wall sample')
    return {'samples': samples,
            'gpu_event_median_ms': statistics.median(x['gpu_ms'] for x in samples),
            'wall_median_ms': statistics.median(x['sync_wall_ms'] for x in samples),
            'gpu_event_block_spread_pct': 100 * (
                max(x['gpu_ms'] for x in samples) / min(x['gpu_ms'] for x in samples) - 1),
            'scope': 'COMPLETE_56HEAD_GRAPH_QKV_REFRESH_QUANT_ALL_WINDOWS_TILE_MERGE_OUTPUT',
            'cuda_event_includes_cpu_enqueue_gaps': True}


def _write_one(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run_gpu(args, report):
    import torch  # exact HEAD is checked before this import
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError('Expected RTX 5070 Ti compute capability SM120')
    if 'RTX 5070 Ti' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected RTX 5070 Ti, not 5090/other GPU')
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as eager
    saved = {key: os.environ.get(key) for key in (*ENV, 'APL_HTRC_DIRECTFLOW_REPORT',
                                               'APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP')}
    device = torch.device('cuda:0')
    bank_b = None
    try:
        os.environ.update(ENV)
        os.environ.pop('APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP', None)
        with torch.inference_mode(), tempfile.TemporaryDirectory(prefix='htrc-56-stage-') as temp:
            for bank_index in range(2):
                flow.clear_session_cache()
                torch.cuda.empty_cache()
                torch.manual_seed(42026 + bank_index)
                bank = tuple(torch.randn((1, HEADS, TOKENS, DIM), device=device,
                                         dtype=torch.bfloat16) for _ in range(3))
                gate_path = Path(temp) / f'bank_{bank_index}_first_call.json'
                os.environ['APL_HTRC_DIRECTFLOW_REPORT'] = str(gate_path)
                first_out = flow.graph_attention_bthd(
                    *bank, heads=HEADS, mode='balanced', tiles=TILES,
                    budget_mib=1024, reserve_mib=256, workspace=True,
                    compare_only=False)
                torch.cuda.synchronize(device)
                gate = json.loads(gate_path.read_text('utf-8'))
                report['strict_bank_checks'].append(verify_first_call(gate, bank_index))
                # Distinct independently seeded source banks, not repeated 4-head proxy.
                tiny = bank[0].flatten()[:16].contiguous().cpu().view(torch.int16)
                report['strict_bank_checks'][-1]['q_source_first16_raw_bits'] = tiny.tolist()
                del first_out
                if bank_index == 1:
                    bank_b = bank
                else:
                    del bank
            if (report['strict_bank_checks'][0]['q_source_first16_raw_bits'] ==
                    report['strict_bank_checks'][1]['q_source_first16_raw_bits']):
                raise AssertionError('Two BF16 QKV source banks appear identical')
            report['numeric_gate'] = 'PASS_TWO_BANK_FULL_OUTPUT_GRAPH_EAGER_BIT_EXACT'
            def graph_call():
                return flow.graph_attention_bthd(
                    *bank_b, heads=HEADS, mode='balanced', tiles=TILES,
                    budget_mib=1024, reserve_mib=256, workspace=True,
                    compare_only=False)
            for _ in range(args.warmups):
                sample_out = graph_call()
                del sample_out
            torch.cuda.synchronize(device)
            report['graph_allocator_before'] = {
                'allocated_mib': torch.cuda.memory_allocated(device) / 2**20,
                'reserved_mib': torch.cuda.memory_reserved(device) / 2**20}
            torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(args.rounds):
                start, stop = (torch.cuda.Event(enable_timing=True),
                               torch.cuda.Event(enable_timing=True))
                start.record()
                t0 = time.perf_counter()
                output = graph_call()
                stop.record()
                stop.synchronize()
                samples.append({'gpu_ms': start.elapsed_time(stop),
                                'sync_wall_ms': (time.perf_counter() - t0) * 1000})
                del output
            report['warmed_graph'] = verify_graph_timing(samples)
            report['graph_allocator_peak'] = {
                'allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
                'reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
                'caveat': 'shared process and graph cache; NOT isolated graph pool size'}
            # Existing eager observer: no duplicate stats/quant/Tile/Merge code.
            # This is distinct from graph timing; profile intervals never treated as graph internals.
            graph_out = graph_call()
            torch.cuda.synchronize(device)
            observed = []
            def stage_observer(event, payload):
                observed.append((event, payload))
            eager_out = eager.flat_attention_bthd(
                *bank_b, heads=HEADS, mode='balanced', tiles=TILES,
                budget_mib=1024, reserve_mib=256, workspace=True,
                observer=stage_observer, query_staging='native_torch_mu',
                kv_staging='native_direct', dispatch='eager')
            torch.cuda.synchronize(device)
            report['eager_stage_attribution'] = verify_observer(observed)
            report['independent_full_output_bit_exact'] = bool(torch.equal(graph_out, eager_out))
            if not report['independent_full_output_bit_exact']:
                raise AssertionError('Independent final 56-head full Graph/Eager output differed')
            report['status'] = 'PASS_ATTRIBUTION_ONLY'
            report['completed'] = True
    finally:
        try:
            flow.clear_session_cache()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expect-head', required=True)
    ap.add_argument('--output', type=Path, default=OUTPUT)
    ap.add_argument('--quiet-confirmed', action='store_true')
    ap.add_argument('--rounds', type=int, default=ROUNDS)
    ap.add_argument('--warmups', type=int, default=WARMUPS)
    args = ap.parse_args()
    result = {'schema': SCHEMA, 'status': 'FAIL_CLOSED', 'completed': False,
              'source_head': None, 'input_kind': 'SYNTHETIC_DISTINCT_BF16_NO_MODEL',
              'tokens': TOKENS, 'heads': HEADS, 'tiles': TILES,
              'mode': 'balanced', 'exact_leaves': 512,
              'strict_bank_checks': [], 'numeric_gate': 'UNRUN',
              'warmed_graph': None, 'eager_stage_attribution': None,
              'h3_model_or_video': 'UNRUN', 'production_promotion': 'PROHIBITED_RESEARCH_ONLY',
              'not_a_graph_internal_stage_profile': True,
              'no_new_kernel_or_dispatch_change': True}
    try:
        validate_request(args)
        actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != args.expect_head:
            raise RuntimeError(f'Exact HEAD mismatch {actual} != {args.expect_head}')
        result['source_head'] = actual
        result = run_gpu(args, result)
    except BaseException as exc:
        result.update(status='FAIL_CLOSED', completed=False,
                      error=f'{type(exc).__name__}: {exc}'[:2000],
                      traceback_tail=traceback.format_exc()[-5000:])
    _write_one(args.output, result)
    print(f'one_json={args.output} status={result["status"]} numeric={result["numeric_gate"]}')
    return 0 if result['status'] == 'PASS_ATTRIBUTION_ONLY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
