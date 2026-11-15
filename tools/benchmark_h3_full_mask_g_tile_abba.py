"""Exact full mask-G Tile-only A/B/B/A against original on RTX 5070 Ti.

Only four independent CUDA Graph captures of ONE prepared 2048-query-row
window are timed. Full Graph/eager, both QKV banks, full Tile PM/PL/PACC and
unchanged Merge must first be bit-exact. Nothing is wired into production.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools'))
from analyze_h3_far_exclusion import summarize_exclusions
from profile_h3_far_exclusion import ENV_KEYS
from benchmark_h3_far_masked_loads_numeric import (
    _alloc_partials, _assert_pinned, compare_far_partials, _compiled_resources,
    EXPECTED_MASKS, HEADS, TILES, ROWS, D, TOKENS, KX, SPLITS, BN,
)
from benchmark_h3_full_mask_g_numeric import (
    _launch_full_candidate, _compare_bitwise, _merge,
)

ARMS = ('control', 'mask_g', 'mask_g', 'control')
REPLAYS = 16
MIN_REDUCTION_PCT = 5.0
MAX_INTER_CAPTURE_SPREAD_PCT = 3.0


def validate_request(args):
    if not 7 <= args.rounds <= 12 or not 5 <= args.warmups <= 12:
        raise ValueError('Requires 7..12 event samples and 5..12 warmups')
    if args.replays_per_sample != REPLAYS:
        raise ValueError('Pinned timing requires 16 replays per event sample')


def _relative_spread_pct(values):
    if len(values) != 2 or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('Each arm needs exactly two positive finite capture medians')
    return (max(values) / min(values) - 1.0) * 100.0


def decide_abba(control_medians, candidate_medians):
    """Conservative two-capture-per-arm admission for the measured Tile only."""
    c = tuple(float(x) for x in control_medians)
    g = tuple(float(x) for x in candidate_medians)
    cs, gs = _relative_spread_pct(c), _relative_spread_pct(g)
    cm, gm = statistics.median(c), statistics.median(g)
    reduction = (1.0 - gm / cm) * 100.0
    worst_pair_reduction = (1.0 - max(g) / min(c)) * 100.0
    stable = (cs <= MAX_INTER_CAPTURE_SPREAD_PCT and
              gs <= MAX_INTER_CAPTURE_SPREAD_PCT)
    passed = stable and reduction >= MIN_REDUCTION_PCT and (
        worst_pair_reduction >= MIN_REDUCTION_PCT)
    if not stable:
        status = 'INCONCLUSIVE_TILE_ONLY'
    elif passed:
        status = 'PASS_TILE_ONLY_SPEED_GATE'
    else:
        status = 'NO_GO_TILE_ONLY'
    return {
        'status': status,
        'control_capture_medians_ms': list(c),
        'mask_g_capture_medians_ms': list(g),
        'control_median_ms': cm,
        'mask_g_median_ms': gm,
        'mask_g_time_reduction_pct': reduction,
        'mask_g_speed_ratio_control_over_candidate': cm / gm,
        'worst_cross_capture_time_reduction_pct': worst_pair_reduction,
        'control_inter_capture_spread_pct': cs,
        'mask_g_inter_capture_spread_pct': gs,
        'inter_capture_stable': stable,
        'threshold_min_time_reduction_pct': MIN_REDUCTION_PCT,
        'max_inter_capture_spread_pct': MAX_INTER_CAPTURE_SPREAD_PCT,
        'all_mask_g_captures_at_least_five_pct_faster_than_all_controls': (
            worst_pair_reduction >= MIN_REDUCTION_PCT),
        'measures_full_tile_kernel_only': True,
        'excludes_preparation_and_merge': True,
        'not_full_attention_or_model_speedup': True,
    }


def resources_admitted(original, candidate):
    """Refuse an unnoticed resource increase at the pinned full-Tile shape."""
    required = ('registers_per_thread', 'shared_bytes_per_cta', 'compiler_spills')
    if any(original.get(k) is None or candidate.get(k) is None for k in required):
        return False
    if (original['registers_per_thread'] != 255 or
        original['shared_bytes_per_cta'] != 49664 or
        original['compiler_spills'] != 0):
        return False
    return (candidate['registers_per_thread'] <= original['registers_per_thread'] and
            candidate['shared_bytes_per_cta'] <= original['shared_bytes_per_cta'] and
            candidate['compiler_spills'] == 0)


def _launch(arm, torch, batch, states, cfg, tr, research_kernel, out):
    if arm == 'control':
        _, compiled = batch.launch_prepared_tile_batch(states, cfg, tr, buffers=out)
        return compiled
    if arm == 'mask_g':
        return _launch_full_candidate(states, research_kernel, out, mask_g=True)
    if arm == 'mirror':
        return _launch_full_candidate(states, research_kernel, out, mask_g=False)
    raise ValueError('Unknown research arm; no production fallback')


def _verify_partials_and_merge(torch, batch, states, cfg, tr,
                               expected, expected_merged, tail, buffers, label):
    torch.cuda.synchronize(states['Q'].device)
    partials = compare_far_partials(torch, expected, buffers)
    merged = _merge(torch, batch, states, cfg, tr, buffers, tail)
    torch.cuda.synchronize(states['Q'].device)
    merge_parity = _compare_bitwise(torch, expected_merged, merged)
    ok = all(x['bit_exact'] for x in partials.values()) and merge_parity['bit_exact']
    return {'phase': label, 'partials': partials,
            'merge': merge_parity, 'full_window_bit_exact': ok}


def run_gpu(args):
    # torch, triton and production h3 are imported only AFTER exact HEAD gate.
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live
    from h3 import htrc_flat_model_shadow as shared
    from h3 import htrc_flat_head_batch as batch
    from htrc_full_mask_g_triton import full_mask_g_partial

    if (not torch.cuda.is_available() or
        torch.cuda.get_device_capability(0) != (12, 0) or
        'RTX 5070 Ti' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Expected NVIDIA RTX 5070 Ti SM120')
    torch.set_grad_enabled(False)
    torch.manual_seed(42026)
    device = torch.device('cuda:0')
    qkv_shape = (1, HEADS, TOKENS, D)
    banks = [tuple(torch.randn(qkv_shape, device=device, dtype=torch.bfloat16)
                   for _ in range(3)) for _ in range(2)]
    previous = {k: os.environ.get(k) for k in ENV_KEYS}
    report = {'banks': [], 'capture_blocks': [], 'compiled': {},
              'numeric_gate': 'UNRUN', 'speed_measurement': 'UNRUN_NUMERIC_GATE'}
    original_outputs = []
    try:
        with tempfile.TemporaryDirectory(prefix='htrc-mask-g-tile-abba-') as tmp:
            env = {
                'APL_HTRC_TILE32_STATS_BT': '16',
                'APL_HTRC_DIRECTFLOW_GROUP': '11',
                'APL_HTRC_DIRECTFLOW_WARMUPS': '4',
                'APL_HTRC_DIRECTFLOW_VB_MODE': 'materialized',
                'APL_HTRC_DIRECTFLOW_GROUP_POLICY': 'legacy',
                'APL_HTRC_DIRECTFLOW_REPORT': str(Path(tmp) / 'first_call.json'),
                'APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP': None,
                'APL_HTRC_GRAPH_CAPTURE_SAFE_MU': '0',
                'APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB': '768',
                'APL_HTRC_DIRECTFLOW_MIN_FREE_MIB': '256',
                'APL_HTRC_DIRECTFLOW_TILES32_ADMISSION': '0',
            }
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            flow.clear_session_cache()
            cfg = shared.flat_config('balanced', TILES)
            tr = live._triton()
            last_item = last_control = last_merged = last_tail = None
            for bi, bank in enumerate(banks):
                full = flow.graph_attention_bthd(
                    *bank, heads=HEADS, mode='balanced', tiles=TILES,
                    budget_mib=1024, reserve_mib=256, workspace=True)
                torch.cuda.synchronize(device)
                if bi == 0:
                    gate = json.loads((Path(tmp) / 'first_call.json').read_text(encoding='utf-8'))
                    if gate.get('status') != 'PASS_FIRST_REAL_CALL' or gate.get('all_groups_bit_exact') is not True:
                        raise AssertionError('Original first-call full Graph/eager numeric gate failed')
                if tuple(full.shape) != (1, TOKENS, HEADS, D):
                    raise AssertionError('Full original attention output shape drift')
                original_outputs.append(full.detach().cpu().clone())
                session = flow._SESSIONS.get((HEADS,))
                if session is None:
                    raise RuntimeError('Missing original prepared Graph session')
                item = session['graphs'].get((TILES, 2048))
                if item is None:
                    raise RuntimeError('Original full 2048-query-row Graph window missing')
                states = item['states']
                _assert_pinned(torch, states, cfg)
                selection = summarize_exclusions(
                    states['EXCL'].detach().cpu().tolist(),
                    states['IDX'].detach().cpu().tolist(),
                    splits=SPLITS, bn=BN, expected_near=KX)
                if (selection['selection_mask_sha256'] != EXPECTED_MASKS[bi] or
                    selection['current_far_dot_blocks'] != 1920 or
                    selection['fully_excluded_blocks'] != 0):
                    raise AssertionError('Source EXCL/IDX bank or original far blocks changed')
                tail = session['tail']
                if tail is None or tuple(tail[0].shape) != (HEADS, 11, D):
                    raise AssertionError('Expected original eleven exact tail tokens')
                control = _alloc_partials(torch, device)
                control_compiled = _launch('control', torch, batch, states, cfg, tr,
                                           full_mask_g_partial, control)
                torch.cuda.synchronize(device)
                control_merged = _merge(torch, batch, states, cfg, tr, control, tail)
                torch.cuda.synchronize(device)
                graph_numeric = _compare_bitwise(torch, item['output'], control_merged)
                if not graph_numeric['bit_exact']:
                    raise AssertionError('Original detached Tile+Merge differs from native Graph window')
                variants = {}
                compiled = {'control': _compiled_resources(control_compiled)}
                for arm in ('mirror', 'mask_g'):
                    out = _alloc_partials(torch, device)
                    arm_compiled = _launch(arm, torch, batch, states, cfg, tr,
                                            full_mask_g_partial, out)
                    parity = _verify_partials_and_merge(
                        torch, batch, states, cfg, tr,
                        control, control_merged, tail, out, f'bank{bi}_{arm}')
                    variants[arm] = parity
                    compiled[arm] = _compiled_resources(arm_compiled)
                    del out
                report['banks'].append({
                    'bank': bi,
                    'selection_mask_sha256': selection['selection_mask_sha256'],
                    'original_full_tile_merge_graph_bit_exact': graph_numeric['bit_exact'],
                    'variants': variants, 'compiled': compiled,
                })
                if bi == 1:
                    last_item = item
                    last_tail = tail
                    last_control = {k: v.clone() for k, v in control.items()}
                    last_merged = control_merged.clone()
                del control, control_merged, full
            if (torch.equal(original_outputs[0], original_outputs[1]) or
                report['banks'][0]['selection_mask_sha256'] ==
                report['banks'][1]['selection_mask_sha256']):
                raise AssertionError('Two original QKV banks are not distinct')
            independent = live.flat_attention_bthd(
                *banks[1], heads=HEADS, mode='balanced', tiles=TILES,
                budget_mib=1024, reserve_mib=256, workspace=True,
                query_staging='native_torch_mu', kv_staging='native_direct',
                dispatch='eager')
            if not torch.equal(independent.cpu(), original_outputs[1]):
                raise AssertionError('Independent bank B full eager/Graph bits differ')
            del independent, original_outputs
            report.update({'first_complete_graph_eager_bit_exact': True,
                           'bank_b_complete_graph_eager_bit_exact': True,
                           'two_distinct_bf16_qkv_banks': True,
                           'different_selection_masks_per_bank': True,
                           'baseline_window_graph_bit_exact': True})
            mirror_ok = all(row['variants']['mirror']['full_window_bit_exact']
                            for row in report['banks'])
            mask_ok = mirror_ok and all(row['variants']['mask_g']['full_window_bit_exact']
                                        for row in report['banks'])
            report['numeric_gate'] = ('PASS_TWO_BANK_FULL_TILE_MERGE_BIT_EXACT'
                                      if mask_ok else 'NO_GO_BENCH_NUMERIC')
            if not mask_ok:
                report.update({'status': 'FAIL_CLOSED', 'completed': False,
                               'parity_admission': 'NO_GO_BENCH_NUMERIC',
                               'error': 'Two-bank mirror or mask_g full PM/PL/PACC/Merge bit mismatch'})
                return report
            ctrl_res = report['banks'][1]['compiled']['control']
            mask_res = report['banks'][1]['compiled']['mask_g']
            for row in report['banks']:
                if (row['compiled']['control'] != ctrl_res or
                    row['compiled']['mask_g'] != mask_res):
                    raise AssertionError('Compiled resources differ across QKV banks')
            report['compiled'] = {'control': ctrl_res, 'mask_g': mask_res}
            report['resource_gate'] = {'admitted': resources_admitted(ctrl_res, mask_res),
                                      'baseline_control': ctrl_res, 'candidate_mask_g': mask_res,
                                      'no_additional_registers_or_shared_memory':
                                      (ctrl_res['registers_per_thread'] == mask_res['registers_per_thread'] and
                                       ctrl_res['shared_bytes_per_cta'] == mask_res['shared_bytes_per_cta'])}
            if not report['resource_gate']['admitted']:
                report.update({'status': 'NO_GO_RESOURCE_GATE', 'completed': True,
                               'speed_measurement': 'UNRUN_RESOURCE_GATE',
                               'speed_capture_blocks': [], 'tile_only_speed': None})
                return report
            states = last_item['states']
            # All gates above must be complete BEFORE creating the first measured Graph.
            for index, arm in enumerate(ARMS, 1):
                out = _alloc_partials(torch, device)
                compiled = _launch(arm, torch, batch, states, cfg, tr,
                                   full_mask_g_partial, out)
                if _compiled_resources(compiled) != report['compiled'][arm]:
                    raise AssertionError('Resource drift during independent capture setup')
                pre = _verify_partials_and_merge(
                    torch, batch, states, cfg, tr, last_control, last_merged,
                    last_tail, out, f'capture{index}_pre')
                block = {'capture_index': index, 'arm': arm,
                         'independent_capture': True,
                         'pre_capture_validation': pre,
                         'replays_per_gpu_event': args.replays_per_sample,
                         'samples_per_capture': args.rounds}
                if not pre['full_window_bit_exact']:
                    report['capture_blocks'].append(block)
                    report.update({'status': 'FAIL_CLOSED', 'completed': False,
                                   'parity_admission': 'NO_GO_CAPTURE_BIT_EXACT',
                                   'error': f'Block {index} before capture differs'})
                    return report
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _launch(arm, torch, batch, states, cfg, tr,
                            full_mask_g_partial, out)
                torch.cuda.synchronize(device)
                for _ in range(args.warmups):
                    graph.replay()
                after_warmup = _verify_partials_and_merge(
                    torch, batch, states, cfg, tr, last_control, last_merged,
                    last_tail, out, f'capture{index}_warmup')
                block['post_warmup_validation'] = after_warmup
                if not after_warmup['full_window_bit_exact']:
                    report['capture_blocks'].append(block)
                    report.update({'status': 'FAIL_CLOSED', 'completed': False,
                                   'parity_admission': 'NO_GO_CAPTURE_BIT_EXACT',
                                   'error': f'Block {index} after warmups differs'})
                    return report
                samples = []
                for _ in range(args.rounds):
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.replays_per_sample):
                        graph.replay()
                    stop.record()
                    stop.synchronize()
                    samples.append(float(start.elapsed_time(stop)) /
                                   args.replays_per_sample)
                block['gpu_event_ms_per_tile_samples'] = samples
                block['median_gpu_event_ms_per_tile'] = statistics.median(samples)
                after_timing = _verify_partials_and_merge(
                    torch, batch, states, cfg, tr, last_control, last_merged,
                    last_tail, out, f'capture{index}_post_timing')
                block['post_timing_validation'] = after_timing
                report['capture_blocks'].append(block)
                if not after_timing['full_window_bit_exact']:
                    report.update({'status': 'FAIL_CLOSED', 'completed': False,
                                   'parity_admission': 'NO_GO_CAPTURE_BIT_EXACT',
                                   'speed_measurement': 'NOT_ADMITTED_REPLAY_MISMATCH',
                                   'error': f'Block {index} after timing differs'})
                    return report
                del graph, out
                gc.collect()
            control_ms = [b['median_gpu_event_ms_per_tile']
                          for b in report['capture_blocks'] if b['arm'] == 'control']
            candidate_ms = [b['median_gpu_event_ms_per_tile']
                            for b in report['capture_blocks'] if b['arm'] == 'mask_g']
            decision = decide_abba(control_ms, candidate_ms)
            report.update({'status': decision['status'], 'completed': True,
                           'tile_only_speed': decision,
                           'speed_measurement': 'ONE_WINDOW_TILE_KERNEL_ABBA',
                           'parity_admission': 'PASS_TWO_BANK_FULL_TILE_MERGE_BIT_EXACT',
                           'timed_full_attention': False,
                           'no_model_runtime_change': True})
            return report
    finally:
        try:
            flow.clear_session_cache()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-head', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--replays-per-sample', type=int, default=REPLAYS)
    args = parser.parse_args()
    report = {
        'schema': 'htrc-full-mask-g-tile-abba-v1',
        'status': 'FAIL_CLOSED', 'completed': False,
        'source_head': None, 'probe': 'ORIGINAL_FULL_TILE_VS_MASK_G_TILE_ABBA',
        'tokens': TOKENS, 'heads': HEADS, 'tiles': TILES, 'splits': SPLITS,
        'near_leaves': KX, 'far_bn': BN, 'window_query_rows': 2048,
        'capture_order': list(ARMS), 'banks': [], 'capture_blocks': [],
        'numeric_gate': 'UNRUN', 'speed_measurement': 'UNRUN_NUMERIC_GATE',
        'h3_model_or_video': 'UNRUN',
        'timing_scope': 'ONE_WINDOW_TILE_KERNEL_ONLY_NO_MERGE',
        'production_promotion': 'PROHIBITED_RESEARCH_ONLY',
        'not_a_full_operator_speedup': True,
    }
    try:
        validate_request(args)
        actual = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != args.expect_head:
            raise RuntimeError(f'Exact HEAD mismatch {actual} != {args.expect_head}')
        report['source_head'] = actual
        report.update(run_gpu(args))
    except BaseException as exc:
        report.update({'status': 'FAIL_CLOSED', 'completed': False,
                       'error': f'{type(exc).__name__}: {exc}'[:1800],
                       'traceback_tail': traceback.format_exc()[-4600:]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"one_json={args.output} status={report['status']} numeric={report.get('numeric_gate')}")
    return 0 if report['status'] in (
        'PASS_TILE_ONLY_SPEED_GATE', 'NO_GO_TILE_ONLY',
        'INCONCLUSIVE_TILE_ONLY', 'NO_GO_RESOURCE_GATE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
