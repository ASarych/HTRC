"""Original Tile32 stability: fresh control Graphs vs one persistent control sentinel.
Only original full Tile timed, no candidate or production change.
Two independent QKV source banks and original Merge must be exact.
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
    EXPECTED_MASKS, HEADS, TILES, ROWS, D, TOKENS, NB, KX, SPLITS, LPS, BN,
)
from benchmark_h3_full_mask_g_numeric import (
    _compare_bitwise, _merge,
)

ARMS = ('control', 'control', 'control', 'control')
REPLAYS = 16
MIN_REDUCTION_PCT = 5.0
MAX_INTER_CAPTURE_SPREAD_PCT = 3.0


def validate_request(args):
    if not 7 <= args.rounds <= 12 or not 5 <= args.warmups <= 12:
        raise ValueError('Requires 7..12 event samples and 5..12 warmups')
    if args.replays_per_sample != REPLAYS:
        raise ValueError('Pinned timing requires 16 replays per event sample')


def _spread_pct(values):
    if len(values) < 2 or not all(math.isfinite(float(x)) and float(x) > 0 for x in values):
        raise ValueError('At least two positive finite medians required')
    return (max(values) / min(values) - 1.0) * 100.0


def decide_control_stability(control_medians, sentinel_before, sentinel_after):
    """Control-only test: separate capture-to-capture drift from shared drift."""
    ctrl = list(map(float, control_medians))
    bef = list(map(float, sentinel_before))
    aft = list(map(float, sentinel_after))
    if len(ctrl) != len(ARMS) or len(bef) != len(ctrl) or len(aft) != len(ctrl):
        raise ValueError('Exactly four fresh and four paired sentinel observations')
    controls = _spread_pct(ctrl)
    sentinels = _spread_pct(bef + aft)
    paired = [abs(x-y)/min(x,y)*100 for x,y in zip(bef,aft)]
    stable_c = controls <= MAX_INTER_CAPTURE_SPREAD_PCT
    stable_s = (sentinels <= MAX_INTER_CAPTURE_SPREAD_PCT and
                max(paired) <= MAX_INTER_CAPTURE_SPREAD_PCT)
    status = 'PASS_CONTROL_STABILITY_ONLY' if stable_c and stable_s else 'UNSTABLE_CONTROL_ONLY'
    if stable_c and stable_s:
        classification = 'STABLE_CONTROL_ONLY'
    elif not stable_c and not stable_s:
        classification = 'SHARED_CONTROL_AND_SENTINEL_VARIATION'
    elif not stable_c:
        classification = 'FRESH_CAPTURE_VARIATION_SENTINEL_STABLE'
    else:
        classification = 'SENTINEL_VARIATION_FRESH_CAPTURE_STABLE'
    return {'status':status, 'classification':classification,
            'fresh_control_medians_ms':ctrl,
            'sentinel_before_medians_ms':bef,'sentinel_after_medians_ms':aft,
            'fresh_capture_spread_pct':controls,'sentinel_all_spread_pct':sentinels,
            'sentinel_local_pre_post_drift_pct':paired,
            'max_allowed_spread_pct':MAX_INTER_CAPTURE_SPREAD_PCT,
            'control_stable':stable_c,'sentinel_stable':stable_s,
            'not_candidate_speed_gate':True}


def _launch_control(batch, states, cfg, tr, out):
    _, compiled = batch.launch_prepared_tile_batch(states, cfg, tr, buffers=out)
    return compiled


def _measure_graph(torch, graph, rounds, replays):
    times=[]
    for _ in range(rounds):
        start=torch.cuda.Event(enable_timing=True)
        stop=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        stop.record()
        stop.synchronize()
        times.append(float(start.elapsed_time(stop))/replays)
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise RuntimeError('Invalid GPU event timing')
    med=statistics.median(times)
    return {'samples_ms_per_tile':times,'median_ms_per_tile':med,
            'event_sample_spread_pct':_spread_pct(times),
            'samples_more_than_3pct_from_median':sum(
                abs(t-med)/med*100 > MAX_INTER_CAPTURE_SPREAD_PCT for t in times)}


def _nvidia_smi_snapshot():
    """Optional pre/post entire-run diagnostic; never used to pass a gate."""
    cmd=['nvidia-smi','--id=0',
         '--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,utilization.memory',
         '--format=csv,noheader,nounits']
    try:
        proc=subprocess.run(cmd,capture_output=True,text=True,errors='replace',timeout=4)
        return {'available':proc.returncode==0,'exit_code':proc.returncode,
                'raw_csv':proc.stdout.strip()[:1100],
                'stderr_tail':proc.stderr[-400:]}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'available':False,'reason':str(exc)[:350]}


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
        with tempfile.TemporaryDirectory(prefix='htrc-control-stability-') as tmp:
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
                control_compiled = _launch_control(batch, states, cfg, tr, control)
                torch.cuda.synchronize(device)
                control_merged = _merge(torch, batch, states, cfg, tr, control, tail)
                torch.cuda.synchronize(device)
                graph_numeric = _compare_bitwise(torch, item['output'], control_merged)
                if not graph_numeric['bit_exact']:
                    raise AssertionError('Original detached Tile+Merge differs from native Graph window')
                compiled = _compiled_resources(control_compiled)
                report['banks'].append({
                    'bank': bi,
                    'selection_mask_sha256': selection['selection_mask_sha256'],
                    'original_full_tile_merge_graph_bit_exact': graph_numeric['bit_exact'],
                    'compiled': {'control': compiled},
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
            ctrl_res = report['banks'][1]['compiled']['control']
            if any(row['compiled']['control'] != ctrl_res for row in report['banks']):
                raise AssertionError('Original compiled resources drift across QKV banks')
            if (ctrl_res['registers_per_thread'] != 255 or
                ctrl_res['shared_bytes_per_cta'] != 49664 or
                ctrl_res['compiler_spills'] != 0):
                raise AssertionError('Baseline resource envelope changed')
            report['compiled'] = {'control': ctrl_res}
            report['numeric_gate'] = 'PASS_TWO_BANK_ORIGINAL_TILE_MERGE_BIT_EXACT'
            report['parity_admission'] = report['numeric_gate']
            report['environment_snapshots'] = {'before': _nvidia_smi_snapshot()}
            states = last_item['states']
            # One persistent original sentinel: capture it exactly once.
            sentinel_out = _alloc_partials(torch, device)
            if _compiled_resources(_launch_control(batch, states, cfg, tr, sentinel_out)) != ctrl_res:
                raise AssertionError('Sentinel baseline resources drift')
            sentinel = torch.cuda.CUDAGraph()
            with torch.cuda.graph(sentinel):
                _launch_control(batch, states, cfg, tr, sentinel_out)
            torch.cuda.synchronize(device)
            for _ in range(args.warmups):
                sentinel.replay()
            sentinel_initial = _verify_partials_and_merge(
                torch,batch,states,cfg,tr,last_control,last_merged,last_tail,
                sentinel_out,'sentinel_initial')
            report['sentinel_initial_parity'] = sentinel_initial
            if not sentinel_initial['full_window_bit_exact']:
                raise AssertionError('Persistent original sentinel numeric mismatch')

            # Four independently captured ORIGINAL control-only graphs.
            for index, arm in enumerate(ARMS,1):
                out = _alloc_partials(torch,device)
                if _compiled_resources(_launch_control(batch,states,cfg,tr,out)) != ctrl_res:
                    raise AssertionError('Fresh original resources drift')
                pre = _verify_partials_and_merge(
                    torch,batch,states,cfg,tr,last_control,last_merged,last_tail,
                    out,f'fresh{index}_pre')
                block = {'capture_index':index,'arm':arm,'independent_capture':True,
                         'replays_per_gpu_event':args.replays_per_sample,
                         'samples_per_capture':args.rounds,'pre_capture_validation':pre}
                if not pre['full_window_bit_exact']:
                    report['capture_blocks'].append(block)
                    raise AssertionError(f'Fresh original control {index} pre-capture mismatch')
                fresh = torch.cuda.CUDAGraph()
                with torch.cuda.graph(fresh):
                    _launch_control(batch,states,cfg,tr,out)
                torch.cuda.synchronize(device)
                for _ in range(args.warmups):
                    fresh.replay()
                block['post_warmup_validation'] = _verify_partials_and_merge(
                    torch,batch,states,cfg,tr,last_control,last_merged,last_tail,
                    out,f'fresh{index}_warmup')
                if not block['post_warmup_validation']['full_window_bit_exact']:
                    report['capture_blocks'].append(block)
                    raise AssertionError(f'Fresh original control {index} warmup mismatch')

                # Same sentinel before and after each new control measurement.
                block['sentinel_before'] = _measure_graph(
                    torch,sentinel,args.rounds,args.replays_per_sample)
                block['fresh_control'] = _measure_graph(
                    torch,fresh,args.rounds,args.replays_per_sample)
                block['sentinel_after'] = _measure_graph(
                    torch,sentinel,args.rounds,args.replays_per_sample)
                block['post_timing_validation'] = _verify_partials_and_merge(
                    torch,batch,states,cfg,tr,last_control,last_merged,last_tail,
                    out,f'fresh{index}_post_timing')
                block['sentinel_post_timing_validation'] = _verify_partials_and_merge(
                    torch,batch,states,cfg,tr,last_control,last_merged,last_tail,
                    sentinel_out,f'sentinel{index}_post_timing')
                report['capture_blocks'].append(block)
                if (not block['post_timing_validation']['full_window_bit_exact'] or
                    not block['sentinel_post_timing_validation']['full_window_bit_exact']):
                    raise AssertionError(f'Original/sentinel output changed after capture {index}')
                del fresh,out
                gc.collect()

            decision = decide_control_stability(
                [b['fresh_control']['median_ms_per_tile'] for b in report['capture_blocks']],
                [b['sentinel_before']['median_ms_per_tile'] for b in report['capture_blocks']],
                [b['sentinel_after']['median_ms_per_tile'] for b in report['capture_blocks']])
            report.update({'status':decision['status'],'completed':True,
                           'control_stability':decision,
                           'speed_measurement':'CONTROL_ONLY_SENTINEL_BRACKET_EVENTS',
                           'candidate_timed':False,'timed_full_attention':False,
                           'no_model_runtime_change':True})
            report['environment_snapshots']['after'] = _nvidia_smi_snapshot()
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
        'schema': 'htrc-tile-control-stability-v1',
        'status': 'FAIL_CLOSED', 'completed': False,
        'source_head': None, 'probe': 'ORIGINAL_TILE_FRESH_CAPTURE_VS_PERSISTENT_SENTINEL',
        'tokens': TOKENS, 'heads': HEADS, 'tiles': TILES, 'splits': SPLITS,
        'near_leaves': KX, 'far_bn': BN, 'window_query_rows': 2048,
        'capture_order': list(ARMS), 'banks': [], 'capture_blocks': [],
        'numeric_gate': 'UNRUN', 'speed_measurement': 'UNRUN_NUMERIC_GATE',
        'h3_model_or_video': 'UNRUN',
        'timing_scope': 'ORIGINAL_TILE_ONLY_SENTINEL_FRESH_NO_MERGE',
        'production_promotion': 'PROHIBITED_RESEARCH_ONLY',
        'not_a_full_operator_speedup': True, 'candidate_timed': False,
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
        'PASS_CONTROL_STABILITY_ONLY', 'UNSTABLE_CONTROL_ONLY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
