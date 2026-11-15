"""Diagnose unprofiled original Tile wall-time stability under declared quiet GPU conditions.
One unchanged persistent CUDA Graph, no GPU Events, no candidate or production change.
Two independent BF16 QKV banks retain the exact original numeric/resource gates.
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
import threading
import time

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

PHASES = ('quiet_before', 'monitored', 'quiet_after')
BLOCKS_PER_PHASE = 6
SAMPLES_PER_BLOCK = 5
REPLAYS = 256
POLL_MS = 200
MAX_WALL_SPREAD_PCT = 3.0


def validate_request(args):
    if not args.quiet_confirmed:
        raise ValueError('Declare a quiet GPU environment with --quiet-confirmed')
    if args.warmups != 5:
        raise ValueError('Pinned experiment requires exactly 5 initial warmup replays')

def _spread_pct(values):
    if len(values) < 2 or not all(math.isfinite(float(x)) and float(x) > 0 for x in values):
        raise ValueError('At least two positive finite medians required')
    return (max(values) / min(values) - 1.0) * 100.0


def decide_wall_stability(phases, telemetry):
    """Observed wall-time stability; NEVER GPU Event or candidate speed admission."""
    if set(phases) != set(PHASES):
        raise ValueError('Exactly three pinned wall-time phases required')
    metrics = {}
    for name in PHASES:
        blocks = phases[name]['blocks']
        if len(blocks) != BLOCKS_PER_PHASE:
            raise ValueError('Incomplete wall-time blocks')
        medians = [b['median_ms_per_replay'] for b in blocks]
        for b in blocks:
            values = b['samples_ms_per_replay']
            if len(values) != SAMPLES_PER_BLOCK or _spread_pct(values) < 0:
                raise ValueError('Incomplete or invalid wall-time samples')
        metrics[name] = {
            'block_medians_ms_per_replay': medians,
            'median_ms_per_replay': statistics.median(medians),
            'block_median_spread_pct': _spread_pct(medians),
        }
    before = metrics['quiet_before']
    after = metrics['quiet_after']
    monitored = metrics['monitored']
    quiet_drift = abs(before['median_ms_per_replay'] - after['median_ms_per_replay']) / min(
        before['median_ms_per_replay'], after['median_ms_per_replay']) * 100
    quiet_stable = (before['block_median_spread_pct'] <= MAX_WALL_SPREAD_PCT and
                    after['block_median_spread_pct'] <= MAX_WALL_SPREAD_PCT and
                    quiet_drift <= MAX_WALL_SPREAD_PCT)
    monitored_stable = monitored['block_median_spread_pct'] <= MAX_WALL_SPREAD_PCT
    if quiet_stable and monitored_stable:
        classification = 'WALL_STABLE_IN_ALL_PHASES'
    elif quiet_stable:
        classification = 'MONITORED_PHASE_VARIATION_ONLY'
    elif monitored_stable:
        classification = 'QUIET_PHASE_VARIATION_OR_INTER_PHASE_DRIFT'
    else:
        classification = 'WALL_VARIATION_WITH_AND_WITHOUT_MONITOR'
    count = len(telemetry.get('samples', []))
    return {
        'status': ('WALL_CONTROL_STABLE_DIAGNOSTIC_ONLY'
                   if quiet_stable and monitored_stable else
                   'WALL_CONTROL_UNSTABLE_DIAGNOSTIC_ONLY'),
        'classification': classification,
        'metrics': metrics,
        'quiet_pre_post_median_drift_pct': quiet_drift,
        'quiet_wall_stable': quiet_stable,
        'monitored_wall_stable': monitored_stable,
        'monitored_telemetry_samples': count,
        'max_wall_spread_pct': MAX_WALL_SPREAD_PCT,
        'not_gpu_event_stability_gate': True,
        'not_candidate_speed_gate': True,
        'no_causal_attribution': True,
    }

def _launch_control(batch, states, cfg, tr, out):
    _, compiled = batch.launch_prepared_tile_batch(states, cfg, tr, buffers=out)
    return compiled


def _wall_phase(torch, graph, device, label):
    """Unprofiled graph replay throughput; wall time includes host launch/sync overhead."""
    blocks = []
    for index in range(1, BLOCKS_PER_PHASE + 1):
        samples = []
        for _ in range(SAMPLES_PER_BLOCK):
            start_ns = time.perf_counter_ns()
            for _ in range(REPLAYS):
                graph.replay()
            torch.cuda.synchronize(device)
            end_ns = time.perf_counter_ns()
            sample = (end_ns - start_ns) / 1e6 / REPLAYS
            if not math.isfinite(sample) or sample <= 0:
                raise RuntimeError('Invalid unprofiled wall-time replay sample')
            samples.append({'start_monotonic_ns': start_ns,
                            'end_monotonic_ns': end_ns,
                            'ms_per_replay': sample})
        med = statistics.median(s['ms_per_replay'] for s in samples)
        raw = [s['ms_per_replay'] for s in samples]
        blocks.append({'block_index': index, 'samples': samples,
                       'samples_ms_per_replay': raw,
                       'median_ms_per_replay': med,
                       'sample_spread_pct': _spread_pct(raw),
                       'samples_more_than_3pct_from_median': sum(
                           abs(s - med) / med * 100 > MAX_WALL_SPREAD_PCT for s in raw)})
    return {'label': label, 'blocks': blocks,
            'replays_per_sample': REPLAYS,
            'samples_per_block': SAMPLES_PER_BLOCK,
            'blocks_per_phase': BLOCKS_PER_PHASE}


def _parse_smi_row(line):
    parts = [x.strip() for x in line.split(',')]
    if len(parts) != 8 or not parts[0] or not parts[1]:
        return None
    def numeric(index):
        try:
            value = float(parts[index])
            return value if math.isfinite(value) else None
        except ValueError:
            return None
    return {
        'driver_timestamp': parts[0], 'pstate': parts[1],
        'sm_clock_mhz': numeric(2), 'mem_clock_mhz': numeric(3),
        'temperature_c': numeric(4), 'power_w': numeric(5),
        'gpu_util_pct': numeric(6), 'memory_util_pct': numeric(7),
    }


class _NvidiaSmiSampler:
    """External process only during MONITORED; its overhead is part of the diagnostic."""
    def __init__(self):
        self.process = None
        self.thread = None
        self.samples = []
        self.invalid_lines = 0

    def _read_lines(self):
        for line in self.process.stdout:
            parsed = _parse_smi_row(line.strip())
            if parsed is None:
                self.invalid_lines += 1
                continue
            parsed['receipt_monotonic_ns'] = time.perf_counter_ns()
            self.samples.append(parsed)

    def start(self):
        cmd = ['nvidia-smi', '--id=0',
               '--query-gpu=timestamp,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,utilization.memory',
               '--format=csv,noheader,nounits', '--loop-ms=' + str(POLL_MS)]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors='replace', bufsize=1)
        self.thread = threading.Thread(target=self._read_lines, daemon=True)
        self.thread.start()

    def stop(self):
        if self.process is None:
            return {'available': False, 'samples': [], 'reason': 'not started'}
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        self.thread.join(timeout=3)
        stderr_tail = self.process.stderr.read()[-500:]
        return {'available': len(self.samples) >= 5,
                'requested_poll_ms': POLL_MS,
                'samples': list(self.samples),
                'invalid_lines': self.invalid_lines,
                'stderr_tail': stderr_tail,
                'time_basis': 'host_perf_counter_ns_on_stdout_receipt_not_driver_sample_time'}


def _compute_clients_snapshot():
    """WDDM may omit graphics/compute clients; do not interpret empty as exclusive GPU."""
    cmd = ['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name,used_gpu_memory',
           '--format=csv,noheader,nounits']
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           errors='replace', timeout=5)
        return {'available': p.returncode == 0,
                'raw_csv': p.stdout.strip()[:4000],
                'stderr_tail': p.stderr[-500:],
                'wddm_visibility_not_guaranteed': True}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'reason': str(exc)[:350],
                'wddm_visibility_not_guaranteed': True}

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
    report = {'banks': [], 'phase_blocks': {}, 'compiled': {},
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

            if getattr(args, 'measurement_protocol', None) == 'host_device':
                # Research-only observer of the original sentinel, no changed math or model.
                from benchmark_h3_tile_host_device_attribution import collect_control_probe
                return collect_control_probe(
                    torch, sentinel, device, report,
                    lambda phase_name: _verify_partials_and_merge(
                        torch, batch, states, cfg, tr, last_control, last_merged,
                        last_tail, sentinel_out, phase_name + '_post'),
                    _nvidia_smi_snapshot, _compute_clients_snapshot)

            # Three phases, exactly one and the SAME original sentinel CUDA Graph.
            report['compute_clients'] = {'before': _compute_clients_snapshot()}
            report['quiet_declaration'] = 'OPERATOR_CONFIRMED_NOT_INDEPENDENTLY_PROVEN'
            monitor_report = {'available': False, 'samples': [], 'reason': 'not started'}
            for phase_name in PHASES:
                if phase_name == 'monitored':
                    monitor = _NvidiaSmiSampler()
                    monitor.start()
                    try:
                        time.sleep(0.75)  # allow the external reader to yield its first sample
                        phase = _wall_phase(torch, sentinel, device, phase_name)
                    finally:
                        monitor_report = monitor.stop()
                else:
                    phase = _wall_phase(torch, sentinel, device, phase_name)
                phase['post_phase_parity'] = _verify_partials_and_merge(
                    torch, batch, states, cfg, tr, last_control, last_merged,
                    last_tail, sentinel_out, phase_name + '_post')
                report['phase_blocks'][phase_name] = phase
                if not phase['post_phase_parity']['full_window_bit_exact']:
                    raise AssertionError('Original sentinel bits changed: ' + phase_name)
            report['telemetry'] = monitor_report
            report['compute_clients']['after'] = _compute_clients_snapshot()
            report['environment_snapshots']['after'] = _nvidia_smi_snapshot()
            if not monitor_report['available']:
                raise RuntimeError('Monitored phase nvidia-smi telemetry sparse/unavailable: inspect ONE JSON')
            decision = decide_wall_stability(report['phase_blocks'], monitor_report)
            report.update({'status': decision['status'], 'completed': True,
                           'wall_stability': decision,
                           'speed_measurement': 'UNPROFILED_SENTINEL_WALL_ONLY',
                           'gpu_event_stability': 'UNRUN_THIS_DIAGNOSTIC',
                           'candidate_timed': False, 'timed_full_attention': False,
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
    parser.add_argument('--quiet-confirmed', action='store_true')
    parser.add_argument('--warmups', type=int, default=5)
    args = parser.parse_args()
    report = {
        'schema': 'htrc-tile-quiet-environment-v1',
        'status': 'FAIL_CLOSED', 'completed': False,
        'source_head': None, 'probe': 'ONE_PERSISTENT_ORIGINAL_TILE_UNPROFILED_QUIET_MONITORED_QUIET',
        'tokens': TOKENS, 'heads': HEADS, 'tiles': TILES, 'splits': SPLITS,
        'near_leaves': KX, 'far_bn': BN, 'window_query_rows': 2048,
        'phase_order': list(PHASES), 'banks': [], 'phase_blocks': {},
        'numeric_gate': 'UNRUN', 'speed_measurement': 'UNRUN_NUMERIC_GATE',
        'h3_model_or_video': 'UNRUN',
        'timing_scope': 'ORIGINAL_TILE_ONLY_UNPROFILED_WALL_NO_MERGE',
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
        'WALL_CONTROL_STABLE_DIAGNOSTIC_ONLY', 'WALL_CONTROL_UNSTABLE_DIAGNOSTIC_ONLY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
