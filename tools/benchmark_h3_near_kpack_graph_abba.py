"""Matched unprofiled 56-head CUDA Graph A/B/B/A for near K8+scale packing.

Candidate is research-only and injected ONLY in flow._one_window during
CUDAGraph capture. Both banks, original eager, PM/PL/PACC and Merge must
pass before ANY timing. Four independent captures control/packed/packed/
control; includes K/V refresh, K8 pack, 105 Graph replay windows, output.
Exactly one fail-closed JSON; no model weights, no production updates.
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
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools'))
from benchmark_h3_56head_stage_attribution import (
    TOKENS, HEADS, TILES, DIM, GROUP_CAP, ENV,
    verify_first_call, verify_graph_timing, _write_one,
)
from benchmark_h3_far_masked_loads_numeric import (
    compare_far_partials, _compiled_resources,
)
from htrc_near_kpack_graph_adapter import (
    NearKGraphAdapter, SCRATCH_BYTES, GROUP_HEADS, MAX_TILES,
    ROWS, SPLITS,
)

SCHEMA='htrc-56head-near-kpack-full-graph-abba-v1'
ROUNDS,WARMUPS=7,5
CAPTURE_ORDER=('control','packed','packed','control')
DEFAULT_OUTPUT=Path(r'D:\ComfyOutput\H3\math_probe\flat_56head_near_kpack_graph_abba.json')


def _spread(values):
    if not values or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
        raise ValueError('Missing/nonpositive A/B/B/A timings')
    return (max(values)/min(values)-1)*100


def decide_abba(blocks):
    if ([b['arm'] for b in blocks] != list(CAPTURE_ORDER)
        or any(len(b['samples']) != ROUNDS for b in blocks)):
        raise ValueError('A/B/B/A blocks incomplete or wrong order')
    controls=[float(b['gpu_event_median_ms']) for b in blocks if b['arm']=='control']
    packed=[float(b['gpu_event_median_ms']) for b in blocks if b['arm']=='packed']
    c=statistics.median(controls)
    p=statistics.median(packed)
    cs,ps=_spread(controls),_spread(packed)
    median_reduction=(1-p/c)*100
    worst_reduction=(1-max(packed)/min(controls))*100
    stable=cs<=3.0 and ps<=3.0
    passes=stable and median_reduction>=5 and worst_reduction>=5
    return {
        'capture_order':list(CAPTURE_ORDER),
        'control_capture_medians_ms':controls,
        'packed_capture_medians_ms':packed,
        'control_capture_spread_pct':cs,
        'packed_capture_spread_pct':ps,
        'required_max_per_arm_capture_spread_pct':3.0,
        'control_median_of_capture_medians_ms':c,
        'packed_median_of_capture_medians_ms':p,
        'median_full_graph_reduction_pct':median_reduction,
        'worst_pair_full_graph_reduction_pct':worst_reduction,
        'required_reduction_pct':5.0,
        'strict_capture_stability':stable,
        'candidate_wins_both_independent_captures':max(packed)<min(controls),
        'promising_full_graph_signal':passes,
        'decision':('PROMISING_RESEARCH_ONLY' if passes else
                    'INCONCLUSIVE_CAPTURE_DRIFT' if not stable else
                    'NO_GO_FULL_GRAPH'),
        'scope':'MATCHED_56HEAD_FULL_GRAPH_INCLUDES_PER_WINDOW_K8_PACK',
        'not_real_h3_model_or_video':True,
        'production_promotion_prohibited':True,
    }


def _raw_equal(torch,a,b):
    if tuple(a.shape)!=tuple(b.shape) or a.dtype!=b.dtype:
        return False
    if a.dtype==torch.bfloat16:
        return bool(torch.equal(a.contiguous().view(torch.int16),
                                b.contiguous().view(torch.int16)))
    if a.dtype==torch.float32:
        return bool(torch.equal(a.contiguous().view(torch.int32),
                                b.contiguous().view(torch.int32)))
    raise TypeError('Only raw BF16/F32 outputs admitted')


def _partials(torch,dev,h,tiles):
    shape=(h,tiles,SPLITS,ROWS)
    return {'pm':torch.empty(shape,device=dev,dtype=torch.float32),
            'pl':torch.empty(shape,device=dev,dtype=torch.float32),
            'acc':torch.empty(shape+(DIM,),device=dev,dtype=torch.float32)}


def strict_packed_window_numeric(torch,flow,batch,shared,live,adapter,bank_index):
    """Compare raw PM/PL/PACC+Merge of last 8 heads/last full 1024-row window."""
    stage=flow._SESSIONS.get((GROUP_CAP,))
    if stage is None:
        raise AssertionError('Missing captured 8-head K8 Graph session')
    item=stage['graphs'].get((TILES,TILES*ROWS))
    if item is None:
        raise AssertionError('Missing full 16-tile 1024-row Graph window')
    states=item['states']
    cfg=shared.flat_config('balanced',TILES)
    tr=live._triton()
    dev=states['Q'].device
    ref=_partials(torch,dev,GROUP_CAP,TILES)
    packed=_partials(torch,dev,GROUP_CAP,TILES)
    _,ctrl_compiled=batch.launch_prepared_tile_batch(states,cfg,tr,buffers=ref)
    _,pk_compiled=adapter.launch(states,cfg,tr,buffers=packed)
    torch.cuda.synchronize(dev)
    fields=compare_far_partials(torch,ref,packed)
    if not all(row['bit_exact'] for row in fields.values()):
        raise AssertionError('56-head candidate raw PM/PL/PACC bank %d not bit exact'%bank_index)
    tail=stage['tail']
    def merge(partial):
        output=torch.empty((GROUP_CAP,TILES,ROWS,DIM),device=dev,dtype=torch.float32)
        result,_=batch.launch_prepared_merge_batch(
            states,cfg,tr,partial,tails=tail,output=output,merge_rows=32)
        return result
    merged_original=merge(ref)
    merged_packed=merge(packed)
    torch.cuda.synchronize(dev)
    full_bit_exact=_raw_equal(torch,merged_original,merged_packed)
    if not full_bit_exact:
        raise AssertionError('56-head candidate Merge bank %d not bit exact'%bank_index)
    resources={
        'original':_compiled_resources(ctrl_compiled),
        'near_kpacked':_compiled_resources(pk_compiled)}
    orig=resources['original'];pk=resources['near_kpacked']
    if any(row[k] is None for row in (orig,pk)
           for k in ('registers_per_thread','shared_bytes_per_cta','compiler_spills')):
        raise AssertionError('Missing compiled resources')
    if (pk['compiler_spills'] or orig['compiler_spills']
        or pk['registers_per_thread']>orig['registers_per_thread']
        or pk['shared_bytes_per_cta']>orig['shared_bytes_per_cta']):
        raise AssertionError('K8 candidate resource/spill gate FAIL')
    del ref,packed,merged_original,merged_packed
    return {
        'bank':bank_index,
        'original_56head_graph_eager_parity':'CHECKED_IN_FIRST_CALL_OR_INDEPENDENT_REFERENCE',
        'tested_head_group':[48,56],
        'tested_full_window_rows':TILES*ROWS,
        'pm_pl_pacc':fields,
        'merge_bit_exact':True,
        'all_partial_bits_exact':True,
        'compiled_resources':resources,
    }


def _first_and_second_bank(torch,flow,live,banks,gate_path,adapter=None,
                           reference_a=None,reference_b=None,
                           raw_checks=False):
    """Capture once per arm. A first call uses ORIGINAL eager oracle."""
    from h3 import htrc_flat_head_batch as batch
    from h3 import htrc_flat_model_shadow as shared

    os.environ['APL_HTRC_DIRECTFLOW_REPORT']=str(gate_path)
    def graph_call(bank):
        return flow.graph_attention_bthd(
            *bank,heads=HEADS,mode='balanced',tiles=TILES,
            budget_mib=1024,reserve_mib=256,workspace=True,
            compare_only=False)
    if adapter is None:
        a=graph_call(banks[0])
    else:
        with adapter.capturing():
            a=graph_call(banks[0])
        # Also retain Python tensor references after the patch is lifted:
        # both captured geometries use the exact same stable scratch pointer.
        flow._SESSIONS[(GROUP_CAP,)]['research_near_k_refs']=adapter.retention()
    torch.cuda.synchronize(banks[0][0].device)
    first=json.loads(gate_path.read_text('utf-8'))
    first_result=verify_first_call(first,0)
    if reference_a is not None and not _raw_equal(torch,a.cpu(),reference_a):
        raise AssertionError('Independent capture bank A output disagrees with reference')
    num=[]
    if adapter is not None and raw_checks:
        num.append(strict_packed_window_numeric(
            torch,flow,batch,shared,live,adapter,0))
    # The same captured Graph session must be refreshed using a distinct bank.
    b=graph_call(banks[1])
    torch.cuda.synchronize(banks[1][0].device)
    if reference_b is not None and not _raw_equal(torch,b.cpu(),reference_b):
        raise AssertionError('Bank B full 56-head result differs from eager reference')
    if adapter is not None and raw_checks:
        num.append(strict_packed_window_numeric(
            torch,flow,batch,shared,live,adapter,1))
    if adapter is not None and len(num)!=2 and raw_checks:
        raise AssertionError('Both candidate PM/PL/PACC banks required')
    return a.cpu(),b.cpu(),first_result,num


def _measure(torch,call,rounds):
    device=torch.device('cuda:0')
    samples=[]
    for _ in range(rounds):
        start,stop=(torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True))
        start.record()
        t0=time.perf_counter()
        output=call()
        stop.record()
        stop.synchronize()
        samples.append({'gpu_ms':float(start.elapsed_time(stop)),
                        'sync_wall_ms':(time.perf_counter()-t0)*1000})
        del output
    return verify_graph_timing(samples)


def _prepare_adapter(torch,flow):
    from h3 import htrc_flat_head_batch as batch
    from htrc_near_kpack_triton import (
        pack_near_key_and_scale,near_kpacked_tile_kernel)
    return NearKGraphAdapter(
        torch,flow,batch,pack_near_key_and_scale,near_kpacked_tile_kernel)


def run_gpu(args,result):
    import torch  # CUDA only AFTER exact-source verification in main()
    if (not torch.cuda.is_available()
        or torch.cuda.get_device_capability(0)!=(12,0)
        or 'RTX 5070 Ti' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Pinned RTX 5070 Ti SM120 required')
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live
    dev=torch.device('cuda:0')
    env_keys=(*ENV,'APL_HTRC_DIRECTFLOW_REPORT',
              'APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP',
              'APL_HTRC_DIRECTFLOW_WARMUPS')
    previous={k:os.environ.get(k) for k in env_keys}
    old_report_path=None
    try:
        os.environ.update(ENV)
        os.environ['APL_HTRC_DIRECTFLOW_WARMUPS']='4'
        os.environ.pop('APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP',None)
        with torch.inference_mode(),tempfile.TemporaryDirectory(prefix='htrc-near-56-abba-') as temp:
            banks=[]
            for bi in range(2):
                torch.manual_seed(42026+bi)
                banks.append(tuple(torch.randn((1,HEADS,TOKENS,DIM),device=dev,
                                               dtype=torch.bfloat16) for _ in range(3)))
            first_bits=[b[0].flatten()[:32].cpu().contiguous().view(torch.int16).tolist()
                        for b in banks]
            if first_bits[0]==first_bits[1]:
                raise AssertionError('Two BF16 QKV banks unexpectedly identical')
            result['two_distinct_bf16_qkv_banks']=True
            result['source_first32_q_bits']=first_bits
            # An independent unchanged native EAGER oracle for bank B is
            # established before any candidate timing.
            eager_b=live.flat_attention_bthd(
                *banks[1],heads=HEADS,mode='balanced',tiles=TILES,
                budget_mib=1024,reserve_mib=256,workspace=True,
                query_staging='native_torch_mu',kv_staging='native_direct',
                dispatch='eager')
            torch.cuda.synchronize(dev)
            reference_b=eager_b.detach().cpu().clone()
            del eager_b
            gc.collect()
            torch.cuda.empty_cache()

            # Strict preflight, NO speed measurements yet: original Graph
            # establishes reference bank A, then packed Graph checks BOTH
            # complete 56-head outputs and raw full-window PM/PL/PACC+Merge.
            flow.clear_session_cache()
            original_a,original_b,original_gate,_=_first_and_second_bank(
                torch,flow,live,banks,Path(temp)/'numeric_original.json',
                reference_b=reference_b)
            result['numeric_original_first_call']=original_gate
            if not _raw_equal(torch,original_b,reference_b):
                raise AssertionError('Original complete 56-head bank B/eager mismatch')
            reference_a=original_a.clone()
            del original_a,original_b
            flow.clear_session_cache()
            gc.collect()
            torch.cuda.empty_cache()

            research=_prepare_adapter(torch,flow)
            pa,pb,candidate_gate,parity=_first_and_second_bank(
                torch,flow,live,banks,Path(temp)/'numeric_packed.json',
                adapter=research,reference_a=reference_a,
                reference_b=reference_b,raw_checks=True)
            if not (_raw_equal(torch,pa,reference_a)
                    and _raw_equal(torch,pb,reference_b)
                    and len(parity)==2
                    and all(row['all_partial_bits_exact'] and row['merge_bit_exact']
                            for row in parity)):
                raise AssertionError('Strict two-bank numerical preflight FAIL')
            result['preflight_candidate_first_call']=candidate_gate
            result['preflight_candidate_raw_partials']=parity
            result['candidate_shared_scratch']=research.recorded_scratch()
            result['numeric_gate']='PASS_TWO_BANK_FULL_56HEAD_AND_RAW_PM_PL_PACC_BIT_EXACT'
            del pa,pb,research
            flow.clear_session_cache()
            gc.collect()
            torch.cuda.empty_cache()

            # All timed captures are independent, with replay warmups
            # AFTER capture. Pack kernel is captured INSIDE 105 window Graphs
            # and therefore is included in full-call timing.
            result['candidate_speed_measurement']='IN_PROGRESS_ABBA'
            for ci,arm in enumerate(CAPTURE_ORDER):
                flow.clear_session_cache()
                gc.collect()
                torch.cuda.empty_cache()
                reserved_base=int(torch.cuda.memory_reserved(dev))
                initial_free_mib=torch.cuda.mem_get_info(dev)[0]/(1<<20)
                adapter=_prepare_adapter(torch,flow) if arm=='packed' else None
                a,b,strict,num=_first_and_second_bank(
                    torch,flow,live,banks,Path(temp)/f'capture_{ci}.json',
                    adapter=adapter,reference_a=reference_a,
                    reference_b=reference_b,raw_checks=arm=='packed')
                del a,b
                if arm=='packed' and (len(num)!=2 or adapter is None):
                    raise AssertionError('Candidate capture lacked two-bank raw gate')
                def measured_call():
                    return flow.graph_attention_bthd(
                        *banks[1],heads=HEADS,mode='balanced',tiles=TILES,
                        budget_mib=1024,reserve_mib=256,workspace=True,
                        compare_only=False)
                for _ in range(args.warmups):
                    warm=measured_call()
                    del warm
                torch.cuda.synchronize(dev)
                timed=_measure(torch,measured_call,args.rounds)
                after=measured_call()
                torch.cuda.synchronize(dev)
                exact_end=_raw_equal(torch,after.cpu(),reference_b)
                del after
                if not exact_end:
                    raise AssertionError('Timed Graph replay changed complete bank B output')
                result['capture_blocks'].append({
                    'index':ci,'arm':arm,
                    'first_call_all_56_heads_strict_eager_bit_exact':True,
                    'separate_bank_B_full_output_bit_exact':True,
                    'pre_timing_full_output_bit_exact':True,
                    'pre_timing_raw_partials':(
                        'PASS_TWO_BANK_PM_PL_PACC_AND_MERGE'
                        if arm=='packed' else 'UNCHANGED_CONTROL'),
                    'post_timing_full_output_bit_exact':exact_end,
                    'raw_candidate_numeric':num,
                    'gpu_event_median_ms':timed['gpu_event_median_ms'],
                    'gpu_event_spread_pct':timed['gpu_event_block_spread_pct'],
                    'wall_median_ms':timed['wall_median_ms'],
                    'samples':timed['samples'],
                    'reserved_before_capture_mib':reserved_base/(1<<20),
                    'reserved_after_timing_mib':torch.cuda.memory_reserved(dev)/(1<<20),
                    'pre_capture_free_mib':initial_free_mib,
                    'scratch':adapter.recorded_scratch() if adapter is not None else None,
                    'runtime_graph_capture_hard_768mib_gate':'PASSED',
                })
                del adapter
                flow.clear_session_cache()
                gc.collect()
                torch.cuda.empty_cache()
            result['abba_decision']=decide_abba(result['capture_blocks'])
            result['candidate_speed_measurement']='MEASURED_FULL_56HEAD_GRAPH_INCLUDES_PACK'
            decision=result['abba_decision']['decision']
            result['status']=('PASS_NUMERIC_PROMISING_FULL_GRAPH' if
                              decision=='PROMISING_RESEARCH_ONLY' else
                              'PASS_NUMERIC_INCONCLUSIVE_CAPTURE' if
                              decision=='INCONCLUSIVE_CAPTURE_DRIFT' else
                              'PASS_NUMERIC_NO_GO_FULL_GRAPH')
            result['completed']=True
    finally:
        try:flow.clear_session_cache()
        finally:
            for k,v in previous.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expect-head',required=True)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--quiet-confirmed',action='store_true')
    p.add_argument('--rounds',type=int,default=ROUNDS)
    p.add_argument('--warmups',type=int,default=WARMUPS)
    args=p.parse_args()
    report={
        'schema':SCHEMA,'status':'FAIL_CLOSED','completed':False,
        'source_head':None,'probe':'ONE_COMPLETE_56HEAD_NEAR_K8_GRAPH_ABBA',
        'input_kind':'SYNTHETIC_56_DISTINCT_BF16_QKV',
        'heads':HEADS,'tokens':TOKENS,'tiles':TILES,'group_cap':GROUP_CAP,
        'near_leaves':512,'near_tokens':8192,
        'shared_packed_k8_and_scale_bytes_per_captured_session':SCRATCH_BYTES,
        'numeric_gate':'UNRUN','candidate_speed_measurement':'UNRUN_NUMERIC_GATE_ONLY',
        'capture_blocks':[],'abba_decision':None,
        'h3_model_or_video':'UNRUN',
        'independent_model_resident_vram':'UNRUN',
        'production_promotion':'PROHIBITED_RESEARCH_ONLY',
        'one_json_no_tensor_dump':True,
    }
    try:
        if not args.quiet_confirmed:
            raise ValueError('Explicit quiet GPU confirmation required')
        if args.rounds!=7 or args.warmups!=5:
            raise ValueError('Fixed ABBA protocol requires rounds7 warmups5')
        if len(args.expect_head)!=40 or any(c not in '0123456789abcdef' for c in args.expect_head):
            raise ValueError('Exact lowercase Git SHA required')
        sha=subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        if sha!=args.expect_head:
            raise RuntimeError('Exact HEAD mismatch '+sha+' != '+args.expect_head)
        report['source_head']=sha
        report=run_gpu(args,report)
    except BaseException as exc:
        report.update(status='FAIL_CLOSED',completed=False,
                      error=f'{type(exc).__name__}: {exc}'[:2200],
                      traceback_tail=traceback.format_exc()[-5000:])
    _write_one(args.output,report)
    print(f"one_json={args.output} status={report['status']} numeric={report['numeric_gate']}")
    return 0 if report['status'] in (
        'PASS_NUMERIC_PROMISING_FULL_GRAPH',
        'PASS_NUMERIC_INCONCLUSIVE_CAPTURE',
        'PASS_NUMERIC_NO_GO_FULL_GRAPH',
    ) else 2


if __name__=='__main__':
    raise SystemExit(main())
