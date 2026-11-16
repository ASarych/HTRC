"""Preliminary matched-QKV 56-head fast(256) vs balanced(512) quality pilot.

Reference is exact FP32 dense attention across every key for 64 sampled Q
positions; quality percentages, not torch.equal between Flat modes.
One JSON, original materialized Graph only. No production/default changes,
no closed split_stream/fused_vb runs. Timings are single-capture pilot,
NOT independent A/B/B/A speed admission.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
OUTPUT = Path(r'D:\ComfyOutput\H3\math_probe\flat_fast_balanced_56head_dense_quality.json')
SCHEMA = 'htrc-56head-fast-balanced-dense-quality-pilot-v1'
MODES=('balanced','fast')
TOKENS=15275
HEADS=56
TILES=16
SEED=42026
SAMPLED=64
ROUNDS=7


def interpret(arms):
    if tuple(arms)!=MODES:
        raise ValueError('Missing ordered balanced/fast arms')
    a,b=(arms[name] for name in MODES)
    for mode,r in zip(MODES,(a,b)):
        if (r.get('status')!='MEASURED' or
            r.get('bank_count')!=2 or
            r.get('rng_seed')!=SEED or
            r.get('mode')!=mode or
            r.get('tokens')!=TOKENS or
            r.get('heads')!=HEADS or
            r.get('window_tiles')!=TILES or
            r.get('sampled_query_positions')!=a.get('sampled_query_positions') or
            r.get('reference',{}).get('name')!='dense_exact_scaled_dot_product'):
            raise ValueError('Invalid two-bank dense quality input or geometry')
        x=r.get('results',{})
        if list(x)!=['materialized']:
            raise ValueError('Unwanted VB experiment or missing materialized Graph')
        q=x['materialized'].get('quality',[])
        if len(q)!=2 or any(not z.get('valid') or not math.isfinite(
                   z.get('relative_l2_percent',float('nan'))) for z in q):
            raise ValueError('Invalid dense-reference quality')
        timing=x['materialized'].get('timing',{})
        if (timing.get('status')!='TIMED_NOT_QUALITY_GATE'
            or len(timing.get('samples',[]))!=ROUNDS
            or timing.get('gpu_event_ms_median',0)<=0):
            raise ValueError('Missing complete native Graph pilot timings')
        if x['materialized'].get('different_qkv_banks_produced_bit_exact_same_output') is not False:
            raise ValueError('Two source banks not demonstrably distinct')
    def q(mode,i):
        return arms[mode]['results']['materialized']['quality'][i]
    balanced_ms=a['results']['materialized']['timing']['gpu_event_ms_median']
    fast_ms=b['results']['materialized']['timing']['gpu_event_ms_median']
    return {
        'quality_reference':'exact_FP32_dense_all_keys_sampled_64_query_positions',
        'quality_budget_policy':'REPORT_ONLY_NO_AUTOMATIC_THRESHOLD',
        'same_rng_seed_and_bf16_geometry_each_arm':True,
        'full_qkv_bit_identity_between_arms_not_independently_hashed':True,
        'bank_quality':[
          {'bank':i,
           'balanced_relative_l2_percent':q('balanced',i)['relative_l2_percent'],
           'fast_relative_l2_percent':q('fast',i)['relative_l2_percent'],
           'fast_minus_balanced_percentage_points':(
              q('fast',i)['relative_l2_percent']-
              q('balanced',i)['relative_l2_percent']),
           'balanced_query_rel_l2_p90_percent':q('balanced',i)['per_query_rel_l2_p90_percent'],
           'fast_query_rel_l2_p90_percent':q('fast',i)['per_query_rel_l2_p90_percent'],
           'balanced_query_rel_l2_max_percent':q('balanced',i)['per_query_rel_l2_max_percent'],
           'fast_query_rel_l2_max_percent':q('fast',i)['per_query_rel_l2_max_percent'],
           'balanced_head_rel_l2_max_percent':q('balanced',i)['per_head_rel_l2_max_percent'],
           'fast_head_rel_l2_max_percent':q('fast',i)['per_head_rel_l2_max_percent']}
          for i in (0,1)
        ],
        'single_capture_unpaired_time':{
            'balanced_gpu_event_median_ms':balanced_ms,
            'fast_gpu_event_median_ms':fast_ms,
            'preliminary_fast_reduction_percent':(1-fast_ms/balanced_ms)*100,
            'not_an_independent_capture_ABBA_speedup':True,
        },
        'quality_tradeoff_requires_real_H3_video_validation':True,
    }


def run_gpu():
    # No torch/CUDA import prior to exact pinned git source verification.
    from h3_flat_quality_ab import run_case
    arms={}
    for mode in MODES:
        arms[mode]=run_case(
            tokens=TOKENS,heads=HEADS,mode=mode,tiles=TILES,
            rounds=ROUNDS,budget_mib=1024,reserve_mib=256,
            variants=(),sample_queries=SAMPLED,seed=SEED)
    return {'arms':arms,'comparison':interpret(arms)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expect-head',required=True)
    p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args()
    report={'schema':SCHEMA,'status':'FAIL_CLOSED','completed':False,
            'source_head':None,
            'input':'TWO_IDENTICAL_SEEDED_56HEAD_BF16_QKV_PAIRS_PER_MODE',
            'modes':list(MODES),'heads':HEADS,'tokens':TOKENS,
            'tiles':TILES,'near_leaves':{'balanced':512,'fast':256},
            'query_reference_sample_count':SAMPLED,
            'speed_admission':'UNRUN_INDEPENDENT_CAPTURE_ABBA',
            'h3_model_or_video':'UNRUN',
            'production_promotion':'PROHIBITED_RESEARCH_ONLY',
            'arms':{},'comparison':None}
    try:
        actual=subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        if actual!=args.expect_head:
            raise RuntimeError('Wrong exact source SHA '+actual+' expected '+args.expect_head)
        report['source_head']=actual
        report.update(run_gpu())
        report.update(status='QUALITY_MEASURED_SPEED_PRELIMINARY_ONLY',
                      completed=True)
    except BaseException as exc:
        report.update(status='FAIL_CLOSED',completed=False,
                      error=f'{type(exc).__name__}: {exc}'[:1800],
                      traceback_tail=traceback.format_exc()[-4000:])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"one_json={args.output} status={report['status']}")
    return 0 if report['status']=='QUALITY_MEASURED_SPEED_PRELIMINARY_ONLY' else 2

if __name__=='__main__':
    raise SystemExit(main())
