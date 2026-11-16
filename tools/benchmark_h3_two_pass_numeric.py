"""Pinned two-bank numerical gate for original Tile PASS1 -> PASS2.

The UNMODIFIED Triton _tile_kernel is launched twice with the same
arguments/buffers: far-only PASS=1 then near-resume PASS=2. Compares
raw FP32 PM/PL/PACC and original Merge. CUDA timings deliberately UNRUN.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from analyze_h3_far_exclusion import summarize_exclusions
from profile_h3_far_exclusion import ENV_KEYS
from benchmark_h3_far_masked_loads_numeric import (
    _alloc_partials, _assert_pinned, _compiled_resources,
    compare_far_partials, EXPECTED_MASKS,
    HEADS, TILES, ROWS, D, TOKENS, NB, KX, SPLITS, BN,
)
from benchmark_h3_full_mask_g_numeric import _merge, _compare_bitwise
from htrc_two_pass_tile_adapter import (
    launch_two_pass_tile, validate_original_pass_source,
)

SCHEMA = 'htrc-tile-far-near-pass12-numeric-v1'
OUTPUT = Path(r'D:\ComfyOutput\H3\math_probe\flat_tile_far_near_pass12_numeric.json')


def run_gpu():
    import torch  # imported ONLY after exact source HEAD admission
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live
    from h3 import htrc_flat_model_shadow as shared
    from h3 import htrc_flat_head_batch as batch

    if (not torch.cuda.is_available()
        or torch.cuda.get_device_capability(0) != (12,0)
        or 'RTX 5070 Ti' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Exact RTX 5070 Ti SM120 required')

    torch.set_grad_enabled(False)
    source = (ROOT / 'h3' / 'htrc_flat_triton.py').read_text('utf-8')
    validate_original_pass_source(source)
    torch.manual_seed(42026)
    dev = torch.device('cuda:0')
    shape = (1,HEADS,TOKENS,D)
    banks = [tuple(torch.randn(shape, device=dev, dtype=torch.bfloat16)
                   for _ in range(3)) for _ in range(2)]
    old = {key:os.environ.get(key) for key in ENV_KEYS}
    results = {'banks':[], 'candidate_speed_measurement':'UNRUN_NUMERIC_ONLY',
               'capture_blocks':[], 'production_modified':False}
    full_outputs = []
    try:
        with tempfile.TemporaryDirectory(prefix='htrc-tile-pass12-') as temp:
            overrides = {
                'APL_HTRC_TILE32_STATS_BT':'16',
                'APL_HTRC_DIRECTFLOW_GROUP':'11',
                'APL_HTRC_DIRECTFLOW_WARMUPS':'4',
                'APL_HTRC_DIRECTFLOW_VB_MODE':'materialized',
                'APL_HTRC_DIRECTFLOW_GROUP_POLICY':'legacy',
                'APL_HTRC_DIRECTFLOW_REPORT':str(Path(temp)/'first_call.json'),
                'APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP':None,
                'APL_HTRC_GRAPH_CAPTURE_SAFE_MU':'0',
                'APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB':'768',
                'APL_HTRC_DIRECTFLOW_MIN_FREE_MIB':'256',
                'APL_HTRC_DIRECTFLOW_TILES32_ADMISSION':'0',
            }
            for k,v in overrides.items():
                if v is None: os.environ.pop(k,None)
                else: os.environ[k]=v
            flow.clear_session_cache()
            cfg = shared.flat_config('balanced',TILES)
            tr = live._triton()
            fingerprints=[]
            for bi, bank in enumerate(banks):
                full = flow.graph_attention_bthd(
                    *bank, heads=HEADS, mode='balanced', tiles=TILES,
                    budget_mib=1024, reserve_mib=256, workspace=True)
                torch.cuda.synchronize(dev)
                if bi == 0:
                    first = json.loads(Path(overrides[
                        'APL_HTRC_DIRECTFLOW_REPORT']).read_text('utf-8'))
                    if (first.get('status')!='PASS_FIRST_REAL_CALL'
                        or first.get('all_groups_bit_exact') is not True):
                        raise AssertionError('Original complete first-call Graph/eager failed')
                if tuple(full.shape)!=(1,TOKENS,HEADS,D):
                    raise AssertionError('Original 4-head output shape changed')
                full_outputs.append(full.detach().cpu().clone())
                stage = flow._SESSIONS.get((HEADS,),{})
                item = stage.get('graphs',{}).get((TILES,2048))
                if item is None:
                    raise AssertionError('Pinned 32-tile original Graph window missing')
                states = item['states']
                _assert_pinned(torch,states,cfg)
                summary = summarize_exclusions(
                    states['EXCL'].detach().cpu().tolist(),
                    states['IDX'].detach().cpu().tolist(),
                    splits=SPLITS,bn=BN,expected_near=KX)
                fp=summary['selection_mask_sha256']
                if (fp!=EXPECTED_MASKS[bi]
                    or summary['current_far_dot_blocks']!=1920):
                    raise AssertionError(f'Bank{bi} selection geometry/source drift')
                fingerprints.append(fp)

                control=_alloc_partials(torch,dev)
                _, compiled_control=batch.launch_prepared_tile_batch(
                    states,cfg,tr,buffers=control)
                torch.cuda.synchronize(dev)
                tail=stage['tail']
                ctrl_merged=_merge(torch,batch,states,cfg,tr,control,tail)
                torch.cuda.synchronize(dev)
                baseline_vs_graph=_compare_bitwise(
                    torch,item['output'],ctrl_merged)
                if not baseline_vs_graph['bit_exact']:
                    raise AssertionError('Original standalone Tile/Merge differs from Graph window')

                candidate=_alloc_partials(torch,dev)
                _, compiled_passes=launch_two_pass_tile(
                    batch,states,cfg,tr,buffers=candidate)
                torch.cuda.synchronize(dev)
                raw=compare_far_partials(torch,control,candidate)
                merge_out=_merge(torch,batch,states,cfg,tr,candidate,tail)
                torch.cuda.synchronize(dev)
                merged=_compare_bitwise(torch,ctrl_merged,merge_out)
                exact=all(p['bit_exact'] for p in raw.values()) and merged['bit_exact']
                results['banks'].append({
                    'index':bi,'selection_mask_sha256':fp,
                    'original_window_graph_bit_exact':baseline_vs_graph,
                    'pass1_pass2_same_original_fp32_buffers':True,
                    'candidate_partial_pm_pl_pacc':raw,
                    'candidate_merged_output':merged,
                    'candidate_full_window_bit_exact':exact,
                    'compiled':{
                        'control':_compiled_resources(compiled_control),
                        'far_pass1':_compiled_resources(compiled_passes[0]),
                        'near_pass2':_compiled_resources(compiled_passes[1]),
                    },
                })
                del control,candidate,ctrl_merged,merge_out,full
            if (torch.equal(full_outputs[0],full_outputs[1])
                or fingerprints[0] == fingerprints[1]):
                raise AssertionError('Independent QKV banks did not change output/selection')
            eager_b=live.flat_attention_bthd(
                *banks[1],heads=HEADS,mode='balanced',tiles=TILES,
                budget_mib=1024,reserve_mib=256,workspace=True,
                query_staging='native_torch_mu',
                kv_staging='native_direct',dispatch='eager')
            torch.cuda.synchronize(dev)
            if not torch.equal(eager_b.cpu(),full_outputs[1]):
                raise AssertionError('Bank B complete native eager Graph parity failed')
            results.update({
                'two_distinct_bf16_qkv_banks':True,
                'first_complete_graph_eager_bit_exact':True,
                'second_complete_graph_eager_bit_exact':True,
                'candidate_both_banks_bit_exact':all(
                    row['candidate_full_window_bit_exact']
                    for row in results['banks']),
                'candidate_complete_56head_graph':'UNRUN',
                'h3_model_or_video':'UNRUN',
                'speed_capture_blocks':[],
                'candidate_speed_measurement':'UNRUN_NUMERIC_ONLY',
            })
            return results
    finally:
        try: flow.clear_session_cache()
        finally:
            for key,value in old.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expect-head',required=True)
    p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args()
    report={
        'schema':SCHEMA, 'status':'FAIL_CLOSED','completed':False,
        'source_head':None,
        'probe':'ORIGINAL_PASS0_VS_ORIGINAL_PASS1_THEN_PASS2',
        'heads':HEADS,'tiles':TILES,'tokens':TOKENS,
        'exact_leaves':KX,'splits':SPLITS,'bn':BN,
        'candidate_speed_measurement':'UNRUN_NUMERIC_ONLY',
        'capture_blocks':[],'h3_model_or_video':'UNRUN',
        'candidate_complete_56head_graph':'UNRUN',
        'production_promotion':'PROHIBITED_RESEARCH_ONLY',
        'banks':[],
    }
    try:
        actual=subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],
            text=True).strip()
        if actual!=args.expect_head:
            raise RuntimeError('Exact HEAD mismatch '+actual+' != '+args.expect_head)
        report['source_head']=actual
        data=run_gpu()
        report.update(data)
        if report['candidate_both_banks_bit_exact']:
            report.update(status='PASS_TWO_PASS_NUMERIC_ONLY',
                          completed=True,
                          parity_admission='PASS_TWO_BANK_PM_PL_PACC_MERGE_BIT_EXACT')
        else:
            report.update(status='NO_GO_TWO_PASS_NUMERIC',
                          completed=True,
                          parity_admission='NO_GO_TWO_PASS_BIT_EXACT')
    except BaseException as exc:
        report.update(status='FAIL_CLOSED',completed=False,
                      error=f'{type(exc).__name__}: {exc}'[:1800],
                      traceback_tail=traceback.format_exc()[-4500:])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('one_json='+str(args.output)+' status='+str(report['status']))
    return 0 if report['status'] in (
        'PASS_TWO_PASS_NUMERIC_ONLY','NO_GO_TWO_PASS_NUMERIC') else 2


if __name__=='__main__':
    raise SystemExit(main())
