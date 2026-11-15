"""Full Tile far+near and original Merge strict two-bank mask-G numeric gate.

NO performance timing, new production imports, model weights, or Comfy edits.
The research unmasked mirror must match original full Tile and Merge first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0,str(ROOT/'tools'))
from analyze_h3_far_exclusion import summarize_exclusions
from profile_h3_far_exclusion import ENV_KEYS
from benchmark_h3_far_masked_loads_numeric import (
    _alloc_partials, _assert_pinned, compare_far_partials,
    _compiled_resources, EXPECTED_MASKS,
    HEADS, TILES, ROWS, D, TOKENS, NB, KX, SPLITS, LPS, BN,
)


def _launch_full_candidate(states, kernel, output, *, mask_g: bool):
    q=states['Q']
    qs=tuple(map(int,q.stride()))
    v=states['V']
    direct_v=bool(states.get('_direct_v',False))
    vs=tuple(map(int,v.stride())) if direct_v else (0,0,0)
    keys=int(states['K'].shape[1])
    if keys!=15264 or tuple(states['KS'].shape)!=(HEADS,keys):
        raise AssertionError('Unexpected packed exact K8/KS geometry')
    if direct_v and (v.ndim!=3 or v.shape[0]!=HEADS or v.shape[-1]!=D):
        raise AssertionError('Unexpected direct V layout')
    if not direct_v and not v.is_contiguous():
        raise AssertionError('Staged V must be contiguous')
    vv = v if direct_v else v.view(HEADS*keys,D)
    return kernel[(HEADS*TILES,SPLITS)](
        q,states['MU'].view(HEADS*TILES,D),
        states['G'].view(HEADS*TILES,NB,D),
        states['LZ'].view(HEADS*TILES,NB),
        states['VB'].view(HEADS*TILES,NB,D),
        states['EXCL'].view(HEADS*TILES,NB),
        states['IDX'].view(HEADS*TILES,KX),
        states['K'].view(HEADS*keys,D),
        vv,states['KS'].view(HEADS*keys),
        states['SG'].view(HEADS*TILES,NB),
        output['pm'].view(HEADS*TILES,SPLITS,ROWS),
        output['pl'].view(HEADS*TILES,SPLITS,ROWS),
        output['acc'].view(HEADS*TILES,SPLITS,ROWS,D),
        NB=NB,KX=KX,LPS=LPS,EPS=2048,
        TILES=TILES,KEYS=keys,T=ROWS,D=D,BN=BN,LEAF=16,
        MASK_G=int(mask_g),
        Q_SH=qs[0],Q_ST=qs[1],Q_SR=qs[2],Q_SD=qs[3],
        V_DIRECT=int(direct_v),V_SH=vs[0],V_SS=vs[1],V_SD=vs[2],
        SCALE=1.0/math.sqrt(D),num_warps=4,num_stages=2,
    )


def _compare_bitwise(torch, ref, candidate):
    if (ref.device!=candidate.device or ref.dtype!=torch.float32 or
        candidate.dtype!=torch.float32 or ref.shape!=candidate.shape):
        raise AssertionError('Merge dtype/device/shape mismatch')
    differ=ref.view(torch.int32)!=candidate.view(torch.int32)
    count=int(differ.count_nonzero().item())
    first=torch.nonzero(differ,as_tuple=False)
    first_idx=[int(x) for x in first[0].cpu().tolist()] if count else None
    good=torch.isfinite(ref)&torch.isfinite(candidate)
    max_abs=float((ref[good]-candidate[good]).abs().max().item()) if bool(good.any().item()) else None
    return {'bit_exact':count==0,'bit_mismatch_count':count,
            'total_elements':ref.numel(),'first_mismatch_index':first_idx,
            'max_finite_abs_difference':max_abs}


def _merge(torch, batch, states, cfg, tr, partial, tail):
    out=torch.empty((HEADS,TILES,ROWS,D),device=states['Q'].device,dtype=torch.float32)
    merged,_=batch.launch_prepared_merge_batch(states,cfg,tr,partial,tails=tail,
                                                 output=out,merge_rows=32)
    return merged


def _check_variant(torch, batch, states, cfg, tr, base, base_out, tail, kernel, name, mask_g):
    scratch=_alloc_partials(torch,states['Q'].device)
    compiled=_launch_full_candidate(states,kernel,scratch,mask_g=mask_g)
    torch.cuda.synchronize(states['Q'].device)
    partials=compare_far_partials(torch,base,scratch)
    partial_exact=all(v['bit_exact'] for v in partials.values())
    # Always compute the existing Merge for diagnosis, even on a partial mismatch.
    merged=_merge(torch,batch,states,cfg,tr,scratch,tail)
    torch.cuda.synchronize(states['Q'].device)
    output=_compare_bitwise(torch,base_out,merged)
    return {'variant':name,'pm_pl_pacc':partials,'all_partial_bits_exact':partial_exact,
            'merge_output':output,'full_window_bit_exact':partial_exact and output['bit_exact'],
            'compiled':_compiled_resources(compiled)}


def run_gpu():
    # Every CUDA/Triton dependency is deliberately imported after exact-HEAD check.
    import torch
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live
    from h3 import htrc_flat_model_shadow as shared
    from h3 import htrc_flat_head_batch as batch
    from htrc_full_mask_g_triton import full_mask_g_partial

    if (not torch.cuda.is_available() or
        torch.cuda.get_device_capability(0)!=(12,0) or
        'RTX 5070 Ti' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Pinned RTX 5070 Ti SM120 required')
    torch.set_grad_enabled(False)
    torch.manual_seed(42026)
    device=torch.device('cuda:0')
    shape=(1,HEADS,TOKENS,D)
    banks=[tuple(torch.randn(shape,device=device,dtype=torch.bfloat16)
                 for _ in range(3)) for _ in range(2)]
    old={key:os.environ.get(key) for key in ENV_KEYS}
    report={'banks':[],'speed_capture_blocks':[]}
    original_outputs=[]
    try:
        with tempfile.TemporaryDirectory(prefix='htrc-full-mask-g-') as tmp:
            env={
                'APL_HTRC_TILE32_STATS_BT':'16',
                'APL_HTRC_DIRECTFLOW_GROUP':'11',
                'APL_HTRC_DIRECTFLOW_WARMUPS':'4',
                'APL_HTRC_DIRECTFLOW_VB_MODE':'materialized',
                'APL_HTRC_DIRECTFLOW_GROUP_POLICY':'legacy',
                'APL_HTRC_DIRECTFLOW_REPORT':str(Path(tmp)/'first_call.json'),
                'APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP':None,
                'APL_HTRC_GRAPH_CAPTURE_SAFE_MU':'0',
                'APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB':'768',
                'APL_HTRC_DIRECTFLOW_MIN_FREE_MIB':'256',
                'APL_HTRC_DIRECTFLOW_TILES32_ADMISSION':'0',
            }
            for key,value in env.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
            flow.clear_session_cache()
            cfg=shared.flat_config('balanced',TILES)
            tr=live._triton()
            for bi,bank in enumerate(banks):
                full=flow.graph_attention_bthd(*bank,heads=HEADS,mode='balanced',
                    tiles=TILES,budget_mib=1024,reserve_mib=256,workspace=True)
                torch.cuda.synchronize(device)
                if bi==0:
                    gate=json.loads((Path(tmp)/'first_call.json').read_text(encoding='utf-8'))
                    if gate.get('status')!='PASS_FIRST_REAL_CALL' or gate.get('all_groups_bit_exact') is not True:
                        raise AssertionError('Original full first-call Graph/eager failed')
                if tuple(full.shape)!=(1,TOKENS,HEADS,D):
                    raise AssertionError('Full source Graph result shape changed')
                original_outputs.append(full.detach().cpu().clone())
                session=flow._SESSIONS.get((HEADS,))
                if session is None:raise AssertionError('Prepared Graph session missing')
                item=session['graphs'].get((TILES,2048))
                if item is None:raise AssertionError('Pinned 2048-row window missing')
                states=item['states']
                _assert_pinned(torch,states,cfg)
                selection=summarize_exclusions(states['EXCL'].detach().cpu().tolist(),
                    states['IDX'].detach().cpu().tolist(),splits=SPLITS,bn=BN,
                    expected_near=KX)
                fingerprint=selection['selection_mask_sha256']
                if fingerprint!=EXPECTED_MASKS[bi]:
                    raise AssertionError(f'Stale/unexpected EXCL bank{bi}: {fingerprint}')
                if selection['current_far_dot_blocks']!=1920 or selection['fully_excluded_blocks']!=0:
                    raise AssertionError('Unexpected BN64 source leaf coverage')

                tail=session['tail']
                if tail is None or tuple(tail[0].shape)!=(HEADS,11,D):
                    # 15,275 % 16 == 11 and tails are exact key/value tokens.
                    raise AssertionError('Missing eleven exact tail K/V tokens')
                control=_alloc_partials(torch,device)
                _,ref_compiled=batch.launch_prepared_tile_batch(states,cfg,tr,buffers=control)
                torch.cuda.synchronize(device)
                ref_merged=_merge(torch,batch,states,cfg,tr,control,tail)
                torch.cuda.synchronize(device)
                graph_window_parity=_compare_bitwise(torch,item['output'],ref_merged)
                if not graph_window_parity['bit_exact']:
                    raise AssertionError('Baseline standalone full Tile+Merge not equal to captured Graph output')
                mirror=_check_variant(torch,batch,states,cfg,tr,control,
                    ref_merged,tail,full_mask_g_partial,'mirror',False)
                variants={'mirror':mirror}
                if mirror['full_window_bit_exact']:
                    variants['mask_g']=_check_variant(torch,batch,states,cfg,tr,
                        control,ref_merged,tail,full_mask_g_partial,'mask_g',True)
                report['banks'].append({
                    'bank':bi,
                    'selection_mask_sha256':fingerprint,
                    'original_far_bn64_blocks':selection['current_far_dot_blocks'],
                    'baseline_full_tile_merge_equals_graph_window':graph_window_parity,
                    'original_compiled':_compiled_resources(ref_compiled),
                    'variants':variants,
                })
                del control,ref_merged,full
            if (torch.equal(original_outputs[0],original_outputs[1]) or
                report['banks'][0]['selection_mask_sha256']==report['banks'][1]['selection_mask_sha256']):
                raise AssertionError('Two QKV banks were not independent')
            independent=live.flat_attention_bthd(*banks[1],heads=HEADS,
                mode='balanced',tiles=TILES,budget_mib=1024,reserve_mib=256,
                workspace=True,query_staging='native_torch_mu',
                kv_staging='native_direct',dispatch='eager')
            if not torch.equal(independent.cpu(),original_outputs[1]):
                raise AssertionError('Bank B source full eager/Graph mismatch')
            mirror_ok=all(b['variants']['mirror']['full_window_bit_exact'] for b in report['banks'])
            mask_ok=mirror_ok and all(b['variants'].get('mask_g',{}).get('full_window_bit_exact',False)
                                      for b in report['banks'])
            report.update({'two_distinct_bf16_qkv_banks':True,
                'first_complete_graph_eager_bit_exact':True,
                'bank_b_complete_graph_eager_bit_exact':True,
                'baseline_window_graph_parity':True,
                'mirror_full_numeric_bit_exact':mirror_ok,
                'mask_g_full_numeric_bit_exact':mask_ok,
                'full_masked_attention':'FAR_NEAR_AND_MERGE_WINDOW_ONLY',
                'no_cuda_event_timing':True,'speed_measurement':'UNRUN_NUMERIC_ONLY'})
            return report
    finally:
        try:flow.clear_session_cache()
        finally:
            for key,value in old.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expect-head',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report={'schema':'htrc-full-mask-g-numeric-v1',
        'status':'FAIL_CLOSED','completed':False,
        'probe':'ORIGINAL_PASS0_FULL_TILE_AND_MERGE_VS_MASK_G',
        'source_head':None,'tokens':TOKENS,'heads':HEADS,'tiles':TILES,
        'splits':SPLITS,'near_leaves':KX,'far_bn':BN,
        'window_only_query_rows':2048,
        'timing_scope':'NUMERIC_ONLY_NO_CANDIDATE_TIMING',
        'speed_measurement':'UNRUN_NUMERIC_ONLY',
        'production_promotion':'PROHIBITED_RESEARCH_ONLY',
        'h3_model_or_video':'UNRUN',
        'no_cuda_event_timing':True,'speed_capture_blocks':[],
        'banks':[]}
    try:
        actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],
                                       text=True).strip()
        if actual!=args.expect_head:
            raise RuntimeError(f'Exact HEAD mismatch {actual} != {args.expect_head}')
        report['source_head']=actual
        result=run_gpu()
        report.update(result)
        if not report['mirror_full_numeric_bit_exact']:
            report['status']='FAIL_CLOSED'
            report['parity_admission']='NO_GO_FULL_RESEARCH_MIRROR'
            report['error']='Research full unmasked clone fails baseline partials or merge'
        elif not report['mask_g_full_numeric_bit_exact']:
            report['status']='FAIL_CLOSED'
            report['parity_admission']='NO_GO_FULL_MASK_G_BIT_EXACT'
            report['error']='Mask G full Tile partials or original Merge differ in FP32 bits'
        else:
            report['status']='PASS_FULL_MASK_G_NUMERIC_ONLY'
            report['completed']=True
            report['parity_admission']='PASS_FULL_WINDOW_PARTIALS_AND_MERGE_ONLY'
    except BaseException as exc:
        report['status']='FAIL_CLOSED'
        report['completed']=False
        report['error']=f'{type(exc).__name__}: {exc}'[:1800]
        report['traceback_tail']=traceback.format_exc()[-4500:]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"one_json={args.output} status={report['status']} parity={report.get('parity_admission','UNRUN')}")
    return 0 if report['status']=='PASS_FULL_MASK_G_NUMERIC_ONLY' else 2


if __name__=='__main__':
    raise SystemExit(main())
