"""Selected Flat HTRC CUDA attention for real MiniMax H3 denoising.

Unlike the shadow gate, this function returns Flat's output to H3 out_proj.
The caller (ComfyNodes attention_runtime) owns model residency, admission,
fallback policy and telemetries. This file owns only bounded per-attention-call
GPU K/V caches, query workspaces and tiled grouped kernels.

Requirements: B=1 self-attention, native BHSD, no mask, 56 (or arbitrary
positive number of) genuinely distinct heads, D=128, inference only.
No synthetic/repeated heads; no cross-step K/V caching; no model hook.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

from . import htrc_flat_head_batch as batch
from . import htrc_flat_model_shadow as shared


_TRITON_NAME = "_apl_htrc_external_live_flat_triton"


def _triton():
    """Load THIS HTRC checkout's kernels, not ComfyNodes' old Flat module."""
    path=Path(__file__).with_name("htrc_flat_triton.py").resolve()
    loaded=sys.modules.get(_TRITON_NAME)
    if loaded is not None:
        if Path(loaded.__file__).resolve()!=path:
            raise RuntimeError("Conflicting HTRC Flat Triton module roots")
        return loaded
    spec=importlib.util.spec_from_file_location(_TRITON_NAME,path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Missing canonical Flat Triton kernels at {path}")
    module=importlib.util.module_from_spec(spec)
    sys.modules[_TRITON_NAME]=module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_TRITON_NAME,None)
        raise
    return module


@torch.inference_mode()
def flat_attention_bthd(
    q, k, v, *, heads, mode="fast", tiles=16,
    budget_mib=1024, reserve_mib=256, workspace=True,
    observer=None,
):
    """Return the actual [1,S,H,128] BF16/FP16 Flat attention output.

    Q,K,V are raw plain post-projection/post-RoPE BHSD CUDA tensors.
    The caller must not invoke this function in training or with mask.
    Intermediate buffers are disposed at return. The only kept tensor is
    the output, so the H3 memory manager can release its attention reserve.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("HTRC Flat live backend is inference-only")
    cap=shared.capture_from_bhsd(q,k,v,int(heads))
    s=int(cap["q_qhd"].shape[0])
    count=int(heads)
    if count < 1 or s < 16:
        raise ValueError("Requires at least 1 real head and 16 tokens")
    if not isinstance(tiles,int) or tiles<1:
        raise ValueError("tiles must be a positive integer")
    if budget_mib<1 or reserve_mib<0 or budget_mib<=reserve_mib+128:
        raise ValueError("Budget must exceed reserve plus 128 MiB fixed buffer")
    cfg=shared.flat_config(mode,tiles)
    tr=_triton()
    tr._dll()
    dev=q.device
    # This is the caller-owned full output, not a QKV duplicate. It remains
    # alive until the layer consumes it; include its bytes in memory admission.
    out=torch.empty((s,count,128),device=dev,dtype=q.dtype)
    max_group,plan=shared._head_plan(
        cap,count,tiles,cfg,budget_mib=int(budget_mib),
        reserve_mib=int(reserve_mib),workspace=bool(workspace),
    )
    source=q.dtype
    details={
        "backend":"htrc_flat_gpu",
        "heads":count, "distinct_native_head_ids":True,
        "tokens":s,"mode":mode,"tiles":tiles,
        "query_windows":math.ceil(s/(tiles*64)),
        "group_capacity":max_group,
        "number_of_groups":math.ceil(count/max_group),
        "budget_mib":int(budget_mib),
        "reserve_mib":int(reserve_mib),
        "output_mib":out.numel()*out.element_size()/batch.MIB,
        "free_at_plan_mib":plan["free_at_plan_mib"],
        "kv_cache_scope":"one_attention_call",
        "workspace_reused":bool(workspace),
    }
    try:
        for begin,end in batch.iter_head_groups(count,max_group):
            ids=list(range(begin,end))
            # All K/V bytes are computed ONCE for this genuine head group.
            cache=batch.prepare_gpu_head_kv_cache(cap,ids,tr)
            ws=(shared.alloc_workspace(cap,len(ids),tiles,cfg)
                if workspace else None)
            tails=shared._tail(cap,ids)
            try:
                for qi in range(0,s,tiles*64):
                    qe=min(s,qi+tiles*64)
                    valid=qe-qi
                    src=cap["q_qhd"][qi:qe]
                    if valid<tiles*64:
                        # Repeat last true Q row. Correct the partial tile
                        # mean using valid_query_rows in preparation.
                        src=torch.cat((
                            src,src[-1:].expand(tiles*64-valid,-1,-1)
                        ),dim=0)
                    view=dict(cap)
                    view["q_qhd"]=src
                    states,_,_,_,_=batch.prepare_batched_head_states(
                        view,ids,cfg,tr,tile_count=tiles,
                        quant_backend="triton",input_cast="gpu",
                        kv_cache=cache,query_workspace=ws,
                        valid_query_rows=valid,
                    )
                    parts,_=batch.launch_prepared_tile_batch(
                        states,cfg,tr,buffers=(ws["_partials"] if ws else None)
                    )
                    merged,_=batch.launch_prepared_merge_batch(
                        states,cfg,tr,parts,tails=tails,
                        output=(ws["_out"] if ws else None),
                        merge_rows=32,
                    )
                    # [H,Tiles,64,D] -> [valid,H,D], BF16 output to native
                    # projection. Copy results before reusing workspace.
                    selected=merged.reshape(len(ids),tiles*64,128)[:,:valid]
                    out[qi:qe,begin:end,:].copy_(
                        selected.permute(1,0,2).to(source)
                    )
            finally:
                # Releasing a group never allows reuse across denoising steps,
                # model blocks, prompts, CFG passes or concurrent attention.
                del cache,ws,tails
    except BaseException:
        del out
        raise
    if observer is not None:
        observer("htrc_flat_gpu_dispatch",details)
    return out.unsqueeze(0)
