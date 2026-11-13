"""Exact all-key dense attention for sampled H3 queries; quality percentages.

The reference is NOT eager/materialized Flat. It is standard scaled
dot-product softmax using *all* key/value tokens and FP32 accumulations.
Sampling query rows bounds O(N^2) GPU time/VRAM on long contexts.
These quality numbers describe sampled query positions, not whole-video
perceptual quality or a measured all-query exact baseline.
"""
from __future__ import annotations

import math


def query_positions(tokens: int, samples: int) -> tuple[int, ...]:
    if (type(tokens) is not int or tokens < 1 or
        type(samples) is not int or samples < 1):
        raise ValueError("tokens and samples must be positive integers")
    if samples >= tokens:
        return tuple(range(tokens))
    if samples == 1:
        return (tokens // 2,)
    positions=tuple((i*(tokens-1))//(samples-1) for i in range(samples))
    assert positions[0] == 0 and positions[-1] == tokens-1
    assert len(set(positions)) == samples
    return positions


def exact_dense_sampled_queries(q, k, v, positions, *, query_batch: int = 8):
    """Return [sample_count,heads,128] FP32 dense reference on all keys.

    Input BF16 BHSD [1,H,S,128], global noncausal attention as the
    fixed H3 self-attention operator; no causal mask or sparse leaves.
    FP32 logits, softmax and value reduction intentionally differ from
    BF16 mixed-precision SDPA for maximal reference precision.
    """
    import torch
    if (not isinstance(positions,tuple) or not positions or
        query_batch<1):
        raise ValueError("Need nonempty positions and query_batch >= 1")
    if any(not isinstance(x,int) for x in positions):
        raise TypeError("Invalid query index type")
    if (any(x.ndim!=4 or x.shape[0]!=1 or x.shape[-1]!=128
            or x.dtype!=torch.bfloat16 or x.device.type!="cuda"
            for x in (q,k,v))
        or q.shape!=k.shape or q.shape!=v.shape):
        raise ValueError("Expected identical BF16 CUDA post-RoPE BHSD QKV")
    h=int(q.shape[1]); tokens=int(q.shape[2])
    if any(x<0 or x>=tokens for x in positions):
        raise ValueError("Query sample index outside sequence")
    if len(set(positions))!=len(positions):
        raise ValueError("Duplicate query positions")
    out=torch.empty((len(positions),h,128),dtype=torch.float32,device=q.device)
    scaling=1.0/math.sqrt(128.0)
    with torch.inference_mode():
        for head in range(h):
            # One head at a time avoids an N x N matrix and bounds
            # temporary logits to [query_batch, N].
            kh=k[0,head].float().contiguous()
            vh=v[0,head].float().contiguous()
            for start in range(0,len(positions),query_batch):
                ix=positions[start:start+query_batch]
                qh=q[0,head,list(ix),:].float().contiguous()
                logits=(qh@kh.transpose(0,1))*scaling
                probabilities=torch.softmax(logits,dim=-1)
                out[start:start+len(ix),head,:]=probabilities@vh
    return out


def quality_against_exact(full_output, reference, positions):
    """Percent relative L2 error (global and per-head/query distribution)."""
    import torch
    if (full_output.ndim!=4 or full_output.shape[0]!=1 or
        reference.ndim!=3 or reference.shape[-1]!=128 or
        reference.shape[0]!=len(positions) or
        full_output.shape[2]!=reference.shape[1] or
        full_output.shape[-1]!=128):
        raise ValueError("Full-output/reference sample geometry mismatch")
    ix=torch.as_tensor(positions,device=full_output.device,dtype=torch.long)
    candidate=full_output[0].index_select(0,ix).float()
    reference=reference.float()
    finite=bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    if not finite:
        return {"valid":False,"finite":False,"relative_l2_percent":None,
                "per_head_rel_l2_percent":None,
                "per_query_rel_l2_p50_percent":None,
                "per_query_rel_l2_p90_percent":None,
                "per_query_rel_l2_max_percent":None,
                "mean_abs":None,"max_abs":None,"sampled_queries":len(positions)}
    diff=candidate-reference
    numerator=torch.linalg.vector_norm(diff)
    denominator=torch.linalg.vector_norm(reference).clamp_min(1.0e-30)
    per_head_n=torch.linalg.vector_norm(diff,dim=(0,2))
    per_head_d=torch.linalg.vector_norm(reference,dim=(0,2)).clamp_min(1.0e-30)
    per_query_n=torch.linalg.vector_norm(diff,dim=(1,2))
    per_query_d=torch.linalg.vector_norm(reference,dim=(1,2)).clamp_min(1.0e-30)
    per_head=(per_head_n/per_head_d*100).detach().cpu().tolist()
    per_query=(per_query_n/per_query_d*100)
    return {
        "valid":True,"finite":True,
        "relative_l2_percent":float(numerator/denominator*100),
        "per_head_rel_l2_percent":[float(v) for v in per_head],
        "per_head_rel_l2_p50_percent":float(torch.quantile(
            per_head_n/per_head_d*100,0.5)),
        "per_head_rel_l2_p90_percent":float(torch.quantile(
            per_head_n/per_head_d*100,0.9)),
        "per_head_rel_l2_max_percent":float((per_head_n/per_head_d*100).max()),
        "per_query_rel_l2_p50_percent":float(torch.quantile(per_query,0.5)),
        "per_query_rel_l2_p90_percent":float(torch.quantile(per_query,0.9)),
        "per_query_rel_l2_max_percent":float(per_query.max()),
        "mean_abs":float(diff.abs().mean()),
        "max_abs":float(diff.abs().max()),
        "sampled_queries":len(positions),
        "exact_reference": "FP32_softmax_all_KV_for_sampled_query_positions",
        "sample_coverage": "evenly_spaced_including_endpoints",
    }


def normalized_difference_percent(a,b,reference,positions):
    """Compare two Flat modes; denominator is dense reference, not mode A."""
    import torch
    if a.shape!=b.shape:
        raise ValueError("Different Flat output shapes")
    ix=torch.as_tensor(positions,device=a.device,dtype=torch.long)
    diff=a[0].index_select(0,ix).float()-b[0].index_select(0,ix).float()
    denominator=torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-30)
    if not bool(torch.isfinite(diff).all()):
        return None
    return float(torch.linalg.vector_norm(diff)/denominator*100.0)
