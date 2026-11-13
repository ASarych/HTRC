"""Research-only pre-QKV streaming into HTRC Flat CUDA Graph.

Consumes H3-style *dense BF16* fused projection weights laid out as
[Q(all heads), K(all heads), V(all heads)]. Does not support quantized/NVFP4
weights, Q/K RMSNorm, or RoPE: those must be integrated and numerically gated
before any use with a real H3 model. Not a production replacement.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DIM = 128


def validate_projection(x, weight, bias, heads):
    if torch.is_grad_enabled():
        raise RuntimeError("Pre-QKV streaming is inference-only")
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("Expected dense native X and QKV weight tensors")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("Expected X=[N,width], weight=[3*heads*128,width]")
    if type(heads) is not int or heads < 1:
        raise ValueError("heads must be a positive integer")
    if tuple(weight.shape) != (3 * heads * DIM, x.shape[1]):
        raise ValueError("Fused QKV weight order/shape mismatch")
    if (x.dtype != torch.bfloat16 or weight.dtype != x.dtype
        or x.device != weight.device or x.shape[0] < 16):
        raise ValueError("Requires same-device BF16 X/weight and >=16 tokens")
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or tuple(bias.shape) != (3 * heads * DIM,)
        or bias.dtype != x.dtype or bias.device != x.device
    ):
        raise ValueError("Incompatible fused QKV bias")


def group_weights(weight, bias, heads, begin, end):
    """Select genuine output channels, never re-project unrelated heads."""
    if not (0 <= begin < end <= heads):
        raise ValueError("Invalid native head range")
    a = begin * DIM
    b = end * DIM
    stride = heads * DIM
    wq = weight[a:b]
    wkv = torch.cat((weight[stride+a:stride+b],
                     weight[2*stride+a:2*stride+b]), dim=0)
    if bias is None:
        return wq, wkv, None, None
    bq = bias[a:b]
    bkv = torch.cat((bias[stride+a:stride+b],
                     bias[2*stride+a:2*stride+b]), dim=0)
    return wq, wkv, bq, bkv


def project_kv_chunks(x, wkv, bkv, group_heads, chunk_rows=4096):
    """Compute K/V ONCE for one head group; no full fused QKV tensor."""
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if tuple(wkv.shape) != (2*group_heads*DIM, x.shape[1]):
        raise ValueError("Invalid K/V projection shard")
    n = x.shape[0]
    k = torch.empty((1, group_heads, n, DIM), dtype=x.dtype, device=x.device)
    v = torch.empty_like(k)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        piece = F.linear(x[start:end], wkv, bkv).view(
            end-start, 2, group_heads, DIM)
        k[0, :, start:end, :].copy_(piece[:, 0].permute(1, 0, 2))
        v[0, :, start:end, :].copy_(piece[:, 1].permute(1, 0, 2))
        del piece
    return k, v


def query_window_provider(x, wq, bq, group_heads):
    """Return a writer for stable Graph Q staging, including the last pad."""
    if tuple(wq.shape) != (group_heads*DIM, x.shape[1]):
        raise ValueError("Invalid Q projection shard")

    def fill(start, end, stage):
        if not (0 <= start < end <= x.shape[0]):
            raise ValueError("Invalid Q window")
        real = end-start
        if (tuple(stage.shape[::2]) != (group_heads, DIM)
            or stage.shape[1] < real or stage.dtype != x.dtype
            or stage.device != x.device):
            raise ValueError("Invalid Graph Q staging buffer")
        q = F.linear(x[start:end], wq, bq).view(real, group_heads, DIM)
        stage[:, :real, :].copy_(q.permute(1, 0, 2))
        if real < stage.shape[1]:
            stage[:, real:, :].copy_(
                stage[:, real-1:real, :].expand(
                    -1, stage.shape[1]-real, -1))
    return fill


@torch.no_grad()
def projected_graph_attention(x, weight, *, heads=56, group_heads=8,
                              chunk_rows=4096, bias=None, mode="balanced",
                              tiles=16, budget_mib=1024, reserve_mib=256):
    """All-group output without full QKV. Research-only; no H3 RoPE/norm.

    A group owns full-sequence K/V because Flat queries the entire key range;
    Q is projected on demand per window, into the Graph-owned Q buffer.
    The existing Graph staging still copies K/V once per group: eliminating
    that copy is an *unimplemented* separate optimization.
    """
    validate_projection(x, weight, bias, heads)
    if x.device.type != "cuda":
        raise ValueError("Graph experiment requires CUDA")
    if (type(group_heads) is not int or group_heads < 1
        or heads % group_heads != 0):
        raise ValueError("Research Graph requires equal head groups")
    if chunk_rows < 1 or type(chunk_rows) is not int:
        raise ValueError("Invalid projection chunk_rows")
    from . import htrc_flat_directflow_live as flow

    output = torch.empty((1, x.shape[0], heads, DIM),
                         dtype=x.dtype, device=x.device)
    for begin in range(0, heads, group_heads):
        end = begin + group_heads
        wq, wkv, bq, bkv = group_weights(weight, bias, heads, begin, end)
        k, v = project_kv_chunks(x, wkv, bkv, group_heads, chunk_rows)
        fill_q = query_window_provider(x, wq, bq, group_heads)
        # The K tensor also acts as a SHAPE-ONLY placeholder for Q during
        # capture. The writer replaces every Q staging window BEFORE replay.
        # No full Q tensor is materialized, and no QKV persists across groups.
        block = flow.graph_attention_bthd(
            k, k, v, heads=group_heads, mode=mode, tiles=tiles,
            budget_mib=budget_mib, reserve_mib=reserve_mib,
            workspace=True, compare_only=True, query_provider=fill_q)
        output[:, :, begin:end, :].copy_(block)
        del block, k, v, wq, wkv, bq, bkv, fill_q
    return output
