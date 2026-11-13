"""Opt-in Triton fused H3 QKV activation packer (NOT weight quantization).

Input: a genuine contiguous BF16 [rows, 3*heads*128] projection tensor
AFTER Comfy's in-place fused Q/K RMSNorm/RoPE.
Output: existing INT8 [3,H,full_tokens,128] and FP32 [3,H,full_tokens]
activation storage. One launch jointly handles Q/K/V instead of
materializing three temporary FP32 activation tensors per chunk.

No projective GEMM, no weight reads/slicing, no Graph execution, and no
production backend registration. Import this module only when explicitly
selected with APL_HTRC_QKV_PACK_BACKEND=triton.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

DIM = 128
ROWS_PER_CTA = 16
HEADS_PER_CTA = 4


@triton.jit
def _pack_fused_qkv(
    INPUT, PACKED, SCALES,
    ROWS, CHUNK_START,
    HEADS: tl.constexpr,
    TOTAL_TOKENS: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
):
    r = tl.program_id(0) * R + tl.arange(0, R)
    h = tl.program_id(1) * H + tl.arange(0, H)
    p = tl.program_id(2)  # 0=Q, 1=K, 2=V
    d = tl.arange(0, DIM)

    good_r = r < ROWS
    good_h = h < HEADS
    valid = good_r[:, None, None] & good_h[None, :, None]

    # Input is contiguous [rows, Q(all heads)|K(all heads)|V(all heads)].
    input_offset = (
        r[:, None, None] * (3 * HEADS * DIM)
        + (p * HEADS + h[None, :, None]) * DIM
        + d[None, None, :]
    )
    value = tl.load(INPUT + input_offset, mask=valid, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 2), 1.0e-8) / 127.0
    quant = tl.minimum(
        tl.maximum(libdevice.nearbyint(value / scale[:, :, None]), -127.0),
        127.0,
    ).to(tl.int8)

    # Output is [Q/K/V, HEAD, TOKEN, DIM], so a head group can later
    # unpack contiguous token rows into a Graph-owned BF16 buffer.
    absolute_r = CHUNK_START + r
    output_offset = (
        (p * HEADS + h[None, :, None]) * (TOTAL_TOKENS * DIM)
        + absolute_r[:, None, None] * DIM
        + d[None, None, :]
    )
    scale_offset = (
        (p * HEADS + h[None, :]) * TOTAL_TOKENS
        + absolute_r[:, None]
    )
    tl.store(PACKED + output_offset, quant, mask=valid)
    tl.store(
        SCALES + scale_offset, scale,
        mask=good_r[:, None] & good_h[None, :],
    )


def pack_normalized_fused(fused, packed, scales, *, start, tokens, heads):
    """Pack one native normalized QKV chunk with no FP32 tensor allocations."""
    if (not isinstance(fused, torch.Tensor)
        or fused.device.type != "cuda" or fused.dtype != torch.bfloat16
        or fused.ndim != 2 or not fused.is_contiguous()):
        raise ValueError("Triton QKV pack requires contiguous CUDA BF16 fused output")
    rows = int(fused.shape[0])
    if (not isinstance(packed, torch.Tensor)
        or not isinstance(scales, torch.Tensor)
        or tuple(fused.shape) != (rows, 3 * heads * DIM)
        or tuple(packed.shape) != (3, heads, tokens, DIM)
        or tuple(scales.shape) != (3, heads, tokens)
        or packed.dtype != torch.int8 or scales.dtype != torch.float32
        or not packed.is_contiguous() or not scales.is_contiguous()
        or packed.device != fused.device or scales.device != fused.device
        or not 0 <= start < start + rows <= tokens
        or heads < 1):
        raise ValueError("Incompatible native QKV compact staging geometry")

    grid = (
        triton.cdiv(rows, ROWS_PER_CTA),
        triton.cdiv(heads, HEADS_PER_CTA),
        3,
    )
    _pack_fused_qkv[grid](
        fused, packed, scales,
        rows, start,
        heads, tokens,
        ROWS_PER_CTA, HEADS_PER_CTA,
        num_warps=4,
    )
