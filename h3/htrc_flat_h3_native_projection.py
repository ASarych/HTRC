"""Research-only native MiniMax-H3 pre-QKV projection into grouped Graph buffers.

Calls attention.qkv_proj on chunks WITHOUT reading/slicing/dequantizing
weights. Preserves Comfy's NVFP4/INT8/FP8 operations, model patches and
forward hooks. Then runs Comfy's actual fused Q/K RMSNorm+split-half RoPE.

For now the complete fused projection is repeated for every 8-head group:
56 heads / 8 = SEVEN full projection passes. This is a correctness and
memory gate, NOT a performance-optimized channel-sliced QKV kernel.
No automatic backend promotion. Never cache activations across denoise steps.
"""
from __future__ import annotations

import torch

DIM = 128


def validate(attn, x, rope_freqs, begin, end, q_out, k_out, v_out, *,
             chunk_rows):
    if torch.is_grad_enabled():
        raise RuntimeError("H3 native streaming is strictly inference-only")
    if (not isinstance(x, torch.Tensor) or x.ndim != 2
        or x.dtype != torch.bfloat16 or x.device.type != "cuda"
        or x.shape[0] < 16):
        raise ValueError("Expected CUDA BF16 X [N,hidden] and N>=16")
    heads = int(getattr(attn, "heads", 0))
    dim = int(getattr(attn, "head_dim", 0))
    if dim != DIM or heads < 1 or not callable(getattr(attn, "qkv_proj", None)):
        raise ValueError("Expected real H3 fused QKV projector and D=128")
    if (not isinstance(begin, int) or not isinstance(end, int)
        or not 0 <= begin < end <= heads):
        raise ValueError("Invalid distinct head range")
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= x.shape[0]:
        raise ValueError("Invalid QKV projection chunk")
    expected = (1, end-begin, x.shape[0], DIM)
    for name, buf in (("q", q_out), ("k", k_out), ("v", v_out)):
        if (not isinstance(buf, torch.Tensor)
            or tuple(buf.shape) != expected or buf.device != x.device
            or buf.dtype != x.dtype):
            raise ValueError(f"Invalid {name} destination shape/dtype/device")
    if len({q_out.data_ptr(), k_out.data_ptr(), v_out.data_ptr()}) != 3:
        raise ValueError("Q/K/V destinations must not alias")
    if rope_freqs is not None and (
        not isinstance(rope_freqs, torch.Tensor)
        or rope_freqs.ndim != 6 or rope_freqs.shape[0] != 1
        or rope_freqs.shape[1] != x.shape[0]
        or rope_freqs.device != x.device
        or rope_freqs.dtype != x.dtype
        or rope_freqs.shape[-3]*2 > DIM
    ):
        raise ValueError("Invalid native H3 rope rotation table")
    return heads


def normalized_native_group_chunk(attn, fused, rope_chunk, *,
                                  begin, end, qw=None, kw=None,
                                  rms_rope_inplace=None):
    """H3's exact projection split, native Q/K normalization and head selection.

    This CPU-testable primitive receives the unmodified module's fused
    projection result. It performs no quantized-weight operations itself.
    Returned Q/K/V are views unless a norm module returns new tensors;
    consumers MUST copy them before the next projection chunk.
    """
    rows = int(fused.shape[0])
    heads = int(attn.heads)
    q, k, v = fused.split(heads*DIM, dim=-1)
    if rope_chunk is not None:
        if rms_rope_inplace is None or qw is None or kw is None:
            raise ValueError("Native split-half RoPE requires fused kernel and norm weights")
        q4 = q.view(1, rows, heads, DIM)
        k4 = k.view(1, rows, heads, DIM)
        rms_rope_inplace(
            q4, k4, rope_chunk, qw, kw,
            epsilon=attn.q_norm.eps,
            rot_dim=rope_chunk.shape[-3]*2,
        )
        q, k = q4[0], k4[0]
    else:
        q = attn.q_norm(q.view(rows, heads, DIM))
        k = attn.k_norm(k.view(rows, heads, DIM))
    v = v.view(rows, heads, DIM)
    return q[:, begin:end, :], k[:, begin:end, :], v[:, begin:end, :]


def project_group_native(attn, x, rope_freqs, *, begin, end,
                         q_out, k_out, v_out, chunk_rows=4096,
                         resolve_norms=None, rms_rope_inplace=None):
    """Fill caller-owned group Q/K/V using the unmodified model's qkv_proj.

    q/k output is RoPE-normalized precisely where stock ComfyUI applies it.
    With no rope, call the original q_norm and k_norm modules.
    resolve_norms/rms_rope_inplace are test-only dependency injection points.
    """
    heads = validate(attn, x, rope_freqs, begin, end, q_out, k_out, v_out,
                     chunk_rows=chunk_rows)
    if rope_freqs is not None:
        if resolve_norms is None or rms_rope_inplace is None:
            import comfy.model_management as mm
            import comfy.quant_ops as quant_ops
            if bool(mm.in_training):
                raise RuntimeError("Native H3 chunked Graph refuses training")
            if resolve_norms is None:
                resolve_norms = lambda: (
                    mm.cast_to(attn.q_norm.weight, device=x.device),
                    mm.cast_to(attn.k_norm.weight, device=x.device),
                )
            if rms_rope_inplace is None:
                rms_rope_inplace = quant_ops.ck.rms_rope_split_half_
        qw, kw = resolve_norms()
        if not all(isinstance(t, torch.Tensor) and t.device == x.device
                   for t in (qw, kw)):
            raise ValueError("H3 Q/K norm weights not resident on CUDA")
    else:
        qw = kw = None

    calls = 0
    for start in range(0, x.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x.shape[0])
        rows = stop-start
        # Preserve the exact installed module call: don't slice packed
        # NVFP4 weights, bypass TensorCore quant dispatch, or alter hooks.
        fused = attn.qkv_proj(x[start:stop])
        if (not isinstance(fused, torch.Tensor) or
            tuple(fused.shape) != (rows, 3*heads*DIM) or
            fused.dtype != x.dtype or fused.device != x.device):
            raise ValueError("Native qkv_proj output shape/dtype/device changed")
        q, k, v = normalized_native_group_chunk(
            attn, fused,
            rope_freqs[:, start:stop] if rope_freqs is not None else None,
            begin=begin, end=end, qw=qw, kw=kw,
            rms_rope_inplace=rms_rope_inplace)
        sl = slice(start, stop)
        q_out[0, :, sl, :].copy_(q.permute(1, 0, 2))
        k_out[0, :, sl, :].copy_(k.permute(1, 0, 2))
        v_out[0, :, sl, :].copy_(v.permute(1, 0, 2))
        calls += 1
        del fused, q, k, v
    return {"projection_chunks": calls, "projection_rows": int(x.shape[0]),
            "head_range": [begin, end],
            "full_fused_qkv_per_chunk": True,
            "selective_nvfp4_output_projection": False}


class NativeGroupProducer:
    """Fresh projection for one head group; Q used after K/V writer.

    A model could patch qkv_proj between calls: always invoke the module,
    never cache quantized weights or a stale QKV result.
    """
    def __init__(self, attn, x, rope_freqs, begin, end, chunk_rows=4096):
        self.attn = attn
        self.x = x
        self.rope_freqs = rope_freqs
        self.begin = begin
        self.end = end
        self.chunk_rows = chunk_rows
        self.q = torch.empty((1, end-begin, x.shape[0], DIM),
                             device=x.device, dtype=x.dtype)
        self.stats = None

    def fill_kv(self, k, v):
        self.stats = project_group_native(
            self.attn, self.x, self.rope_freqs,
            begin=self.begin, end=self.end, q_out=self.q,
            k_out=k, v_out=v, chunk_rows=self.chunk_rows)

    def fill_q(self, start, end, stage):
        if self.stats is None:
            raise RuntimeError("K/V producer must populate group Q before replay")
        real = end-start
        if (not isinstance(stage, torch.Tensor) or stage.ndim != 3
            or not 0 <= start < end <= self.x.shape[0]
            or stage.shape[0] != self.end-self.begin
            or stage.shape[1] < real or stage.shape[2] != DIM
            or stage.dtype != self.x.dtype or stage.device != self.x.device):
            raise ValueError("Invalid Graph Q stage for native H3 producer")
        src = self.q[0, :, start:end, :]
        stage[:, :real, :].copy_(src)
        if real < stage.shape[1]:
            stage[:, real:, :].copy_(
                src[:, -1:, :].expand(-1, stage.shape[1]-real, -1))


@torch.no_grad()
def graph_attention_native_h3(attn, x, rope_freqs, *, group_heads=8,
                              chunk_rows=4096, mode="balanced", tiles=16,
                              budget_mib=1024, reserve_mib=256):
    """One native model attention, preserving original quantized qkv_proj.

    Requests one 8-head research session per call with a per-call override.
    It bypasses the conservative head-group ESTIMATE only, not actual
    Graph extra reservation or free-VRAM fail-closed gates.
    Returns same [N, hidden] shape after original out_proj as stock Attention.
    """
    from . import htrc_flat_directflow_live as flow
    heads = int(getattr(attn, "heads", 0))
    if heads < 1 or type(group_heads) is not int or group_heads < 1 or heads % group_heads:
        raise ValueError("Native H3 Flat requires equal genuine head groups")
    if torch.is_grad_enabled():
        raise RuntimeError("Real H3 Graph research requires no-grad eval")
    s = int(x.shape[0])
    # Shape-only metadata placeholders; no real QKV values read from these.
    marker = torch.zeros((1, 1, 1, 1), device=x.device, dtype=x.dtype)
    output = torch.empty((1, s, heads, DIM), device=x.device, dtype=x.dtype)
    summary = []
    for begin in range(0, heads, group_heads):
        end = begin+group_heads
        producer = NativeGroupProducer(attn, x, rope_freqs, begin, end,
                                       chunk_rows=chunk_rows)
        placeholder = marker.expand(1, group_heads, s, DIM)
        result = flow.graph_attention_bthd(
            placeholder, placeholder, placeholder,
            heads=group_heads, mode=mode, tiles=tiles,
            budget_mib=budget_mib, reserve_mib=reserve_mib,
            workspace=True, compare_only=True,
            query_provider=producer.fill_q, kv_provider=producer.fill_kv,
            research_source_kind="native_h3",
            research_group_size=group_heads)
        output[:, :, begin:end, :].copy_(result)
        summary.append(producer.stats)
        del result, producer
    projected = attn.out_proj(output.reshape(s, heads*DIM))
    return projected, {
        "projection_mode": "full_fused_qkv_repeated_per_head_group",
        "projection_passes_per_attention": heads//group_heads,
        "projection_chunk_calls_per_attention": sum(x["projection_chunks"] for x in summary),
        "selective_quantized_weight_slicing": "UNIMPLEMENTED",
        "real_model_graph_math": "HTRC Flat balanced or fast, not stock dense/Sol",
    }
