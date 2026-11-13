"""Research-only: one native fused H3 QKV pass with compact all-head staging.

Never slices or dequantizes model weights. A chunk-sized genuine qkv_proj
produces BF16 Q/K/V, including Comfy's native RMSNorm/RoPE. Only the staged
activations are quantized: per-token/per-head symmetric INT8 with FP32 scales.
Graph-owned group K/V and Q windows are reconstructed just in time.

This is a distinct speed/VRAM trade-off, not the validated Flat backend and
not a video-quality or numerical-parity assertion. No caching across forwards.
"""
from __future__ import annotations

import os
import time

import torch

DIM = 128


class CompactAllHeadsQKV:
    """One projector invocation per token chunk, not per head group."""

    def __init__(self, attn, x, rope_freqs, *, chunk_rows=4096,
                 normalize_chunk=None):
        if torch.is_grad_enabled():
            raise RuntimeError("One-pass compact H3 QKV must run without gradients")
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or
            x.dtype != torch.bfloat16 or x.shape[0] < 1 or
            not callable(getattr(attn, "qkv_proj", None))):
            raise ValueError("Expected BF16 H3 hidden states and qkv_proj")
        if (type(chunk_rows) is not int or not 1 <= chunk_rows <= x.shape[0]):
            raise ValueError("Invalid native QKV chunk size")
        self.pack_backend = os.environ.get(
            "APL_HTRC_QKV_PACK_BACKEND", "torch").strip().lower()
        if self.pack_backend not in ("torch", "triton"):
            raise ValueError("Unknown compact native QKV packing backend")
        if self.pack_backend == "triton" and (
            x.device.type != "cuda" or rope_freqs is None
            or normalize_chunk is not None
        ):
            # Native Comfy fused RMSNorm/RoPE must normalize Q/K IN PLACE
            # in the actual fused projection; a custom injected normalizer
            # or no-RoPE CPU test cannot prove this storage invariant.
            raise RuntimeError(
                "Triton QKV packing requires live CUDA native in-place RMSNorm/RoPE")
        self.tokens = int(x.shape[0])
        self.heads = int(attn.heads)
        if self.heads < 1 or int(attn.head_dim) != DIM:
            raise ValueError("Expected H3 with D=128")
        persistent_bytes = 3 * self.heads * self.tokens * (DIM + 4)
        cap_mib = int(os.environ.get("APL_HTRC_COMPACT_STAGE_CAP_MIB", "1536"))
        if cap_mib <= 0 or persistent_bytes > cap_mib * (1 << 20):
            raise RuntimeError(
                f"Compact native QKV stage requires {persistent_bytes / (1 << 20):.1f} MiB "
                f"but cap is {cap_mib} MiB")
        if x.is_cuda:
            free, _ = torch.cuda.mem_get_info(x.device)
            # Packed stage + at most one fused projection chunk and quant work.
            if int(free) < persistent_bytes + 384 * (1 << 20):
                raise RuntimeError("Compact native QKV stage fails free-VRAM admission")
        self.device = x.device
        self.dtype = x.dtype
        self.chunk_rows = chunk_rows
        # [Q,K,V; head; token; D], symmetric INT8, 1 byte/component.
        self.packed = torch.empty((3, self.heads, self.tokens, DIM),
                                  dtype=torch.int8, device=x.device)
        # Each token/head/component has one FP32 scale, no clipping on decode.
        self.scales = torch.empty((3, self.heads, self.tokens),
                                  dtype=torch.float32, device=x.device)
        self.projection_chunks = 0
        self.total_projection_rows = 0
        self.build_gpu_ms = None
        self.build_wall_seconds = None
        self._build(attn, x, rope_freqs, normalize_chunk=normalize_chunk)

    @property
    def persistent_bytes(self):
        return (self.packed.numel() * self.packed.element_size() +
                self.scales.numel() * self.scales.element_size())

    @staticmethod
    def _encode_into(values, packed_dest, scale_dest):
        """Encode [rows,H,128] BF16 into [H,rows,128] and FP32 scales."""
        fp = values.float()
        scale = fp.abs().amax(dim=-1).clamp_min_(1e-8).div_(127.0)
        fp.div_(scale.unsqueeze(-1)).round_().clamp_(-127, 127)
        packed_dest.copy_(fp.to(torch.int8).permute(1, 0, 2))
        scale_dest.copy_(scale.transpose(0, 1))

    def _build(self, attn, x, rope_freqs, *, normalize_chunk):
        if normalize_chunk is None:
            # Import only the actual live H3 primitive, no quantized-weight
            # transforms. Resolve RMSNorm weights once just like stock Comfy.
            from .htrc_flat_h3_native_projection import normalized_native_group_chunk
            if rope_freqs is not None:
                import comfy.model_management as mm
                import comfy.model_prefetch as prefetch
                import comfy.quant_ops as quant_ops
                if bool(mm.in_training):
                    raise RuntimeError("One-pass native compact QKV refuses training")
                with prefetch.pause_malloc_graph():
                    qw = mm.cast_to(attn.q_norm.weight, device=x.device)
                    kw = mm.cast_to(attn.k_norm.weight, device=x.device)
                def normalize_chunk(fused, freqs):
                    return normalized_native_group_chunk(
                        attn, fused, freqs, begin=0, end=self.heads,
                        qw=qw, kw=kw,
                        rms_rope_inplace=quant_ops.ck.rms_rope_split_half_)
            else:
                def normalize_chunk(fused, freqs):
                    return normalized_native_group_chunk(
                        attn, fused, None, begin=0, end=self.heads)

        pack_normalized_fused = None
        if self.pack_backend == "triton":
            # Lazy import: torch pack remains the default, and CPU tests
            # never require Triton/CUDA.
            from .htrc_flat_h3_pack_triton import pack_normalized_fused
        wall = time.perf_counter()
        start = end = None
        if x.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        for begin in range(0, self.tokens, self.chunk_rows):
            stop = min(begin + self.chunk_rows, self.tokens)
            fused = attn.qkv_proj(x[begin:stop])
            if (not isinstance(fused, torch.Tensor) or
                tuple(fused.shape) != (stop-begin, 3*self.heads*DIM) or
                fused.dtype != x.dtype or fused.device != x.device):
                raise RuntimeError("Native QKV projector output contract changed")
            q, k, v = normalize_chunk(
                fused,
                rope_freqs[:, begin:stop] if rope_freqs is not None else None)
            if pack_normalized_fused is None:
                for part, value in enumerate((q, k, v)):
                    self._encode_into(
                        value, self.packed[part, :, begin:stop, :],
                        self.scales[part, :, begin:stop])
            else:
                # Unlike generic torch fallback, the fused pack kernel
                # reads directly from the entire original contiguous
                # projection tensor AFTER Q/K have been normalized.
                # Verify that the installed Comfy fused norm kept Q/K/V
                # as views of the same original data. Never silently pack
                # un-normalized BF16 Q/K if a future Comfy version changes
                # its in-place kernel.
                part_size = self.heads * DIM * fused.element_size()
                expected_ptrs = tuple(
                    fused.data_ptr() + i * part_size for i in range(3))
                if (not fused.is_contiguous()
                    or any(
                        value.data_ptr() != expected
                        for value, expected in zip((q, k, v), expected_ptrs)
                    )):
                    raise RuntimeError(
                        "Native RMSNorm/RoPE changed fused QKV aliasing; "
                        "Triton fast pack refused")
                pack_normalized_fused(
                    fused, self.packed, self.scales,
                    start=begin, tokens=self.tokens, heads=self.heads)
            self.projection_chunks += 1
            self.total_projection_rows += stop-begin
            del fused, q, k, v
        if end is not None:
            end.record()
            end.synchronize()
            self.build_gpu_ms = float(start.elapsed_time(end))
        self.build_wall_seconds = time.perf_counter() - wall

    def _decode(self, part, begin, end, start, stop, destination):
        """Decode directly into graph-owned BF16 buffers, no full-head BF16 cache."""
        if (not 0 <= part < 3 or not 0 <= begin < end <= self.heads or
            not 0 <= start < stop <= self.tokens or
            not isinstance(destination, torch.Tensor) or
            tuple(destination.shape) != (end-begin, stop-start, DIM) or
            destination.dtype != self.dtype or destination.device != self.device):
            raise ValueError("Invalid compact H3 group destination")
        source = self.packed[part, begin:end, start:stop]
        scales = self.scales[part, begin:end, start:stop].unsqueeze(-1)
        # Single fused TensorIterator GPU op, writes BF16 directly to the
        # graph-owned storage; no BF16/FP32 all-head activation allocation.
        torch.mul(source, scales, out=destination)

    def group(self, begin, end):
        if not 0 <= begin < end <= self.heads:
            raise ValueError("Invalid compact QKV head-group range")
        return CompactGroupProvider(self, begin, end)

    def summary(self):
        return {
            "projection_mode": "native_fused_once_compact_per_token_head_int8",
            "full_fused_qkv_passes_per_attention": 1,
            "projection_chunk_calls_per_attention": self.projection_chunks,
            "projection_rows": self.total_projection_rows,
            "persistent_packed_bytes": self.packed.numel()*self.packed.element_size(),
            "persistent_fp32_scale_bytes": self.scales.numel()*self.scales.element_size(),
            "persistent_total_bytes": self.persistent_bytes,
            "build_gpu_ms": self.build_gpu_ms,
            "build_wall_seconds": self.build_wall_seconds,
            "activation_quantization": "signed_int8_per_head_per_token_fp32_scale",
            "activation_packing_backend": self.pack_backend,
            "fused_pack_without_fp32_activation_staging": self.pack_backend == "triton",
            "quantized_model_weights_unmodified": True,
            "output_parity": "UNTESTED",
        }


class CompactGroupProvider:
    """Borrow compact Q/K/V by group without projecting QKV again."""

    def __init__(self, all_heads, begin, end):
        self.all_heads = all_heads
        self.begin = int(begin)
        self.end = int(end)

    def fill_kv(self, k, v):
        h = self.end-self.begin
        n = self.all_heads.tokens
        for stage in (k, v):
            if (not isinstance(stage, torch.Tensor) or
                tuple(stage.shape) != (1,h,n,DIM)):
                raise ValueError("Invalid graph-owned compact K/V buffer")
        self.all_heads._decode(1,self.begin,self.end,0,n,k[0])
        self.all_heads._decode(2,self.begin,self.end,0,n,v[0])

    def fill_q(self, start, end, stage):
        real = end-start
        h = self.end-self.begin
        if (not isinstance(stage,torch.Tensor) or stage.ndim != 3 or
            stage.shape[0] != h or stage.shape[1] < real or
            stage.shape[2] != DIM or not 0 <= start < end <= self.all_heads.tokens):
            raise ValueError("Invalid graph-owned compact Q window")
        self.all_heads._decode(0,self.begin,self.end,start,end,
                               stage[:,:real,:])
        if real < stage.shape[1]:
            stage[:,real:,:].copy_(
                stage[:,real-1:real,:].expand(-1,stage.shape[1]-real,-1))
