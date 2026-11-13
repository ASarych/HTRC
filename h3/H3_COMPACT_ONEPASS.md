# Research: native MiniMax-H3 one-pass compact QKV staging

**Scope:** HTRC Flat Graph research only. This is not production H3 attention,
not a tested quality replacement, and not a validated end-to-end speedup.
The default native-H3 Graph strategy remains `repeated_group`.

## Why

The first real 2026-11-13 ComfyNodes/NVFP4 canary measured a **single 8-head
group's** genuine 4096-row chunked fused `qkv_proj` on 40,759 tokens:
10 projection calls and 26.680 ms GPU. The original stock full attention
(QKV + norm/RoPE + attention + output projection) took 522.210 ms, but
those timings must not be directly compared. The previous Graph prototype
called the *full* fused projection again for each of 7 head groups: 70
projector calls per H3 attention, before the Graph's compute.

## Implemented design

- `h3/htrc_flat_h3_compact_once.py` uses the original model's
  `attn.qkv_proj(x[start:stop])` **exactly once per 4096-row chunk**,
  then Comfy's native fused Q/K RMSNorm+split-half RoPE.
  Original QuantizedTensor/NVFP4 checkpoint weights are never touched,
  sliced, dequantized or repacked.
- All Q/K/V heads are compressed into a temporary per-forward
  `torch.int8 [3,H,N,128]` staging tensor with FP32 scales
  `[3,H,N]`, one per token/head/Q-or-K-or-V component.
  The staging is explicitly *lossy* and makes no visual or parity claim.
- A group provider reads only the 8 selected heads. The existing
  Graph-owned K/V tensors `[1,8,N,128]` and Q window tensors
  `[8,P,128]` are filled by direct `torch.mul(...,out=...)`
  dequantization in BF16. **No additional QKV projector call** is
  made when moving among head groups or query windows.
- The packed staging is constructed fresh for every model forward.
  Never cache activations across timesteps, models or prompts.
- `h3/htrc_flat_h3_native_projection.py` accepts explicit research
  `projection_strategy="compact_int8_onepass"` and still defaults
  to `"repeated_group"`. Only the existing `compare_only=True`
  HTRC Graph path may consume this new group provider.
- An explicit staging cap `APL_HTRC_COMPACT_STAGE_CAP_MIB=1536`
  and a free-VRAM guard of packed stage plus 384 MiB fail closed.
  They do not override the Graph's separate allocator/headroom gates.

## Estimated storage at measured shape (N=40,759, H=56, D=128)

| Storage | Computed MiB |
|---|---:|
| All Q/K/V as BF16 | 1,671.756 |
| All Q/K/V as signed INT8 | 835.878 |
| FP32 per-row/head/component scales | 26.121 |
| **Persistent compact all-head stage** | **861.999** |
| Legacy single 8-head BF16 Q/K/V staging | 238.822 |
| Temporary 4096-row full fused BF16 output | 168.000 |

These are exact tensor byte counts, **not measured peak VRAM**.
Temporary FP32 packing work, graph-owned K/V buffers, graph pools,
attention output, quant kernels and transfer/allocator effects still
consume GPU memory. The compact stage is larger than a single group,
but avoids six additional *whole-model fused QKV passes*.

## Verification plan

1. CPU-only `python -B tools/test_h3_compact_onepass_cpu.py`:
   actual pack/decode implementations; exact projection call counts;
   two head groups; padded Q; fresh per-forward staging; cap failure;
   Graph default/policy structural checks.
2. ComfyNodes read-only canary: one stock H3 block and original denoise
   output, compact stage built once and all seven groups decoded.
   Record **build GPU ms**, **seven-group decode GPU ms**, **incremental
   peak allocated/reserved VRAM**, and persistent stage bytes in the
   same `h3_native_pre_qkv_canary.json`. Do **not** run Graph, generate
   a second video, or spend cycles on the earlier 4-5% comparison yet.
3. Only if measurements justify the storage tradeoff, profile faster
   fused packing/unpacking kernels. Then test full Graph in a separate
   opt-in research gate and **later** judge generated-video quality.
   Do not promote as a production replacement without those steps.

## State

CPU synthetic prototype (same pack/unpack logic) passed 3 tests on
a non-CUDA runner. Repository exact-source CPU preflight and actual
RTX 5070 Ti GPU timings have not yet run on this new HEAD.
Historical stock full-N-to-short-GEMM numerical mismatch is not a
blocking quality verdict and is not re-tested by this speed gate.
