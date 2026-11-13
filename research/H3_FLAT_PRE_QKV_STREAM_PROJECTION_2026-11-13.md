# H3 Flat — streaming pre-QKV projection, first bounded research gate

Date: 2026-11-13. Experimental HTRC branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
No production or ComfyNodes model interception changed.

## Why this experiment

The existing `h3/htrc_flat_directflow_live.py` starts AFTER H3
has materialized all 56-head Q/K/V tensors. Its grouped CUDA Graph
stages another K/V copy per resident graph session. The APL
`h3/kitchen_chunked_sol.py` implementation intercepts
`Attention.forward` BEFORE `qkv_proj` and consumes fused projection
in 4096-row chunks, so full original BF16 Q/K/V never co-reside.

The goal is to test whether HTRC can also accept a producer which
does not first create a full 56-head QKV output. This is a **transport
and lifetime experiment**, not a replacement for Flat math.

## Implemented first slice

- Added `h3/htrc_flat_pre_qkv_stream.py`. It consumes a BF16
  *dense* fused projection weight with H3's channel order
  `[Q(all heads),K(all heads),V(all heads)]`.
- For each 8-head group, select ONLY that group's Q, K and V output
  channels. The Q weight remains a view; K/V weight shards are
  contiguous per group. The projection does NOT compute unrelated
  head channels again.
- Project K/V for the *full token sequence* in 4096-row input chunks,
  directly filling 8-head BF16 K/V buffers. Flat needs global K/V
  for each query window; pretending both could be discarded after
  4096 rows would break the algorithm.
- Do not create a full-sequence Q. Instead, project Q from X on
  demand immediately before each query-window Graph replay and copy
  into the **existing stable Graph-owned Q stage**, including
  last-window row padding. This avoids a second full Q activation.
- The existing Graph function accepts an optional `query_provider`.
  It is **rejected** unless `compare_only=True`. Legacy production
  input handling, first-real-call eager exact-parity gate and
  independent 768 MiB post-capture Graph gate are unchanged.
- The Group K/V projection currently writes a separate K/V buffer,
  followed by the existing one-time copy into Graph-owned K/V
  stage. Eliminating this particular final copy is a distinct
  pending optimization, not falsely marked complete.

`tools/benchmark_h3_pre_qkv_stream_ab.py` compares two arms on
the SAME synthetic X and BF16 weights:

1. `full_qkv`: one full fused projection, normal Graph Flat
   on all 56 heads with the 8-head grouping.
2. `streamed_kv_lazy_q`: one group K/V projection at a time,
   per-window lazy Q projection and the identical Graph Flat
   fixed math.

The benchmark includes the **entire QKV projection** cost in GPU
event and synchronized wall medians, plus Graph staging, K8,
window Tile/Merge, and BTHD output assembly; it excludes graph
capture from warmed timings but records cold wall/peak separately.
Only 64 selected final output rows are saved on CPU for numerical
comparison; no whole GPU output is retained between arms.
Per-arm preliminary Graph reports live only in a temporary
directory. Exactly ONE persistent aggregate JSON is produced.

## Bounded memory hypothesis, NOT measured memory

At 56 heads, dim=128, BF16:
- 15,275 tokens: full fused QKV 626.5 MiB, eight-head fused
  QKV equivalent 89.5 MiB.
- 18,723 tokens: full fused QKV 767.9 MiB, eight-head fused
  QKV equivalent 109.7 MiB.
- 40,759 tokens: full fused QKV 1,671.8 MiB, eight-head fused
  QKV equivalent 238.8 MiB.
- 82,000 tokens: full fused QKV 3,363.3 MiB, eight-head fused
  QKV equivalent 480.5 MiB.

The actual candidate never stores the full grouped Q, only K/V
plus a per-window Q, producer scratch, input X/weights and Graph
session buffers. Allocator peak deltas will depend on CUDA Graph,
cuBLAS and live buffers; the theoretical size ratio is not
a promised peak-VRAM reduction.

## Essential limits

**This first slice is NOT a model-compatible H3 provider.**
The synthetic BF16 dense projection deliberately omits H3 Q/K
RMSNorm, RoPE, other model hooks and NVFP4 quantized projection.
The actual checkpoint's quantization and model hook behavior
must be implemented and checked independently before any model
integration. A group-sliced BF16 `F.linear` is NOT equivalent to
calling H3's existing quantized `self.qkv_proj` in production.
Different GEMM shapes may produce BF16 rounding differences;
the benchmark measures numerical differences rather than
assuming strict identity.

The Graph code still trusts compare-only synthetic K/V and does
not claim full-video consistency, real model latency or production
eligibility. No changes to fixed exact leaves, K8 quantizer, Flat
routing, softmax, graph cap, default backend or ComfyNodes.

## One local check and one JSON

Run on Windows from the exact HTRC experimental checkout
(using the Python already validated for local Torch+Triton):

    & $py -B -m unittest discover -s tools -p test_h3_pre_qkv_stream.py -v
    & $py -B tools\benchmark_h3_pre_qkv_stream_ab.py --execute --tokens 15275,18723 --heads 56 --group-heads 8 --hidden-width 1024 --chunk-rows 4096 --mode balanced --tiles 16 --max-extra-mib 768 --rounds 3

Single persistent output:
`D:\ComfyOutput\H3\math_probe\flat_pre_qkv_stream_ab.json`.

`--hidden-width 1024` is deliberately a *synthetic* inexpensive
input projection width, not proof of H3's true hidden width; it
makes this first architecture gate less costly. A later probe may
use the actual width only after quantized projection and
normalization are supported.

## Status at initial commit

A separate local CPU BF16 `F.linear` shard/chunk arithmetic
smoke test passed on three small shapes, including group tails.
The newly committed repository unittest suite is **UNRUN**
on the exact checkout; CUDA Graph numerical and end-to-end
timing A/B are **UNRUN** pending user's RTX 5070 Ti.
One JSON is the required output. No promotion to production.
