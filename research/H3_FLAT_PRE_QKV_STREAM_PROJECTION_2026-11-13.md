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

## RTX 5070 Ti first pre-QKV A/B: measured, memory-positive, GPU-slower

The user supplied `flat_pre_qkv_stream_ab.json` from the initial
synthetic BF16 dense projection slice. It reports `MEASURED` on
15,275 and 18,723 tokens for both `full_qkv` and
`streamed_kv_lazy_q`, with 56 distinct heads, 8-head Graph
groups, 4096 input rows per projection chunk, 1024-wide
synthetic hidden state, balanced 512 exact leaves, 16 query
tiles and unchanged 768 MiB Graph-extra limit.

| Tokens | Full QKV GPU median | Streamed GPU median | Delta | Full absolute peak allocated | Stream absolute peak allocated |
|---|---:|---:|---:|---:|---:|
| 15,275 | 71.961 ms | 75.228 ms | +4.54% | 1,150.002 MiB | 618.811 MiB |
| 18,723 | 97.261 ms | 101.566 ms | +4.43% | 1,331.595 MiB | 679.066 MiB |

Peak allocated VRAM falls by 531.191 / 652.529 MiB
(46.19% / 49.00%). Warm absolute reserved peaks were
1178 -> 644 MiB and 1352 -> 714 MiB. These are PyTorch
allocator measures within the model-free synthetic runner,
not complete card usage with actual H3 weights. The cold
wall values should not be used for a speed conclusion because
the first arm includes one-time CUDA/JIT/capture effects.
Three warmed timing rounds on the same inputs provide only
a preliminary latency comparison.

The two arms are **bitwise equal on 64 sampled output query
positions** for both token lengths (max_abs=0, relative_L2=0).
This establishes identical sampled Flat math in the synthetic
BF16 projection experiment, not full output parity and NOT
real H3 quality/video parity. The real-model blockers remain
RMSNorm, RoPE, NVFP4 projection packing, model patch hooks,
model-resident VRAM and end-to-end video.

## Follow-up slice: eliminate group K/V copy; avoid repeated tiny Q GEMMs

The first streamed version projected a temporary 8-head K/V
and copied it to separate Graph-owned BF16 stages, then ran
one Q GEMM for each Graph query window. These are the two
remaining avoidable dataflow costs:

- `kv_mode=graph_owned` accepts a compare-only callback
  filling the **stable Graph-owned K/V** buffers directly with
  4096-row projected chunks. Stage K8/KS and exact tail are
  regenerated from these exact destination tensors on every
  call, not reused from the previous denoising step. A
  one-element expanded BF16 tensor provides only shape/device
  metadata to the runtime: both producers must overwrite the
  Graph stages before warmup/capture/replay. Production cannot
  pass this callback.
- `q_mode=group` projects all Q for only **one 8-head group**
  once and copies windows from that group buffer into the
  stable Graph Q stage. This adds one group-Q live tensor but
  removes the many small `F.linear` calls per window. It does
  NOT materialize 56-head full Q and does not change Flat math.

The expanded one-JSON A/B now measures four arms for the
same X/weights/heads: `full_qkv`,
`streamed_kv_lazy_q` (the original memory-positive but slower
control), `direct_kv_lazy_q` (isolate K/V copy elimination),
and `direct_kv_group_q` (test one Q GEMM per head group).
Each case records cold and warmed absolute/relative peak
memory, GPU event and synchronized wall medians, plus sampled
bitwise/relative-L2 comparison. Every arm is separately
fail-closed, with an independent Graph session reset.
`APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB=768` remains unchanged.

This is still **synthetic BF16 only**. It neither patches the
actual H3 model nor interprets a new QKV rounding difference
as proof of accepted video quality.

### Next single local run

From the exact updated research checkout, reuse the same
ComfyUI CUDA/Triton Python environment:

    & $py -B -m unittest discover -s tools -p test_h3_pre_qkv_stream.py -v
    & $py -B tools\benchmark_h3_pre_qkv_stream_ab.py --execute --tokens 15275,18723 --heads 56 --group-heads 8 --hidden-width 1024 --chunk-rows 4096 --mode balanced --tiles 16 --budget-mib 1024 --reserve-mib 256 --max-extra-mib 768 --rounds 3

Exactly ONE persistent JSON:
`D:\ComfyOutput\H3\math_probe\flat_pre_qkv_stream_ab.json`.

New CPU repository suite and all four GPU arms: **UNRUN**
in this new commit. No backend promotion.
