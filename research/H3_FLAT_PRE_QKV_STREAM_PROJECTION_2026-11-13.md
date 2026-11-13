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

## RTX 5070 Ti second A/B: memory-for-speed Pareto frontier

User-provided `flat_pre_qkv_stream_ab(1).json` was produced by the
four-arm code at `9ba03242e531fc530ac15da609dde26bad630751`.
Both lengths and all four arms report `MEASURED`, with three
warmed GPU timings; same synthetic dense BF16 X and weights,
56 genuine distinct heads, 8-head Graph sessions, H3-like
fused channel order, 4096-row K/V chunks and 768 MiB hard
Graph-extra cap. No real H3 checkpoint, NVFP4, RMSNorm, RoPE,
video, out_proj, or cross-block offload effect is measured.

| Arm | 15,275 GPU median ms | 18,723 GPU median ms | 15,275 warm peak allocated MiB | 18,723 warm peak allocated MiB |
|---|---:|---:|---:|---:|
| full_qkv | **71.953** | **97.489** | 1149.299 | 1331.361 |
| streamed_kv_lazy_q | 75.765 | 101.675 | 618.811 | 679.354 |
| direct_kv_lazy_q | 73.564 | 99.709 | **573.036** | **620.217** |
| direct_kv_group_q | **72.870** | **98.907** | 603.036 | 656.785 |

Relative to full_qkv:

- `direct_kv_group_q` costs **+0.917 ms / +1.419 ms**
  (+1.275% / +1.455%) and saves **546.263 MiB /
  674.576 MiB** warm peak allocated. This is the fastest
  streamed option (best current latency/memory balance).
- `direct_kv_lazy_q` costs **+1.611 ms / +2.221 ms**
  (+2.239% / +2.278%) and saves **576.263 MiB /
  711.144 MiB**. It saves a further 30 / 36.568 MiB
  against group-Q but is 0.694 / 0.802 ms slower.
- The original `streamed_kv_lazy_q` is dominated by
  `direct_kv_lazy_q` on both timings and allocated peak.
  Preserve it for attribution only, not as a proposed mode.
- All three streamed variants are bit-equal to the full-QKV
  control **on 64 sampled query positions** at each length,
  with `max_abs=0` and `relative_l2_percent=0`. This
  is not complete output or real-model parity.

The extra saved VRAM is **not a speedup by itself**. It can
improve total H3 denoise throughput if it is actually
reinvested to prevent GPU/CPU model offload, reduce
synchronization, increase resident working buffers or
permit a better kernel schedule. This is a
**system-level hypothesis**; the synthetic benchmark
cannot validate such an improvement. In particular,
`hidden_width=1024` and dense BF16 weights are not the
real H3 NVFP4 layout, and the measured timings exclude
the model's 49-block pipeline, all other allocations
and the actual H3 out projection.

### Disposition and next gate

Keep two opt-in research references:
`direct_kv_group_q` (balanced) and `direct_kv_lazy_q`
(minimum peak VRAM). Do not keep optimizing synthetic
head grouping for differences under a millisecond.
Do not promote either to production; existing legacy
and fail-closed parity gates remain unchanged.

Next **meaningful** gate is an actual H3 pre-QKV
provider, matching `sol_kitchen_chunked`'s model
interception and ComfyUI layer hooks. It must honor the
real `self.qkv_proj` (including NVFP4 packed weight
semantics), Q/K norm, RoPE, input/output types, and
must NOT silently replace quantized projection by
BF16 F.linear. For HTRC Flat, global K/V visibility
is still required across query windows: either
selective channel projection or a bounded staging
representation must be shown numerically and
performance-safe. Measure **whole denoise-step**
GPU+wall time, peak resident VRAM, offload transfers,
and full-output/visual parity. An advantage in total
inference at the same or lower VRAM is the target;
a local 1%-2% latency penalty is not in itself a
reason to abandon a >500 MiB memory saving.

This commit records the user-supplied measurements
only. CPU/GPU tests and actual H3 integration are
**UNRUN** in this commit.

## Native H3 QKV: real checkpoint integration gate (research-only)

**Exact upstream anchor**: Comfy-Org/ComfyUI
`comfy/ldm/minimax/model.py` blob
`f2ab28114bd297c88f081782211baf67fee3a961`.
H3 Attention.forward invokes
`q,k,v=self.qkv_proj(x).split(heads*128)` followed by
`comfy.quant_ops.ck.rms_rope_split_half_` with
`comfy.model_management.cast_to(q_norm.weight/k_norm.weight)`
and `rot_dim=rope_freqs.shape[-3]*2`, then original
`self.out_proj`. The verified ComfyNodes
`h3/kitchen_chunked_sol.py` uses the original projector on
4096-row chunks and `model_prefetch.pause_malloc_graph()`
for Q/K norm weight casts.

HTRC now contains `h3/htrc_flat_h3_native_projection.py`:

1. Invoke the actual `attn.qkv_proj(x[start:stop])` on the
   currently installed H3 model. This retains CUDA NVFP4,
   INT8/ConvRot, FP8, BF16, Comfy weight hooks, and per-call
   prefetched model weight semantics. It does NOT inspect,
   dequantize or repackage the checkpoint.
2. Split real native `[Q(all),K(all),V(all)]` projected chunks.
   Apply the same fused in-place Q/K RMSNorm+RoPE using the
   matching slice `rope_freqs[:,start:stop]` and the
   current model Q/K norm weights.
3. Copy ONLY the selected 8 genuine heads into
   Graph-owned K/V slots plus a group Q stage, respecting
   stable CUDA Graph addresses and refreshing on every call.
4. Replay identical HTRC Flat kernels; produce
   `[N,hidden]` through the original model `out_proj`.
   This is an explicit `compare_only` research path,
   never a default or a validated denoise backend.

**Known computational blocker:** the original quantized
fused `qkv_proj` still emits ALL 56 heads per chunk.
The research adapter performs one full projection for
each eight-head group: **7 complete projection passes**
per 56-head attention call, compared to one pass in
the usual full-QKV path. Real NVFP4 weight-row slicing
without dequantization or patch bypass is NOT proven.
Do not call the adapter an optimized native QKV kernel
or promise it beats the existing full projection.
This first slice establishes actual model compatibility
and numerical/VRAM behavior before selecting a
one-pass or channel-selective implementation.

The Graph function adds `research_source_kind="native_h3"`
and `research_group_size=8` to its compare-only gate
for accurate telemetry and explicit per-call session
selection. The default production `legacy`,
post-capture `MAX_EXTRA_MIB=768`, minimum free-VRAM
gate, original eager Flat strict parity and fixed Flat
math remain untouched. The grouping override bypasses
only a conservative capacity estimate, never actual
allocator/free-memory guards.

ComfyNodes has a *separate* explicit
`htrc_native_pre_qkv_probe` attention profile:
it wraps a single selected block for its first eligible
call, first executes and **returns the original**
`Attention.forward` result, then runs an isolated
native HTRC chunked projection canary on one head group.
It samples 64 contiguous positions against a SECOND
genuine model projection call and records RMS-RoPE Q/K,
V, numerical error, timing and allocator peaks in
`D:\ComfyOutput\H3\math_probe\h3_native_pre_qkv_canary.json`.
A different GEMM batch can give nonzero numerical error;
bitwise equality is recorded, not assumed. It never
overwrites the actual denoiser output, and restores its
forward patch after ModelPatcher cleanup/clone lifecycle.

An optional `APL_H3_NATIVE_QKV_CANARY_FULL_GRAPH=1`
enables the COMPLETE research Graph output from that one
block, measuring total local time and memory; it still
returns the unmodified original result and recomputes
fused QKV seven times. This is a costly, explicitly
opted-in canary, NOT a full generation benchmark.

**Test status:** CPU native chunk/position-slicing tests
and ComfyNodes probe lifecycle contract tests were added
but **UNRUN** in the developer environment. User's
RTX 5070 Ti has NOT executed native quantized QKV,
real RMS-RoPE, pre-QKV Graph, or a complete denoise.
No real VRAM saving or speedup is claimed by this slice.

**Next performance question after native canary:** can
we build a channel-selective packed-NVFP4 projector or
one-pass K/V staging without seven repeated fused GEMMs?
Only then compare the whole 50-block H3 denoise,
model residency, offload traffic, latency and quality
against the current optimized baseline.
