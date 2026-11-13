# HTRC Flat long context: bounded Graph tiles and streaming score architecture

Date: 2026-11-13. Research-only slice, starting from
experiment/flat-directflow-real-h3-20261112 @
498cf8458b0a8e4dc31e1ea384b59a19d09bc1e7.
No ComfyNodes changes and no production/main promotion.

## Decision

The real six-step HyperFlow CUDA Graph ON/OFF A/B on 15,275
tokens showed a warmed denoise time of 48 s Graph vs 64 s eager, with all
six stored step latents and final latents bit-for-bit equal. But Sol was
faster in earlier, not fully matched, comparisons. Repeating the same A/B
is not the engineering priority. We need to determine whether Flat can
outscale Sol on 40k--80k tokens while remaining within RTX 5070 Ti 16 GiB
and retaining fixed leaf counts and output quality.

The balanced/512-leaf CUDA Graph experiment was ALREADY committed at
498cf845 and real H3 first-call gate passed. This slice does not redo it.

## What was implemented

1. **Opt-in 16/32/64 Graph-window geometry.**
   h3/htrc_flat_directflow_live.py previously required exactly 16 tiles.
   It now accepts tiles in (16,32,64), with fast=256 or balanced=512
   exact leaves, centre group 1 and BF16 QKV unchanged. The canonical
   default remains 16 tiles, eager remains default dispatch, and
   unsupported geometry fails closed. Reusing the Graph is keyed by
   mode, sequence geometry and tile size. The existing per-head
   allocation planner, graph-pool memory cap and exact first-real-call
   eager parity check remain in control. The JSON now records chosen
   tiles, query windows and total per-attention graph replays.
   There is NO GPU claim for 32/64 tiles yet; only 16 tiles has a real
   completed-video evidence base.
2. **No-GPU source-derived scaling model.**
   tools/estimate_h3_flat_long_context.py mirrors the conservative
   per-head scratch planner and counts centre/leaf pairs, G/VB/LZ/SG
   writes, rowmax/tile rereads, BF16 QKV/output volume, group caps and
   graph replay counts by candidate window size. Its scope is
   analytical; it cannot predict an actual Flat-vs-Sol crossover,
   model-resident admission, graph cache overhead or wall clock speed.
3. **Isolated first streaming-score kernel slice** in
   h3/htrc_flat_triton.py: new compile-time SCORE_ONLY=1 specialization
   skips loading V, computing VB and writing VB/SV, but calculates and
   stores exactly the original G/LZ/SG. Default SCORE_ONLY=0 follows
   the pre-existing arithmetic and writes everything; no live caller
   sets SCORE_ONLY=1. The standalone
   tools/benchmark_h3_flat_score_only.py compares the same kernel in
   both modes on nonzero distinct-head synthetic Q/K/V, requires
   bitwise G/LZ/SG/SEL/IDX/EXCL and a never-written VB sentinel,
   times only the stats launch and writes one JSON.
   It does NOT implement the later far Tile recomputation or output.
   The candidate can be slower or numerically fail: do not promote it.
4. **CPU-only contracts**, tools/test_estimate_h3_flat_long_context.py,
   cover the 15k/18k/40k/82k/223k token geometries, two budgets,
   quadratic stats traffic, fast vs balanced, fail-closed planner,
   unchanged default specialization, GPU benchmark syntax and
   invalid inputs. These do not exercise CUDA Graph or Triton CUDA.

## Equations (56 heads, dim 128, fixed centre group 1)

For N query tokens:
- centre count C = ceil(N / 64);
- full key-leaf count L = floor(N / 16);
- score/stat pairs = 56 C L, **quadratic in N**;
- G = 128 B int8, VB = 256 B bf16, LZ = 4 B fp32,
  SG = 4 B fp32: **392 bytes per pair written**;
- rowmax rereads G/LZ/SG (~136 B/pair), far tile reads
  G/VB/LZ/SG (~392 B/pair), plus score/exclusion writes.
- Four materialized BF16 Q/K/V + attention output occupy
  4 * 56 * N * 128 * 2 bytes independent of Graph scratch.

| Tokens N | Full-head G/VB/LZ/SG logical write | Four BF16 QKV+output |
|---:|---:|---:|
| 18,723 | ~7.0 GiB | ~1.00 GiB |
| 40,759 | ~33.2 GiB | ~2.18 GiB |
| 82,000 | ~134.3 GiB | ~4.38 GiB |
| 223,605 | ~998 GiB | ~11.94 GiB |

These are **logical bytes per one full attention call**, NOT
simultaneously live VRAM, allocator peaks, physical DRAM traffic
or measured GPU durations. In particular, the 998 GiB is NOT
an allocation attempt.

## Source-derived Graph replay counts (capacity planning only)

With conservative 1024 MiB workspace budget, 256 MiB reserve and
group cap at most 11; actual memory manager can reduce or reject:

| N | tiles16 | tiles32 | tiles64 |
|---:|---:|---:|---:|
| 18,723 | 114 | 80 | 70 |
| 40,759 | 480 | 380 | 280 |
| 82,000 | 1,539 | 1,148 | 1,176 |
| 223,605 | 12,264 | infeasible | infeasible |

"Infeasible" means the **per-group scratch planner** rejects
that tiles/budget combination; it is NOT a full-model admission result. Tiles32 reduces **launch/replay COUNT** for
82k at 1024 MiB but does not prove throughput gain: larger kernels
and smaller head groups may outweigh fewer Graph replays.

At 82k and a hypothetical 2048 MiB group budget, counts are
567, 492, 399 respectively. These numbers are NOT permission to
allocate 2 GiB with only 16 GiB total GPU memory.

## Next architecture: selection-only + far recomputation (PARTIAL RESEARCH CODE)

The truly consequential target is the quadratic G/VB materialization,
not merely the number of Graph dispatches.

Stage A is split into risk-controlled pieces. The new SCORE_ONLY=1
_stats_kernel specialization produces unchanged INT8 G, FP32 LZ/SG
and omits the BF16 VB write and dot. This directly saves
**256 bytes of VB writes per centre-leaf pair**, ~65.3% of the
392-byte G/VB/LZ/SG state write, *for Stage A only*. Existing
rowmax/top-k then obtains the same selection from G/LZ/SG.
The first isolated GPU benchmark is written, but **UNRUN**.
A later, more aggressive fused score kernel could eliminate G/SG
materialization as well, retaining only SEL plus EXCL. That later
kernel is NOT IMPLEMENTED. No performance saving is claimed yet.

Stage B (streaming far, NOT IMPLEMENTED): recompute VB (and as
necessary G/LZ) from the original K/V and MU for bounded leaf
batches, apply the original EXCL, and feed those short-lived
values directly into far-field Tile computation.
Preserve the original split order, online softmax, exact 256/512
near leaves and tail, and merge. No global G/VB tables.
Avoid re-evaluating the same leaf stats more than once per query
centre and split; the original Tile kernel can repeat chunks if
mispartitioned. Recalculation **increases tensor-core operations**;
the kernel can lose even if traffic falls. This is an experiment,
not a declared performance improvement.

Numerical gates:
- Comparison uses the *same* Q/K/V and fixed Flat modes (no Q quality).
- Stage A: test identical selected IDX/EXCL on genuine distinct heads,
  including near ties and partial final query window. Due to
  non-associative float reductions, selection set parity must be proven,
  not assumed.
- Stage B: gated G/VB/LZ/SG quant and full output. Pure transport
  changes demand bitwise output; reduction rearrangements need
  tight error metrics and unchanged near-leaf sets, plus real video.
- Compare full call GPU events, synchronized wall and VRAM with model
  weights resident. Never equate a score-only microkernel speed
  with complete attention or video speed.
- No data-dependent approximation, reduced leaves, Q coefficient,
  cross-step K/V caching or untested production integration.

The initial SCORE_ONLY stats specialization and its oracle A/B now
exist. It still writes G/LZ/SG and is not a fused score-only
kernel. Do not attempt a production two-pass swap until the
selection and subsequent far-recomputation gates pass.

## What 223k-token Full HD means

The earlier 362-frame run stopped during memory admission. At
223,605 tokens the four BF16 Q/K/V/output tensors alone are 11.94 GiB.
Prior model-streaming free VRAM was 9.58 GiB and HTRC preflight
required 17.31 GiB. Enlarging Graph windows **cannot** fix this.
QKV streaming/compression and model residency are separate
prerequisites. Sol chunked also needs its own large routing workspace.
The source model reports geometry only and does not claim Full HD is
supported.

## Status and next independent gate

- Existing 16-tile real-H3 Graph correctness & video: historical PASS.
- Newly enabled tiles32/tiles64: GPU numerical/performance UNRUN.
- Source-based capacity model: CPU suite status UNRUN until executed.
- SCORE_ONLY stats candidate: implemented, GPU A/B UNRUN.
- Fused SEL-only and streamed far Tile kernels: NOT IMPLEMENTED/UNRUN.
- Sol crossover and full-hd 223k memory feasibility: UNKNOWN.

No repeated full-video tests requested; first decide by measured GPU
kernel/attention scaling on fixed 40k/82k inputs when the new
streaming-score candidate exists. Preserve the original eager and
Graph paths as controls.


## Subsequent split-stream VB implementation (2026-11-13)

The isolated four-split VB producer/consumer, explicitly controlled by
APL_HTRC_DIRECTFLOW_VB_MODE=split_stream, is implemented on the follow-up
research branch experiment/flat-long-context-streamed-vb-20261113.
See H3_FLAT_SPLIT_STREAMED_VB_2026-11-13.md there. Old eager and
materialized Graph modes remain defaults and are untouched. GPU numerical
and speed tests for streamed mode remain UNRUN; this follow-up changes
peak VB scratch use, not total quadratic work or Full HD QKV materialization.
