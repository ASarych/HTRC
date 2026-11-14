# HTRC Tile32: RB=2 row-ownership structural experiment

Date: 2026-11-14. Repository: ASarych/HTRC.
Branch: experiment/flat-memory-to-speed-balanced-20261113.
Base exact HEAD: 2f9726a5f6c6cd9da48cc1f36e2a8a53b9591476.

## Motivation and existing results

The NVIDIA RTX 5070 Ti (SM120, 16 GiB) materialized
Balanced/512 profile showed that the existing Tile32
kernel has 255 registers/thread, about 49.5 KiB
allocated shared memory/CTA, ~15.57% achieved occupancy
and a two-CTA/SM resource cap on its four-warp launches.
This is NOT proof that merely increasing occupancy will
increase attention throughput.

Previous opt-in compiler-knob tests were both negative:
- late_q: exact partial numeric PASS; shared went from
  49,664 to 41,472 B, 255 registers persisted,
  mean paired time was about 1.05% slower. NO_GO.
- stage1: exact partial numeric PASS; 255 to 219
  registers and 49,664 to 40,960 B shared, yet still
  only two CTAs/SM. Two independent executions found
  ~20.40% and ~20.26% extra Tile time. NO_GO.

Those experiments are CLOSED. This slice does not
re-run or combine them.

## Narrow mechanism, no new CUDA arithmetic

The Triton _tile_kernel ALREADY implements RB as a
compile-time query-row partitioning factor:

    RT = T // RB
    r = tl.program_id(2) * RT + tl.arange(0, RT)

RB1: one 64-row CTA per (tile, leaf split).
RB2: two disjoint 32-row CTAs per (tile, leaf split).
Every row and split writes its original PM, PL and PACC
offset exactly once; total output shapes are unchanged.
The four-warp, two-stage Tensor Core loops, BN64, leaf
selection, exact token budget, QKV representation,
near/far softmax arithmetic and Merge remain unchanged.

The launch grid changes only for explicit research
candidate RB2:
- CONTROL: (heads * 32, 4, 1) = (128,4,1),
  512 CTAs for one 2,048-query-row window.
- RB2: (heads * 32, 4, 2) = (128,4,2),
  1,024 CTAs for the SAME query rows and outputs.

This deliberately doubles CTAs and duplicates some
per-tile/leaf loads. It might reduce each CTA's active
accumulator/operand footprint, allowing higher residency
or better issue latency, OR it might lose due to extra
work. An improved resource count is NOT a speed result.

Historical note: a single-head sweep earlier listed a
tile_rb=2 variant. This new gate is DISTINCT: flattened
four-head materialized Balanced/512 CUDA Graph Tile,
two independent QKV banks, bit-exact PM/PL/PACC
comparison, and four independent Graph captures.

## Opt-in implementation and strict refusal scope

Existing h3/htrc_flat_head_batch.launch_prepared_tile_batch
has an optional research_resource_variant keyword.
Omission still compiles RB=1, LATE_NEAR_Q=0, and the
normal two Triton stages: production is unchanged.
This slice adds only variant 'rb2' to the existing
research allowlist, with the same bounded shape gate:
- 15,275 QKV tokens; 15,264 full 16-token K leaves;
- exactly four heads, 32 query tiles, 64 rows/tile,
  512 exact leaves, four splits, group1;
- materialized Vbar, INT8 G/K, BF16 dot, BN_NEAR=64,
  BN_FAR=64, four warps, two stages;
- no fused Vbar, no streamed Vbar, no RB4, no
  late_q + RB2, no stage1 + RB2.

Only the research branch gets this opt-in. Default
callers do not pass it; no change to the underlying
h3/htrc_flat_triton._tile_kernel math.

## Safety and measurement

Run only after ComfyUI/Blender are closed. The existing
PowerShell runner uses a disposable exact-HEAD checkout,
runs the repository's CPU gate BEFORE GPU and leaves
the installed ComfyUI tree and main branch untouched.
CPU checks include disjoint row ownership for RB1
and RB2 and fixed-shape/CLI/ABBA source contracts.

The exact 15,275-token, four-distinct-head,
Balanced/512 input is generated twice with
independent seeded BF16 QKV banks. The existing
unmodified complete attention first-call must
pass eager-vs-Graph bit parity. Bank B is rechecked
against independently run eager Flat.

Against exactly identical prepared input states,
RB1 and RB2 PM/PL/PACC must be BIT-EXACT in BOTH
banks; any difference FAILS CLOSED. No soft numerical
tolerance. The kernel compiler's registers/thread,
allocated shared bytes/CTA and compiler spill count
are captured for each specialization. A Python
resource upper bound is included but does not claim
measured occupancy.

Four independent Tile-only CUDA Graph captures
in CONTROL/RB2/RB2/CONTROL order each get five
warmup replays and seven timed CUDA events by
default. Only if both candidate capture block
medians are below both control blocks AND the
median-of-block-median ratio shows >=5% improvement
does the report mark PROMISING_TILE_ONLY. Even
then full-attention output parity, independently
captured full operator timing and model-resident
VRAM remain additional UNRUN gates.

Single report (the existing ONE JSON, overwrite
on user-requested rerun only):

    D:\ComfyOutput\H3\math_probe\flat_tile_resource_abba.json

Source script: tools/benchmark_h3_tile_resource_abba.py
Exact runner: tools/run_h3_tile_resource_abba_safe.ps1
CLI: -Candidate rb2 -Rounds 7 -ExactHead <40-char SHA>

The old late_q and stage1 results remain documented
in H3_TILE_RESOURCE_ABBA_ADMISSION_2026-11-14.md.
No extra CSV or duplicate report file is created.

## Status at implementation commit

CPU helper tests: 10 PASS, 1 SKIP locally (full HTRC
source absent in local isolated Python directory);
the complete source contract runs in Windows exact
checkout before GPU. New RB2 GPU numeric/Graph ABBA:
UNRUN. H3 checkpoint, video, full-attention timing,
production promotion: UNRUN / PROHIBITED.

No acceleration or accuracy change is claimed.


## 2026-11-15 — user RTX 5070 Ti RB2 A/B/B/A result: NO_GO

Original uploaded `flat_tile_resource_abba(3).json`:
- Raw bytes: **4,544**, SHA-256
  `46a3511bcee974b65b17562d632f6b6d03ebf531f620d6c41fa8d3709bacf881`.
- Exact tested source HEAD:
  `16455263b3e41d000d7c591e4d1b640dabe585b5`.
- Schema `htrc-tile32-research-resource-abba-v1`;
  `status=PASS_NUMERIC_TILE_ONLY`, `completed=true`,
  `candidate=rb2`, `source_head` matches.
- Same benchmark setup as admission: RTX 5070 Ti SM120,
  15,275 tokens, 4 distinct BF16 QKV heads, balanced/512,
  32 tiles, 2,048 query rows in one measured Tile window.
  Two distinct QKV banks yielded bit-exact PM/PL/PACC
  versus control; first-call complete Graph/eager and
  bank-B eager parity gates both PASSED.
- The result is **one-window Tile-only Graph performance**,
  NOT full-attention throughput, model denoise, H3, VAE
  or video performance. No Nsight capture in this run.

Compiled resources and launch geometry (from JSON):

| Metric | Control RB1 | Candidate RB2 |
|---|---:|---:|
| Registers/thread | 255 | 140 |
| Shared memory, bytes/CTA | 49,664 | 37,376 |
| Compiler-reported spills | 0 | 0 |
| Four-warp CTAs for one window | 512 | 1,024 |
| Query rows per CTA | 64 | 32 |
| Register upper bound, CTA/SM | 2 | 3 |
| Shared upper bound, CTA/SM | 2 | 2 |
| Combined optimistic CTA/SM | 2 | 2 |

Registers reduced **45.10%**, shared memory **24.74%**.
Three simultaneously resident 128-thread CTAs would
require BOTH registers/thread <= 170 and shared memory
<= floor(102400/3) = **34,133 bytes/CTA** (ignoring
granularity). RB2 meets the register requirement but
misses the shared requirement by **3,243 bytes/CTA**;
thus **no third CTA residency was admitted** despite
the larger CTA grid.

Four independent Tile-only captures in control/RB2/RB2/
control order, each with 5 warmups and 7 GPU-event samples:

| Block | Variant | Median Tile time, ms |
|---|---|---:|
| 1 | control | 0.389216006 |
| 2 | RB2 | 0.516672015 |
| 3 | RB2 | 0.517216027 |
| 4 | control | 0.390175998 |

Control median-of-block-medians = **0.389696002 ms**;
RB2 = **0.516944021 ms**. RB2 **took 32.653% longer**,
ratio control/candidate **0.753845651**. Both RB2
capture blocks were slower than both control captures.
The configured >=5% speedup plus two-capture consistency
gate FAILS, even though all numerical parity gates PASS.
**Decision: RB2 = NO_GO_TILE_ONLY.**

Interpretation limits:
- Doubling CTAs without increasing residency, together
  with duplicated per-CTA operand/state loads, is
  compatible with this slowdown. The counterfactual
  source of slowdown has not been established by a
  stall/transaction breakdown.
- Reducing registers is not synonymous with increasing
  attained occupancy; the SMEM gate remains binding.
- This result does not support changing the default
  head-batch launcher, the existing materialized Tile32,
  Stats BT16, BN64, RB1, two stages or any production
  path. The research-only RB2 opt-in remains available
  but is NOT admitted for runtime promotion.

### Falsifiable possible next research direction — not implemented

There is a **specific new resource threshold**, unlike
a blind RB4/RB8 sweep: RB2 already meets the register
ceiling for 3 CTA/SM but needs 3,243 fewer shared bytes.
The earlier standalone `late_q` experiment at RB1
reduced shared memory by 8,192 B/CTA but did not
change register pressure and was slightly slower.
That SMEM reduction cannot be assumed additive at RB2.

A **RB2 + LATE_NEAR_Q** combined specialization would
be justified for **compile/resource admission only**:
first measure its actual registers, shared bytes and
spills, and check the 3-CTA hardware constraints.
Stop before any speed claim if it fails either resource
ceiling or exact PM/PL/PACC parity. Only if it crosses
the concrete 3-CTA resource boundary (and preserves
numerics) is an independent-capture Tile-only A/B/B/A
worth considering; then apply the SAME >=5%-faster and
both-block consistency gate. Even three resident CTAs
are not a guarantee of faster computation.

No new GPU execution, combined kernel, code test or
full-H3 experiment was performed while documenting
this measured RB2 result.
