# HTRC Tile32 — isolated register / shared-memory A/B/B/A admission

Date: 2026-11-14. Research branch:
`experiment/flat-memory-to-speed-balanced-20261113`, based on
`d911060d6852c93d2b891f57fa2e317eadaff777`.

## Why now

The user's RTX 5070 Ti Nsight report
`flat_materialized_resource_ncu.json` was verified on
`01996061cec72ba252a3634892c9c642ab90c78c`: 12 CUDA Graph
kernel nodes and strict eager parity PASS. For balanced/512,
15,275 tokens and four heads, Tile under NCU replay had
422.96 us median, achieved occupancy 15.57%,
255 registers/thread, 50,688 shared bytes/CTA and a
register/SMEM occupancy limit of two resident CTAs.
DRAM peak percentage 21.73% and SM peak 64.54% do not prove
that HBM traffic is the main bottleneck.

To allow 3 instead of 2 128-thread Tile blocks on this SM120
(65536 registers/SM and 102400 shared bytes/SM), both resource
ceilings must be met *simultaneously*: <=170 registers/thread
and <=34133 shared bytes/CTA before allocation granularity.
Even 3 CTAs do not prove greater throughput.

## Independent research variants

The existing batch launcher has a new **optional** keyword
`research_resource_variant`. Omission produces the exact
old Tile kernel launch arguments and default compilation;
all original Graph/eager paths omit the keyword.

Two mutually exclusive candidates are allowed, with a hard
source-level shape gate pinned to four heads, 32 tiles,
64 rows/tile, 15,264 aligned keys, 512 exact leaves,
4 splits, BN=64, 4 warps, and original `num_stages=2`:

- `late_q`: only sets the existing `_tile_kernel`
  `LATE_NEAR_Q=1` compile-time option, which moves preparation
  of near-field Q operand to after the far-field loop. Still
  `num_stages=2` and `RB=1`.
- `stage1`: only changes `num_stages` from 2 to 1. The original
  `LATE_NEAR_Q=0` and `RB=1` remain unchanged.

No candidate modifies Stats, Rowmax, top-k, near-field leaf
selection, quality budget, BF16/INT8 representation, or the
near/far accumulation formula. This is **not** a fused Vbar,
streamed VB, or TMA/multicast candidate; those remain NO_GO.
The `late_q+stage1` combination and RB2 have NOT been admitted
until the separate one-knob cases are understood.

## GPU protocol (only after CPU gate)

`tools/benchmark_h3_tile_resource_abba.py` uses existing
`flow.graph_attention_bthd` with two independently generated
BF16 Q/K/V banks at 15,275 tokens, four heads and Tile32.
The first real synthetic call must pass mandatory complete
output eager/Graph parity. Bank B is checked again against
fresh independent eager Flat. Bank A and B must differ.

For one original full 32-tile window at each QKV bank,
call the same existing `batch.launch_prepared_tile_batch`
with control and selected candidate against identical
prepared leaf stats and K/V state. Compare **PM, PL and
PACC bit-for-bit** before any timing. Record compiler
`n_regs`, `n_spills`, and allocated shared bytes per CTA.
These values come from the returned Triton compiled-kernel
metadata, not fabricated from an analytic model.

Then, on bank B, perform four independent **Tile-only CUDA
Graph captures** in A/B/B/A order (control, candidate,
candidate, control). Each block has five warmup replays,
then seven warmed CUDA-event samples by default, with a
bit-exact PM/PL/PACC check before capture, after warmups,
and after timing. A candidate speed signal requires >=5%
median-of-block-medians improvement AND both independent
candidate capture-block medians strictly below both control
block medians.

**Important:** This does NOT time complete attention and
must not be compared directly to the earlier ~4.285-ms
complete-call control. It does not establish H3/video
speedup, VRAM residency, or production suitability. Even
if kernel-only ABBA wins, full-attention independent-capture
ABBA and model-resident memory acceptance are still needed.
New compiler variants can change arithmetic association;
any PM/PL/PACC mismatch is a strict FAIL, not a tolerance.

## Execution, fail-closed contract

`tools/run_h3_tile_resource_abba_safe.ps1` builds a disposable
detached exact-HEAD checkout, checks the user has closed
ComfyUI and Blender, runs CPU tests, and calls the one-JSON
GPU script. No GPU profiling is launched if CPU tests fail.
No H3 checkpoint, paid inference, video output or production
CUDA kernel changes occur.

Single output:
`D:\ComfyOutput\H3\math_probe\flat_tile_resource_abba.json`

The user may choose `-Candidate late_q` or `-Candidate stage1`
for an independent result; the fixed JSON path is intentionally
reused to avoid multiple duplicate files. Old successful
Nsight DRAM/resource reports remain untouched.

**At commit:** local CPU helper tests PASS; exact-repo source
gate and all GPU tests UNRUN until the user's Windows run.
Do not report speedup before the resulting JSON is inspected.


## 2026-11-14 — completed local late_q A/B/B/A: NO_GO

User-uploaded flat_tile_resource_abba.json produced on RTX 5070 Ti.
Source HEAD 8e491e2af6f46f2d849270c593f011274198470c.
SHA-256 of the uploaded JSON bytes:
a4b7d5d86a605660901d37304c07f6a058b1efe3fee1efc0f726256817812b98
(4,224 bytes). Raw JSON is not committed to GitHub; this is
an auditable summary of the user-provided artifact.

Acceptance gates from the actual JSON:
- status=PASS_NUMERIC_TILE_ONLY, completed=true.
- candidate=late_q, source_head exact; 15,275 tokens,
  four independent BF16 heads, 32-tile windows, 512 near leaves.
- Both independent Q/K/V banks: PM/PL/PACC **bit-exact** to
  default materialized control. Bank A/B output differs.
- First-call complete Graph-vs-eager gate PASS and additional
  bank-B eager parity PASS.
- Four separately captured Tile-only CUDA Graph blocks
  in control, late_q, late_q, control order, seven warmed
  timings per block and five extra warmups.

Measured (compiled Triton kernel metadata; no Nsight in this run):

| Field | Control | late_q |
|---|---:|---:|
| Registers/thread | 255 | 255 |
| Shared bytes/CTA | 49,664 | 41,472 |
| Compiler spills | 0 | 0 |
| Optimistic CTA residency per SM | 2 | 2 |
| Per-variant block median-of-medians, ms | 0.391184002 | 0.395295992 |

Independent block medians, ms:
- Block 1 control: 0.391328007
- Block 2 late_q: 0.395231992
- Block 3 late_q: 0.395359993
- Block 4 control: 0.391039997

Shared decreased **8,192 bytes/CTA (16.50%)**, but
register count remains **255/thread**. Both register and
shared-memory limits independently remain at two CTAs/SM.
No occupancy increase was measured or admitted.

control_over_candidate_ratio=0.9895976928: the
candidate took ~1.051% longer in the Tile-only test,
and both independent candidate captures were slower
than either control capture. Neither the at-least-5%
speed criterion nor its both-captures consistency gate
passed. **Final late_q = NO_GO_TILE_ONLY**.
This is a negative performance result, NOT a failed
numerical test and NOT a change in output quality.

Keep existing LATE_NEAR_Q=0 default and do not
re-run late_q or promote it into production. The
standalone stage1 opt-in is already implemented
but **UNRUN**; evaluate it independently via the
same exact-HEAD safe runner with -Candidate stage1.
Do not blend late_q+stage1, alter exact near leaves,
or infer full-model speed from this single kernel.


## 2026-11-14 — completed local stage1 A/B/B/A: NO_GO

User-uploaded flat_tile_resource_abba(1).json from actual
NVIDIA GeForce RTX 5070 Ti. Source EXACT HEAD:
b9a71eb7b9f749278156baec7c9cb546f5011196.
SHA-256 of the 4,224-byte JSON:
8ac39ffe9414acae656b65f3dec51377b7a9b5aeed8ad6275abe677ef23e2a8c.
This repo stores only this summary, not raw user data.

Strict acceptance, from the uploaded JSON:
- schema=htrc-tile32-research-resource-abba-v1.
- status=PASS_NUMERIC_TILE_ONLY, completed=true;
  candidate=stage1, source_head matches, 15,275
  tokens, 4 independent heads, 32-tile balanced/512,
  one 2,048-query-row Tile window.
- PM, PL, PACC are **bit-exact** for BOTH independent
  BF16 QKV banks, compared with unchanged control.
  First full-call eager/Graph parity and bank-B eager
  parity both PASS. Distinct banks confirmed.
- Four separate Tile-only CUDA Graph captures:
  control, stage1, stage1, control. Each collected
  seven warmed CUDA-event timings; NOT Nsight timing.
- No real H3 checkpoint, complete attention timing,
  video, model-resident VRAM or production promotion.

| Measurement | Control (2 stages) | stage1 (1 stage) |
|---|---:|---:|
| Compiled registers per thread | 255 | 219 |
| Shared memory per CTA | 49,664 B | 40,960 B |
| Compiler-reported spills | 0 | 0 |
| Optimistic CTA residency per SM | 2 | 2 |
| Median-of-two-block medians | 0.388048008 ms | 0.467199996 ms |

Independent A/B/B/A block medians in ms:
- Block 1 control: 0.388864011
- Block 2 stage1: 0.467231989
- Block 3 stage1: 0.467168003
- Block 4 control: 0.387232006

Stage1 decreased registers by 36/thread (14.12%)
and allocated shared memory by 8,704 B/CTA (17.53%)
relative to this run's compiled control. Both register
and shared-memory limits remained **2 resident CTAs/SM**.
The lowered resources did not gain residency. Compiler
spills remained zero in both specializations.

Timing: control/candidate ratio = 0.830582216; stage1
is 20.40% SLOWER than control, and both candidate
capture medians are slower than both control captures.
The >=5%-faster and both-captures consistency gate
fails. **Final stage1: NO_GO_TILE_ONLY**.
This is a negative *speed* result with successful
numerical parity, not a numerical failure.

Caution: the lower number of Triton stages may diminish
load/compute overlap, but this is a hypothesis, not
validated by stall reasons, tensor-core counters or
unprofiled full-operator traces. Do not describe stage1
as proof that pipeline stages caused all lost performance.

### Resource-pressure experiment closeout

Both isolated one-knob variants have now been rejected:
- late_q: registers 255, shared memory 41,472 B;
  Tile +1.05% time, 2 CTA/SM. NO_GO.
- stage1: registers 219, shared memory 40,960 B;
  Tile +20.40% time, 2 CTA/SM. NO_GO.

The two experiments used separate control GPU samples;
compare ratios *within* each paired A/B/B/A, not absolute
control timings across runs. Do not combine late_q+stage1
without a fresh structural reason: neither isolated change
proved useful.

The existing materialized Tile32/STATS_BT16,
LATE_NEAR_Q=0, two Triton stages, RB=1 remains the
unchanged reference. No default kernel, dispatch or
ComfyUI code change. No additional GPU test is necessary
to close these two simple compiler-knob hypotheses.

A next engineering question is whether **reorganizing
Tile's far/near computation** can shorten the accumulator
and operand live ranges *without* discarding the working
pipeline and without duplicating near work. Previous
split/fused-VB experiments were already negative; do not
blindly repeat them. Before implementing a new kernel,
examine existing Tile compiled resources and current
specializations, then form a single falsifiable, numerically
exact opt-in experiment with independent-capture control.
The hardware gate for 3 CTAs/SM would still require both
<=170 registers/thread and <=34,133 B shared/CTA
before allocation granularity. That threshold is a
necessary constraint, NOT a speedup forecast.
