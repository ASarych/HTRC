# HTRC: RB2 + late-near-Q conditional compiler resource admission

Date: 2026-11-15. Research branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Base HEAD: `10680acded8c0a3063a8c2fc10eb01e1cd781c5c`.

## Falsifiable question and existing negative evidence

Does **RB=2 + LATE_NEAR_Q=1**, with the original two-stage
`_tile_kernel`, reduce compiled shared memory enough for
**three 128-thread Tile CTAs per SM** on RTX 5070 Ti SM120,
while retaining bit-exact partials and zero compiler spills?
This is a necessary resource gate, NOT a throughput forecast.

Existing independent Tile-only CUDA Graph A/B/B/A results:

| Previous variant | Registers/thread | Shared bytes/CTA | CTA/SM resource limit | Tile-only result |
|---|---:|---:|---:|---|
| Control (RB1, late 0, stages 2) | 255 | 49,664 | 2 | Baseline |
| Late-Q only (RB1) | 255 | 41,472 | 2 | 1.05% slower; NO_GO |
| Stage1 only (RB1) | 219 | 40,960 | 2 | 20.4% slower, repeated; NO_GO |
| RB2 only | 140 | 37,376 | 2 | 32.65% slower; NO_GO |

These tests have already been closed. Do not repeat them
or claim additive changes to compiler resource usage.
A combined specialization can compile differently from
either separate candidate. The working default remains
RB1, LATE_NEAR_Q=0 and two stages.

From measured device constraints:
- Registers/SM: 65,536; shared memory/SM: 102,400 B;
  128 threads/CTA (4 warps).
- Necessary upper bound for three CTAs:
  **at most 170 registers/thread AND 34,133 shared B/CTA**.
- RB2's observed registers (140) already meet the first,
  but shared memory (37,376) is **3,243 B too large**.
- Memory/register allocation granularity may reduce actual
  residency. Three optimistic CTAs are not measured achieved
  occupancy and do not prove speedup.

## Minimal opt-in implementation

No edits to `h3/htrc_flat_triton.py`, state math, dense
reference, quality budget, head staging, full attention
Graph session or production routing.

The existing batch Tile launcher has an optional
`research_resource_variant` argument, omitted by all
production callers. Add **`rb2_late_q`** to that strict
research-only allowlist, only for the original pinned shape:
- 15,275 tokens; 15,264 full aligned keys; 4 distinct
  BF16 heads; 32 tiles of 64 query rows; balanced/512;
- Materialized Vbar, 4 leaf splits, BN64/BN_NEAR64,
  INT8 state and keys, group1, 4 warps, 2 stages,
  no fused/streamed-VB or feature combinations with stage1.

For this one candidate, both existing Triton constexpr
flags are enabled: `RB=2`, `LATE_NEAR_Q=1`.
The launch grid is `(128,4,2)` for one window, vs
`(128,4,1)` control. Each CTA writes disjoint
32-query-row slices of the original `PM/PL/PACC`.
Default kernel arguments are **unchanged**.

## Execute strict resource gate before Tile timing

Existing exact-head research runner performs the CPU
source-contract suite first. Any failure stops before
CUDA. GPU test uses the established two independent
BF16 Q/K/V banks, one synthetic 2,048-query-row
materialized window, existing full-attention first-call
eager-vs-Graph parity and fresh bank-B eager check.

The existing unchanged control Tile and combined Tile
must produce bit-for-bit identical `PM`, `PL`, and
`PACC` on BOTH banks. No numerical tolerance.
A mismatch is **FAIL_CLOSED**, not NO_GO_RESOURCE_GATE.

After compiling both kernels, obtain actual returned
Triton metadata for registers/thread, allocated
shared bytes/CTA, and compiler spills. Use installed
GPU SM limits and calculate necessary 3-CTA bounds.

- If registers exceed 170, shared bytes exceed 34,133,
  compiler spills are positive/unknown, or the calculated
  resource limit remains below 3 CTA/SM: result status
  `NO_GO_RESOURCE_GATE`, `completed=true`, two-bank
  numeric gates PASS, `abba_blocks=[]`,
  `tile_only_speed.admission=UNRUN_RESOURCE_GATE`.
  **No Tile-only independent Graph capture/timing runs.**
- Only when BOTH compiler resource bounds pass and
  compiler spills are known zero: admit four independent
  Tile-only CUDA Graph captures, in A/B/B/A order.
  Five warmup replays and seven timed CUDA events per
  block by default, strict before/after PM/PL/PACC parity.
  Status `PASS_NUMERIC_TILE_ONLY` then includes
  `resource_gate.status=ADMIT_TILE_TIMING_ONLY`,
  `abba_blocks` (four captures) and a speed decision.
  Both candidate blocks must be faster than both
  control blocks, and median improvement must be >=5%
  to mark `PROMISING_TILE_ONLY`. Resource admission
  itself is NOT a speed decision.
- CUDA, strict QKV/partial parity, compilation errors,
  missing mandatory register/shared metadata or unsupported GPU:
  `FAIL_CLOSED` with a traceback in the SAME JSON.
  Unknown or positive spill count with valid mandatory
  resources is instead `NO_GO_RESOURCE_GATE`.

No real H3 checkpoint, VAE or video; no full-attention
speed measurement, no paid inference. Even a promising
Tile-only result needs independent full-attention
A/B/B/A and model-resident VRAM gates before promotion.

## One output, one source of truth

Same single user report path as previous Tile experiments:

`D:\ComfyOutput\H3\math_probe\flat_tile_resource_abba.json`

This overwrites the last Tile resource file only on
the user's explicitly triggered run. Historical negative
experiments are preserved in the repository's research
documents, not duplicated output JSON files.

Run `tools/run_h3_tile_resource_abba_safe.ps1` in
an exact detached checkout with:
`-ExactHead <new_exact_SHA> -Candidate rb2_late_q -Rounds 7`.

CPU tests check the positive and negative 3-CTA
constraints, compiler spills (including missing metrics),
the mandatory pre-timing return, source allowlist and
one-JSON terminal status. No loose acceptance on errors.

## Implementation status

CPU tests executed locally against exact constructed
benchmark/test/runner Git blob sources:
**14 PASS, 1 SKIP** (only full HTRC source-file gate
requires a repository checkout; it is mandatory in
Windows runner and verified independently against the
updated GitHub launcher source). Python compile PASS.
Windows full source-suite and new GPU resource trial
**UNRUN**. Production remains unchanged.

## 2026-11-15 — combined RB2+Late-Q meets resource bound, but NO_GO Tile speed

Evidence: user-supplied flat_tile_resource_abba(4).json, **5,162 bytes**,
SHA-256 **894b37fe8c5904db2977da3fd05a3e06dabc1d67023a009deffdf9894d6ef4f7**.
Source exactly d9bce6149df03c3a6c0355029810d0bcf69d6e63,
schema htrc-tile32-research-resource-abba-v1,
candidate rb2_late_q, status PASS_NUMERIC_TILE_ONLY,
completed=true. JSON itself is not committed. The evidence
is from RTX 5070 Ti SM120, 15,275 BF16 QKV rows, four
distinct heads, balanced/512 and one 2,048-query-row
materialized Tile32 window.

All strict numerical gates PASSED:
- Two independent BF16 QKV banks: PM/PL/PACC bit-exact
  to RB1 control.
- First complete Graph attention call exact to eager;
  separate bank B eager/Graph parity PASS.
- Compiled kernel had zero reported spills.
- Resource gate status ADMIT_TILE_TIMING_ONLY: actual
  registers and shared bytes met both necessary bounds
  for *optimistic* three CTAs/SM. This status is NOT
  approval for production or evidence of time gain.

| Compiler/resource field | Control RB1 | RB2 + Late-Q |
|---|---:|---:|
| Registers/thread | 255 | **168** |
| Allocated shared bytes/CTA | 49,664 | **33,280** |
| Compiler spills | 0 | 0 |
| Warps/CTA | 4 | 4 |
| Optimistic resource bound, CTA/SM | 2 | **3** |
| CTA grid for one window | 128 x 4 x 1 | 128 x 4 x 2 |
| Total CTAs for window | 512 | 1,024 |

Three candidate CTAs at 128 threads/CTA request
3 x 168 x 128 = 64,512 registers and
3 x 33,280 = 99,840 shared B per SM; both are below
the observed device limits 65,536 regs and 102,400 B.
Actual achieved occupancy was NOT measured, so do NOT
state that three CTAs were observed resident. The
resource gate is a necessary estimate, with potential
allocation granularity and other occupancy limiters.

Four **independent** CUDA Graph captures of this
Tile-only kernel with 7 timings per block and five
untimed warmups, in control/candidate/candidate/control
(A/B/B/A) order:

| Block | Variant | Median GPU-event duration, ms |
|---|---|---:|
| 1 | Control RB1 | 0.390911996 |
| 2 | RB2 + Late-Q | 0.533343971 |
| 3 | RB2 + Late-Q | 0.533024013 |
| 4 | Control RB1 | 0.390879989 |

Median-of-two-block medians:
- Control **0.390895993 ms**
- Candidate **0.533183992 ms**
- Ratio control/candidate **0.733135275**
- Candidate takes **36.4005% longer** than paired
  control: both independent candidate captures slower
  than both independent control captures.
- Speed admission **NO_GO_TILE_ONLY**, even though
  compiler resource gate and all numeric gates PASS.

Prior RB2 alone was also slower (+32.65% within its
own matched four-block A/B/B/A). Those controls came
from separate runs, so do NOT attribute the marginal
difference between the RB2-alone and combined timings
to Late-Q without a *direct* paired RB2-vs-combined test,
which is not justified by two observed NO_GOs.

### Technical interpretation and final decision

The existing _tile_kernel splits the T=64 query rows
across program_id(2) for RB=2, but **each 32-row CTA
still loops over the same far-leaf range and exact-near
token range** for its leaf split. The shared G, LZ,
EXCL, VB, IDX and other leaf/selection inputs are
loaded by both CTAs while the Q-dependent dot and
accumulation math is divided by row. This doubles CTA
scheduling and duplicates some input/metadata loads,
not necessarily the total query-dependent FLOPs.
The extra work is a plausible cause of the regression,
not an experimentally isolated stall or traffic attribution.

This completes the explicit occupancy threshold hypothesis:
even attaining a compilation resource envelope permitting
three CTA/SM was NOT sufficient for better Tile runtime.
Stop occupancy-only tuning, do not test RB4/RB8,
do not combine further compiler flags without a
new independent mechanism. RB2+Late-Q remains
opt-in research only; the production baseline is
materialized RB1, LATE_NEAR_Q=0, stages=2,
STATS_BT16, unchanged.

If further optimization is pursued, investigate
**useful work per CTA and redundant far/near metadata
loads** in the existing two-stage Tile, while preserving
its correct online softmax, original quality budget,
and GPU-event whole-attention acceptance criteria.
Any proposed design must explain how it removes work
without multiplying CTAs or reducing the two-stage
pipeline; only then create a narrow numerical research
prototype. No new GPU experiment, model inference,
ComfyUI modification or production promotion was
performed for this documentation slice.
