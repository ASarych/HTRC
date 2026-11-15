# HTRC: original Tile32 CUDA Graph control-only stability diagnostic

Date: 2026-11-15. Branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Starting exact HEAD: `d9490a01e642dc4a164499a30ce355a674283087`.

## Why this diagnostic replaces another candidate A/B/B/A

The preceding user-provided
`flat_full_far_vb_cg_tile_abba.json` at
`4f2749f3e93be8975cc790ac3e3f9ff30e04542c`
passed **two-bank full-Tile PM/PL/PACC and original Merge bit parity**,
with actual compilation metadata 255 regs for unchanged original,
254 regs for far VB `.cg`, 49,664 shared B and zero compiler spills.
It did **not** pass the speed gate:

| Capture | Kernel | GPU Event median |
|---|---|---:|
| 1 | Original | 507.576 us |
| 2 | far VB .cg | 374.404 us |
| 3 | far VB .cg | 374.382 us |
| 4 | Original | 383.578 us |

The two original captures differed **32.33%**, versus the 3%
pre-registered inter-capture spread bound. The candidate's
median-of-four arithmetic ratio suggested 15.98%, but that
number is **inadmissible** because original controls were
unstable; the conservative worst-pair gain was only 2.39%.
Result recorded at
`d9490a01e642dc4a164499a30ce355a674283087`:
**INCONCLUSIVE_TILE_ONLY**. Do not discard the slow control,
retrospectively change thresholds, combine with prior
`mask_g` 3.596% result, or run the model based on this report.

The cause of latency variability is UNKNOWN. GPU clock/power
drift, background interference, cache warmth, or effects of
separate CUDA Graph buffers and captures remain hypotheses,
not findings.

## New control-only mechanism

This experiment times **only the unchanged original full Tile**.
There is no research `.cg` candidate, masked-G variant,
compaction, alternate Tensor Core math or new Triton kernel.
The original Graph runtime is used solely to prepare the
same pinned two-bank BF16 QKV states and its native reference.
The control Tile is launched using
`batch.launch_prepared_tile_batch`; merged output uses
`batch.launch_prepared_merge_batch` with 11 exact tail tokens.

One identical **persistent sentinel CUDA Graph** is captured
ONCE, its scratch output frozen, and reused for the entire
sequence. Four separate **fresh control CUDA Graphs**, each
with separate output partial buffers, are captured one at a
time. For each fresh graph, measure the persistent sentinel
BEFORE, the fresh control, then the same persistent sentinel
AFTER, using the same 7 GPU Event samples x 16 replays.
Every fresh capture is warmed for 5 replays before timing;
the sentinel is warmed once and then kept alive throughout.

Comparing a new capture with a temporally adjacent re-used
sentinel can help distinguish:
- *Both original paths drift*: common time-varying
  environment or workload state is plausible;
- *Fresh controls drift, persistent sentinel remains stable*:
  capture/buffer sensitivity is plausible;
- *Fresh controls stable, sentinel drifts*: measurement
  asymmetry; do not call candidate benchmarks healthy;
- *Both stable*: the controlled baseline is reproducible
  under THIS diagnostic schedule, not necessarily under
  another A/B/B/A containing a candidate.

None of those classes uniquely identifies clock throttling,
thermal behavior, cache residency, system interference,
or a compiler bug. Sentinel replays also affect cache and
clock state and therefore the absolute measured Tile time
is NOT directly comparable to the prior A/B/B/A. The test
is about **relative stability**, not speed.

Optional `nvidia-smi` read-only snapshots are taken once
before and after the entire capture suite and include
reported P-state, SM/memory clocks, power, temperature and
utilization. These samples are too sparse to establish
what happened during individual 6 ms timing windows.
Missing `nvidia-smi` is recorded as unavailable, not as
zero clocks or a failed numerical experiment.
No clocks, power caps, drivers, or OS settings are changed.

## Exact original numerical gate before measurements

Pinned hardware and geometry: NVIDIA RTX 5070 Ti
SM120, 15,275 BF16 QKV tokens, four heads, one
2,048-query-row window, 32 x 64-row tiles, four far
BN64 splits, Balanced/512 exact near leaves,
STATS_BT16, original materialized VB, original
11-token exact tail, no model/video inference.

- Both independent BF16 QKV banks must have the unchanged
  original full eager/Graph bitwise parity and different
  `EXCL/IDX` selection masks:
  - A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`
  - B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`
- Each bank's independent original `_tile_kernel(PASS=0)`
  and original Merge must match its native Graph window
  bit-for-bit; bank B must also independently match
  eager full-attention output.
- The compiled unchanged Tile must still report 255
  registers/thread, 49,664 shared bytes/CTA, zero spills.
- A persistent sentinel and every fresh control must
  reproduce original PM/PL/PACC partials and original
  Merge result with zero raw FP32-bit differences
  before/after measured timing. An invalid original
  is `FAIL_CLOSED`, never a performance diagnosis.
- Four fresh independent Graph captures use exactly
  one unchanged control kernel; all phases report
  raw per-event samples, medians, event-level spread
  and the count of samples deviating >3% from median.

## Decision and limitations

`PASS_CONTROL_STABILITY_ONLY`: fresh original median
spread <=3%, and sentinel spread across all 8 bracketing
measurements <=3%, and each sentinel before/after pair
drifts <=3%. This status is **not a candidate speed PASS**.

`UNSTABLE_CONTROL_ONLY`: one or more of the above
stability requirements fail, with explicit classification:
`SHARED_CONTROL_AND_SENTINEL_VARIATION`,
`FRESH_CAPTURE_VARIATION_SENTINEL_STABLE`, or
`SENTINEL_VARIATION_FRESH_CAPTURE_STABLE`.
It is a completed valid negative diagnostic, not a
CPU/CUDA installation failure.

`FAIL_CLOSED`: source HEAD, QKV, numerics, compiler
resources, event timing, or runtime failure; no
admissible stability conclusion.

**Single output JSON** on the user's RTX 5070 Ti:
`D:\ComfyOutput\H3\math_probe\flat_tile_control_stability.json`.
The report preserves the exact commit, independent bank
fingerprints, per-capture bitwise gates, 12 groups of
raw GPU Event samples (4 blocks x 3 phases), optional
nvidia-smi data, spreads, classifications, and a
`candidate_timed=false` assertion.
No original experiment JSON is overwritten.

Research-only files:
- `tools/benchmark_h3_tile_control_stability.py`
- `tools/test_h3_tile_control_stability_cpu.py`
- `tools/run_h3_tile_control_stability_safe.ps1`
- This document.

**no changes to h3/**, original attention math, original
GPU kernels, production dispatcher, ComfyUI, or any
model checkpoint. This is a control-only diagnostic,
not an optimization or speed promotion.

## Authoring-time checks and status

The research source was assembled using exact original
Tile launcher and existing two-bank Graph/eager source
checks. Static audit confirms only original Tile launches
in the measurement path, paired sentinel timing, exact
HEAD before CUDA import and one-JSON Windows runner.
Additional CPU tests cover all four classification
outcomes, invalid samples and fail-closed SHA path.

The actual Python CPU unit suite is **UNRUN** here because
only a GitHub connector is available for this checkout;
the exact-head Windows runner executes both CPU/source
suites before running the GPU. This is not a CPU test PASS.

Target RTX 5070 Ti Graph diagnostic, its numeric checks
and all GPU timing: **UNRUN**. Full H3 model, video,
full-attention candidate speed and production promotion:
**UNRUN**. Preserve the old inconclusive evidence
without reinterpreting it.

## 2026-11-15 — measured original-only fresh/sentinel stability: SHARED variation

Evidence: user-uploaded `flat_tile_control_stability.json`,
**27,779 raw bytes**, SHA-256
`9ded0eb5d8fc8b2327c6506681532c292922172b3447885fbd81822fa5678080`,
tested on **exact source HEAD**
`103f7ff7a95a5139cfe46cb680817c2668dd8f86`.
The original user JSON is not committed to the repository;
this is its measured-result and provenance record.

Source report: schema `htrc-tile-control-stability-v1`,
`status=UNSTABLE_CONTROL_ONLY`, `completed=true`,
`classification=SHARED_CONTROL_AND_SENTINEL_VARIATION`,
`numeric_gate=PASS_TWO_BANK_ORIGINAL_TILE_MERGE_BIT_EXACT`,
`speed_measurement=CONTROL_ONLY_SENTINEL_BRACKET_EVENTS`,
`candidate_timed=false`.
The original CUDA Graph and independent original full
Tile+Merge on **both** independently generated QKV banks
were bit-exact, including separate bank-B Graph/eager
verification. Both original EXCL/IDX masks matched earlier
acceptance fingerprints:
- A: `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`
- B: `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`

Actual GPU scope: NVIDIA RTX 5070 Ti SM120,
15,275 BF16 key tokens, four heads, Balanced/512,
one 2,048-row Tile32 window, four far BN64 splits,
original 11-token tail merged **outside event timing**.
Four independent fresh original Tile CUDA Graph captures
were interleaved with measurements of the SAME
persistent original Tile sentinel CUDA Graph,
captured exactly once. Each of the 12 timed segments
(fresh or sentinel) collected seven CUDA event samples
of 16 graph replays per sample. Five warmup replays
per fresh capture. Pre-capture, post-warmup and
post-timing original full PM/PL/PACC and Merge raw-bit
checks PASS for all four captures and the original
sentinel. The original resource metadata remained
255 registers/thread, 49,664 shared B/CTA, zero spills,
identical on the two QKV banks.

### Measured control-only CUDA event medians

| Capture | Persistent sentinel before (us) | Fresh original (us) | Same sentinel after (us) |
|---|---:|---:|---:|
| 1 | 394.814 | 384.962 | 385.736 |
| 2 | 426.802 | 385.676 | 385.734 |
| 3 | 415.806 | 430.140 | 385.718 |
| 4 | 382.046 | 444.538 | 477.588 |

The independently recaptured original control medians
have **15.47581473%** max/min spread; the eight sentinel
bracketing medians have **25.00797818%** spread. The
sentinel pre/post paired drifts were 2.35343%,
10.64672%, 7.80052% and 25.00798%.
Both control and persistent sentinel fail the
pre-registered <=3% threshold; therefore report
`UNSTABLE_CONTROL_ONLY` and the narrower observed
classification `SHARED_CONTROL_AND_SENTINEL_VARIATION`.

### Raw sample-level evidence and interpretation

All 12 series contain individual large deviations
despite the original kernel and inputs remaining
unchanged. Each series has 7 raw GPU event samples.
Across all 84 timed samples, **47/84** differ by
more than 3% from their OWN seven-sample series median,
as counted in the uploaded diagnostic's
`samples_more_than_3pct_from_median` fields.
Per-series `max/min-1` spreads range from 10.0867%
to 39.2376%, including persistent-sentinel series.
The sentinel-before segment of capture 4 spans
**380.910 to 530.370 us**, while fresh capture 4
spans **385.050 to 512.958 us**. Capture 4's final
sentinel-after segment spans **415.146 to 563.024 us**.

This establishes that temporal variability can
affect the *same already captured CUDA Graph*; it is
NOT limited to separate Graph capture, source code,
partial buffers or candidate mathematics. The
results still do NOT distinguish between external
GPU interference, clock dynamics, thermal/power
states, event/replay sampling overhead and other
system effects. Avoid claiming a proven hardware
or Windows scheduler root cause.

Read-only `nvidia-smi` start/end snapshots:
- Start: 2026-11-15 18:46:17.986,
  P1, SM 2632 MHz, memory 13801 MHz,
  49.53 C, 46.15 W, GPU util 9%,
  memory util 3%.
- End: 2026-11-15 18:46:19.079,
  P1, SM 2887 MHz, memory 13801 MHz,
  63.61 C, 165.77 W, GPU util 74%,
  memory util 31%.

These are two coarse, non-simultaneous observations,
not a sampling trace synchronized with the 84 events.
The observed SM clock **increased**, not decreased,
between these snapshots. They do not establish
thermal throttling, clock stability in the event
windows, or interference by another process.

### Decision

**Control stability FAIL; numerical original correctness PASS.**
No candidate was timed and no speed comparison is
admitted. Do NOT reinterpret the preceding far-VB
`.cg` experiment's arithmetic 15.98% difference as
a validated speedup, or promote `.cg`/mask-G on
this evidence. The earlier mask-G ~3.596% Tile-only
gain remains independent and below its required 5%.

Before ANY further kernel optimization or new candidate
A/B/B/A, run a **distinct stability investigation**
that records GPU clock, temperature, power, utilization
and other GPU client activity at finer time granularity
if feasible, and that compares unprofiled repeated
sentinel replays under an explicitly quiet environment.
Do not drop slow samples post hoc to obtain a passing
score. A stable <=3% control-only gate is necessary
for interpreting a narrow >=5% speed threshold; it is
not alone sufficient for full H3 speed or image quality.

`h3_model_or_video=UNRUN`, full attention speed UNRUN,
production promotion PROHIBITED. This commit is
documentation-only: no new GPU tests, Python unit
tests, or changes to `h3/` or production code.