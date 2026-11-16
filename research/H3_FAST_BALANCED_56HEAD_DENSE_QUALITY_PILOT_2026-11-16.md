# HTRC 56-head near budget: dense-relative fast vs balanced pilot

Date: 2026-11-16. Branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Parent after resource closeout:
`df5a62e64d4097c1b3f68d05177f57d54f17ef00`.

## Why this pilot exists

A 4-head Tile32, original PASS0 vs PASS1->PASS2 split
numerical test was **bit-exact** on two independent BF16
QKV banks, but compiled kernel resource metadata gives
an unchanged upper bound of **2 CTAs/SM** on RTX 5070 Ti:
original 255 regs/49,664B shared, far 212/41,472,
near 255/41,216. The split would also add an extra
launch and global FP32 partial buffer traffic; no full
56-head speed A/B was run. That is NO_GO_RESOURCE_ONLY
on the current strict unchanged-math route, not a
measured full-Graph slowdown.

The current original Tile is about 62.74% of recorded
CUDA device activities and its near QK+PV contributes
approximately 89.5% of its nominal QK+PV arithmetic
at 15,275 tokens with 512 near leaves (BN64).

HTRC's canonical owner rule in
`H3_FLAT_DENSE_QUALITY_PERCENT_2026-11-13.md` is to
**measure percentage error vs exact dense FP32
attention**, not automatically reject approximate
research variants for failing `torch.equal`
against materialized Flat. Existing strict production
default remains unchanged.

The HTRC `fast` config is `256` exact near leaves;
`balanced` is `512`. Both are existing live
supported modes, and the live DirectFlow enforces
their exact leaf counts. **Do not inject arbitrary
384 into current production config or Graph cache.**

## One minimal comparison of existing LEGAL modes

`tools/benchmark_h3_fast_balanced_dense_quality_pilot.py`
is opt-in research only; it calls the existing
`h3_flat_quality_ab.run_case` twice, once for
`balanced` then for `fast`, **without any VB
candidate**. One tiny research-only source update
permits `variants=()` for a materialized-only
`run_case`; the original nonempty variants
(`split_stream` and `fused_vb`) remain supported
unchanged and are not run.

Fixed geometry: 15,275 tokens, 56 distinct BF16
heads, 128-dim, Tile16, 2 independently seeded
QKV banks for each mode, seed 42026, matching
tensor geometry and seed across arms. Existing
`torch.manual_seed(seed)` inside `run_case` resets
the CUDA generator before constructing both banks
on each arm. Full inter-arm QKV identity is
inferred from this deterministic construction,
**not independently hashed**.

Exact quality oracle is **FP32 all-key noncausal
attention** with TF32 reference disabled,
evaluated on 64 evenly spaced query rows
(including first/last) for every head and both
banks. No `[N,N]` score matrix, no full-sequence
dense comparison or real-model claim.

Record for both banks and both modes:
- global relative L2 percentage error to dense;
- maximum per-head relative L2 percentage error;
- p90/max per-query error percent;
- `fast - balanced` difference in percentage
  POINTS, not as a ratio of error percentages;
- all seven single-capture warmed full-Graph
  CUDA Event and synchronized host-wall timings;
- PyTorch allocator peak reserved/allocated,
  plus runtime first-call extra reserved MiB;
- any nonfinite, OOM or runtime errors as
  nonpassing `FAIL_CLOSED`, never fabricated
  quality or performance.

**Speed status is preliminary only.** One
independent capture per mode is insufficient
for accepted performance; do NOT promote
the observed median difference as a speedup.
If quality and memory appear acceptable,
a later A/B/B/A with independent captures,
five warmups and seven events per capture
and the original stability/VRAM gate is needed.
Then real H3 video/latent quality must be
evaluated. There is **no invented quality
acceptance threshold** in this pilot; report
the actual error distributions and let the
research/owner decision use those data.

## Output and safe execution

One user-held JSON:
`D:\ComfyOutput\H3\math_probe\flat_fast_balanced_56head_dense_quality.json`.
One Windows exact-HEAD detached checkout runner:
`tools/run_h3_fast_balanced_dense_quality_pilot_safe.ps1`.
No ComfyUI backend changes, no new Triton kernels,
no 384-leaf Graph mode, no new checkpoint/video run,
no automatic production switch. GPU and complete
Windows CPU suites are UNRUN at authoring.

## 2026-11-16 — source-level fix after user Windows FAIL_CLOSED

At exact source `b027754436354087174790c45275160891298949`,
the Windows runner passed its CPU/source suite but the CUDA
pilot returned `FAIL_CLOSED` / `ValueError: Invalid two-bank
dense quality input or geometry`. **No fast-vs-balanced
quality or speed result was admitted**. The submitted console
output does not expose individual arm statuses, and the
original JSON did not retain them: `main()` performed
`report.update(run_gpu())`, while `run_gpu()` called
`interpret()` before returning the local arm dictionary.

Review of the exact source identified a deterministic unsafe
path in a normal Python process: `h3_flat_quality_ab.run_case`
called the live `flow.graph_attention_bthd()` without a
`torch.inference_mode()` context; live DirectFlow explicitly
rejects `torch.is_grad_enabled()`, which is ON by default.
The original console error does not prove this was the ONLY
runtime failure; previously recorded inner errors are
unrecoverable from that generic JSON.

Corrective research-only changes:
- New `_graph_attention_inference` in
  `tools/h3_flat_quality_ab.py` surrounds every live
  research Graph call with `torch.inference_mode()`
  and restores the ambient autograd state on both success
  and exception. No production `h3/` source changes.
- `benchmark_h3_fast_balanced_dense_quality_pilot.py`
  captures each materialized arm's progress **inside the
  persistent report** before interpretation, so the
  one JSON preserves both nested `runtime_error` values
  even when the comparison fails. Failure messages name
  the mode, inner status, original CUDA error and specific
  wrong geometry fields, instead of a generic ValueError.
- CPU-only regressions exercise normal default-grad refusal
  and recovery, nested error preservation, and no false
  success. The PowerShell wrapper now prints any preserved
  per-mode inner error on `FAIL_CLOSED`.

The FP32 sampled-query dense reference, numerical formulas,
two-bank sampling, fast/balanced modes, timing protocol,
768 MiB capture guard and production defaults are
unchanged. The earlier failed run is **not** a numerical
NO_GO, speed result or H3 checkpoint/video result.
Its only useful conclusion is that the pilot pipeline
needed this source/diagnostic repair.

Targeted Linux CPU torch mock smoke of the inference-mode
contract and retained-arm error path: **PASS**.
Full exact-source Windows CPU suites: **UNRUN** at this
document revision. Actual RTX pilot: **UNRUN on the new
source**. Do not backfill nonexistent measurements.
