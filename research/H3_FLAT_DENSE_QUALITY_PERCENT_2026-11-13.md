# HTRC Flat: quality comparison is a PERCENT vs exact dense attention

2026-11-13. Supersedes the *numerical abort policy* in
H3_FLAT_LENGTH_CLASSIFICATION_AND_GATES_2026-11-13.md.
Research only; archived eager Flat and the original materialized Graph
keep their default strict bit-equality guard in production.

## Owner correction accepted

The quality reference is **full dense attention**, NOT materialized
Flat, eager Flat, or numerical agreement between candidate kernels.
Different Flat modes may produce legitimately different approximations
to exact attention. A mismatch is a DATA POINT, not a reason to abandon
other modes or lengths. Do not substitute a Boolean torch.equal result
for a numeric quality metric.

Key fields:

- relative_l2_percent = 100 * ||candidate - exact_dense||_2 /
  max(||exact_dense||_2, 1e-30) over the evaluated query sample.
- per_head_rel_l2_percent and p50/p90/max per-head relative error.
- per_query_rel_l2_p50/p90/max_percent for tail-risk diagnostics.
- mean_abs/max_abs versus exact dense; report percentage-point
  improvement/deterioration versus materialized Graph for each bank.
- bit_exact_to_materialized and discrepancy percent normalized to the
  SAME dense reference: optional diagnostics, **not a gate**.
- GPU event and synchronized full-attention timings; graph memory
  reservation and allocated peaks, independently of quality.

Quality examples:
- materialized Graph error=11.5%, split_stream error=10.1%:
  candidate improves by 1.4 percentage points even if the two modes
  are not bitwise equal.
- fused_vb error=21%: record 21%, continue with next experiment.
- Nonfinite output: preserve explicit NUMERIC_NONFINITE_RECORDED,
  no fabricated percentage, and continue to the next mode.
- A CUDA OOM or kernel compilation failure is NOT an approximation
  quality score; record RUNTIME_FAILED and try further arms safely.

## Exact reference and long-context memory

The quality oracle uses FP32 scaled dot-product attention with
softmax over **every key/value token**, noncausal self-attention,
scale 1/sqrt(128), no near-leaf selection or Flat approximation.
TF32 matmul shortcuts are explicitly disabled for reference computation
and the prior PyTorch setting is restored afterward.
It evaluates evenly spaced query indices (default 64), including
the first and final query, per head and over TWO different QKV banks.
Q batches of 8 avoid allocating an N x N score matrix.

Crucial limitation: this is an exact dense result **for sampled query
rows**, NOT the exact dense output of ALL N query rows. The reported
error percentages apply to that sample and should not be represented
as whole-sequence perceptual/video quality. For a thorough short-N
oracle, increase --sample-queries up to N; this raises exact dense
compute from O(sample_queries*N) to O(N^2). Never default to full
223605 x 223605 attention on a 16 GiB card.

## Runtime numerical policy

h3/htrc_flat_directflow_live.py now has
graph_attention_bthd(..., compare_only=False) with the existing
strict production policy as the default. Strict mode keeps raising
when torch.equal(eager_flat, graph_flat) is false.
The standalone research runner explicitly supplies
compare_only=True. In that mode the first-call report has
status=RECORDED_FIRST_CALL, including relative_l2_to_eager_flat_percent
and max_abs_to_eager_flat for every head group. No mismatch raises,
but CUDA/OOM/memory bounds still apply. The geometry key includes
compare_only so strict verification cannot accidentally reuse a
non-strict session.

h3_flat_quality_ab.py performs quality vs exact dense for:
materialized, split_stream, fused_vb (all Graph), each on two
different BF16 QKV banks. All numerical outputs and quality
rankings are retained, without a PASS/FAIL threshold.
Known broken/uncompiled research kernels can appear as
RUNTIME_FAILED; they are not silently treated as 0% error.

The length matrix in benchmark_h3_flat_scale_matrix.py runs:
15,275 -> 18,723 -> 40,759 -> 82,000 tokens.
Both short lengths remain mandatory for every long hypothesis.
Full-HD 223,605 remains analytical memory-only. Comparison with Sol
is not included and should not be inferred from Flat-internal ranking.

## One JSON and invocation

Only the orchestrator writes
D:\ComfyOutput\H3\math_probe\flat_short_long_matrix.json.

Standalone research worktree, using the installed ComfyUI venv:

    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m unittest discover -s tools -p test_h3_flat_scale_protocol.py -v
    & $py -B tools\benchmark_h3_flat_scale_matrix.py --scope scaling --sample-queries 64 --execute

No automatic video generation and no changes to ComfyNodes. Default
without --execute only emits an experiment plan, no GPU work.

**Prior bit-exact benchmark tools are retained unmodified** as strict
arithmetic regression diagnostics. Do NOT treat their strict
torch.equal gates as a quality oracle or use them to drive the new
matrix. They are intentionally separate from the new non-strict
research workflow.

## Status

- Exact FP32 all-key dense-at-sampled-Q reference: implemented, GPU UNRUN.
- Research-only compare_only first-call metric: implemented, GPU UNRUN.
- One JSON across lengths/modes, nonfatal numerical deviations:
  implemented, GPU UNRUN.
- CPU contracts and source checks: prepared, execution UNRUN unless
  performed separately.
- Old production strict parity, eager and materialized defaults:
  unchanged.
- Sol crossover and real H3 quality: UNKNOWN.


## 2026-11-16 — 56-head acceleration pivot after resource-gate NO_GO

The original PASS0 vs source PASS1->PASS2 numeric experiment
(`flat_tile_far_near_pass12_numeric.json`; SHA-256
`98ad49e7aacc31d61006762607548a3ba31e982644bf20fa41b22880c98e7f0c`;
exact source `6549e75117b5d2585357a4702f673fe781d5fc74`)
passed strict two-bank PM/PL/PACC+Merge parity. But far-pass
212 regs/41,472 shared bytes and near-pass
255 regs/41,216 shared bytes do NOT increase the 2-CTA/SM
resource-based ceiling over original 255 regs/49,664
shared bytes, and add a second kernel and intermediate FP32
traffic. Thus **NO_GO_RESOURCE_ONLY**, without full 56-head
speed timing. See
`H3_TILE_FAR_NEAR_TWO_PASS_NUMERIC_2026-11-16.md`
for evidence, limits and closed mechanisms.

**Owner quality contract remains the one here, NOT a universal
bit-exactness rule.** All research changes that intentionally
change the approximation MUST compare % error vs FP32 exact
dense all-key attention for identical sampled queries on the
same TWO independent BF16 QKV banks. Track per-head and per-query
tail errors and full Graph GPU + wall and VRAM independently.
Do not silently promote a faster mode or equate synthetic
dense-relative errors with perceptual video validation.

The existing live configurations differ in exact-near budget:
`fast=256` and `balanced=512`, both sequential leaf ordering;
the 56-head DirectFlow code currently enforces these values
in its public modes. The highest-value first step is a **paired,
same-QKV fast/balanced quality/speed comparison** at the
15,275-token 56-distinct-head geometry using the already
implemented FP32 dense sampled-query reference and quality
metrics. This compares existing *legal* modes. The current
`tools/h3_flat_quality_ab.py` accepts one fixed mode per
invocation and optional streamed/fused VB variants, so a new
minimal paired-mode orchestrator should reuse its exact-dense
helper while **not** rerunning the closed slow VB arms.

A 384-leaf variant is NOT available by simply injecting a
different config: `h3/htrc_flat_directflow_live.py` checks
`expected_leaves={'fast':256,'balanced':512}[mode]` and
Graph session correctness depends on the fixed geometry.
Any third budget would require a separate opt-in research
dispatch and independent numeric/quality and VRAM gates;
do not patch production defaults or cheat this guard.

No near-budget 56-head paired GPU run, no 384-leaf kernel,
no real H3 video quality check and no production change
were made in this evidence-recording update.
