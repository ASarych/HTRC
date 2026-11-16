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
