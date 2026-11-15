# HTRC Tile32: Near-field PV Tensor Core accumulator research

Date: 2026-11-15.
Branch: `experiment/flat-memory-to-speed-balanced-20261113`.
Measured predecessor HEAD: `5eba2a9ea074892b55a4552357a21d2fbe38f2d2`.

## Why this experiment is not another occupancy sweep

Four isolated Tile-only RTX 5070 Ti A/B/B/A investigations
have been closed as performance NO_GO despite successful
strict partial parity:

| Variant | Relative Tile time | Consequence |
|---|---:|---|
| `LATE_NEAR_Q=1` alone | +1.05% | Same 2 CTA/SM cap |
| `num_stages=1` alone | +20.4% (repeated +20.26%) | Same 2 CTA/SM cap |
| `RB=2` | +32.65% | 1024 vs 512 CTAs, SMEM still blocks 3 CTAs |
| `RB=2, LATE_NEAR_Q=1` | +36.4% | 3-CTA resource envelope, still slower |

Resource reduction / achieved occupancy are not a
goal by themselves. Splitting one tile into two CTAs
also duplicates some shared far/near loads. Stop
RB/occupancy variants without a new dataflow mechanism.

## Single falsifiable change

The existing `h3/htrc_flat_triton.py::_tile_kernel`
ALREADY implements `NEAR_PV_ACC`:

Control (near-field, DOT1, BF16 PV):

    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)

Research candidate (same source kernel, same inputs):

    acc = tl.dot(p.to(tl.bfloat16), vv, acc=acc * alpha[:, None])

The candidate moves the rescaled FP32 online-softmax
accumulator into the Tensor Core `tl.dot` acc argument.
It may avoid a separate matrix addition and reduce
temporary accumulator/operand lifetime or work issued
per CTA; it might instead increase register/shared
usage or instruction latency. It does NOT eliminate
all far/near loops, near K/V gathers or selection
loads, so do not assume a speedup.

**Numerical caveat:** Tensor Core accumulation can
alter FP32 rounding/association. Mathematical online
softmax formula is preserved, but bit-exact PM/PL/PACC
parity with the pinned control is NOT guaranteed.
A single mismatch is FAIL_CLOSED, **before timing**.

This is a research-only `near_pv_acc` variant of the
existing batch Tile launcher:
- `RB=1` and the exact same (128,4,1) CTA grid,
  no doubled state fetches;
- `LATE_NEAR_Q=0` and original `num_stages=2`;
- `FAR_PV_ACC=0` (unchanged, only near-field altered);
- same BF16 PV and INT8 Q/K, BN=64, 4 warps;
- 15,275 source tokens, 15,264 aligned K, 4 BF16
  heads, 32x64 query rows/window, balanced/512,
  four far/near leaf splits, materialized VB.
- leaf stats, top-k/selection, G/LZ/IDX/K/V caches,
  exact near leaves, merge, output and quality
  thresholds remain unchanged.
- Only an explicit research `research_resource_variant`
  enables this; omitted keyword passes
  `NEAR_PV_ACC=0` and compiles the old specialization.
  Normal Graph, eager, ComfyUI and production dispatch
  do not pass this parameter.

The Triton kernel source is **not edited** for this
experiment; its existing specialized branch is used.
Previous negative variants remain available for
reproducibility but are not re-tested here.

## Strict paired GPU decision protocol

The exact HEAD safe runner checks the repo's CPU
source-contract suite first. It refuses to start when
ComfyUI port 8188 or Blender is active, uses a disposable
detached checkout and leaves installed software intact.

GPU benchmark in `tools/benchmark_h3_tile_resource_abba.py`:
1. Two independent seeded BF16 Q/K/V banks with the
   pinned balanced/512, 4-head Tile32 shape.
2. Existing full first-call Graph vs eager parity PASS,
   plus separate bank B full eager/Graph parity PASS.
3. Control and candidate consume IDENTICAL prepared
   materialized states. Verify **bit-exact PM, PL, PACC
   in both banks**. Abort FAIL_CLOSED on any mismatch;
   **no speed measurements** in that case.
4. Record actual Triton compiled registers/thread,
   shared bytes/CTA and spills in the JSON. No inferred
   performance from compiler resources.
5. If strict numerical gates pass, four independent
   Tile-only CUDA Graph captures: CONTROL / NEAR_ACC /
   NEAR_ACC / CONTROL. Five post-capture warmup
   replays and seven timed GPU events per capture,
   checks for unchanged bit-exact partials before and
   after warmup and timing. Same CTA grid and input
   bank B throughout timed blocks.
6. Only `>=5%` improvement of median block medians
   AND both independent candidate medians lower than
   BOTH controls is a `PROMISING_TILE_ONLY` signal.
   Else `NO_GO_TILE_ONLY`, despite numeric PASS.
7. Do NOT infer full-attention speed, real H3 denoise,
   16GiB checkpoint fit, quality or video performance.
   Those are UNRUN, even if one Tile window wins.

The one report is written to the established fixed path:

`D:\ComfyOutput\H3\math_probe\flat_tile_resource_abba.json`

Only one JSON; on strict mismatch it is marked
`FAIL_CLOSED` with traceback and no A/B/B/A blocks.
The previous empirical outcomes remain recorded
in GitHub research documents.

## Implementation and status

New source only in research opt-in / test wrapper /
CPU contracts; no modifications to `_tile_kernel`
math source or production kernels.

Local Python:
- 16 CPU tests PASS, 1 source-check test SKIP
  (full HTRC checkout not mounted in this container).
- Python `py_compile` PASS.
- Explicit false-HEAD smoke test returned one
  `FAIL_CLOSED` JSON, exit code 2, no GPU invocation.
- Full-source assertions are checked independently
  against exact GitHub blobs, and the Windows
  detached exact-checkout runner runs them before GPU.

**GPU tests UNRUN.** There is no observed speed
or proof of numerical bit-exactness for the candidate
yet. Production remains unchanged.


## 2026-11-15 — Measured near_pv_acc decision: NO_GO

Raw user-uploaded flat_tile_resource_abba(6).json: 4,965 bytes,
SHA-256 d9de9d39cb0aefb970fc8d08c5cf4b236c8ef181542ecb08a2e49da6393a6890.
Exact source HEAD: d35b35fb24562ba0b6bf0b26fdbf3bb40136dba2.
The raw report is user-held; this commit records its grounded fields.

Report schema htrc-tile32-research-resource-abba-v1,
status PASS_NUMERIC_TILE_ONLY, completed=true,
candidate near_pv_acc; four distinct BF16 heads, 15,275
tokens, balanced/512, one 2,048-row Tile32 window.
Both independent QKV banks have bit-exact PM/PL/PACC
versus default materialized control; first-call full Graph
vs eager and separate bank-B eager parity both PASS.

Compiler result (both control and near_pv_acc identical):
255 registers/thread, 49,664 B shared/CTA, zero reported
compiler spills, 4 warps/CTA. CTA grid (128,4,1) = 512
for both; resource upper bound remains 2 CTA/SM.

Four independent CUDA Graph captured blocks, seven
warmed GPU-event samples each, A/B/B/A order:

| Capture | Arm | Block median (ms) |
|---|---|---:|
| 1 | control | 0.385343999 |
| 2 | near_pv_acc | 0.386976004 |
| 3 | near_pv_acc | 0.386783987 |
| 4 | control | 0.384831995 |

First control block contained an isolated 0.963616-ms
sample (other six samples around 0.384-0.387 ms).
The median is not driven by that isolated outlier.
Median-of-block-medians: control 0.385087997 ms;
candidate 0.386879995 ms. Control/candidate ratio
0.995368076. Candidate time **+0.4654%**, and neither
candidate capture is faster than either control capture.
The required >=5% improvement and both-captures criterion
FAILS; verdict **NO_GO_TILE_ONLY** despite strict
numerical PASS.

This only measures a single Tile kernel window,
NOT complete Graph attention, H3 inference, denoise or video.
The no-improvement result does NOT demonstrate the compiler
generated identical instructions or reveal which pipeline
component dominates. Default NEAR_PV_ACC=0 remains unchanged.
Stop micro-toggling existing compiler flags without a
different dataflow/work-reduction hypothesis.

### Next investigation: exclusion mask geometry, not an immediate kernel rewrite

In the far loop, EXCL masks near-selected leaves *after*
loading G/LZ, performing the far QK dot, and computing
candidate leaf scores. Masked-away leaves can consume
far-field dot work even though their exact tokens are
processed in the near branch. A potential far compaction
must be justified by measured selection masks first:
how many 64-leaf far blocks are completely excluded
and how many eligible far leaves remain per split.
If complete excluded blocks are rare, a trivial branch
skip is unlikely to help; a packed gather would need
a separate, genuinely worthwhile cost analysis.

Do not conflate theoretical far block reduction with
whole-attention speed gain. GPU work for the next
diagnostic (if implemented) is UNRUN at this commit.
No production source modified in this evidence slice.
