# 2026-11-16: bounded near K8+scale reindexing, strict numeric gate

Branch `experiment/flat-memory-to-speed-balanced-20261113`.
Parent HEAD: `d670daa5e892122a49d1a02d06466059b924f52f`.
No production `h3/` code, default CUDA Graph mode, ComfyUI or model
has been modified. **This is a hypothesis and unrun candidate, NOT
a demonstrated speedup.**

## Recovered evidence before a new kernel

The uploaded `flat_56head_graph_kernel_profile.json` showed two
independent 56-head BF16 QKV banks with all-output Graph/eager bit
equality and 66.667778 ms unprofiled complete Graph median.
Corrected device activity profile (excluding the enclosing
`torch.profiler.record_function`) has 1,211 CUDA activities,
66.075230 ms device-duration sum: Tile **41.456084 ms** (62.74%)
and leaf Stats **15.082391 ms** (22.83%). Tile + Stats account for
85.57% of activity duration. This is synthetic geometry, not
H3/video timing.

The existing pinned 15,275-token Balanced/512 split kernel
uses 954 full far leaves and **512 x 16 = 8,192 selected exact
near tokens** per query centre. Four split programs preserve
the same original near token ordering and online-softmax
accumulation. Reordering leaves across BN64 would change FP32
reduction association; the earlier far-packed experiment
already failed strict raw-bit PL/PACC and is closed.

Prior numerical experiments also reject RB2, late-Q, stage1,
near-PV-acc and naive fused-VB on their respective speed gates.
This new proposal is NOT an occupancy/BT/warp knob and does
not change leaf selection or numerical precision.

## Why NOT to pack all selected K8+V+scales

At **8 heads x 16 centres** or 4 heads x 32 centres,
selected-near materialization of K8 (1 B/element),
V BF16 (2 B/element) and K-scale FP32 (4 B/token)
for 8,192 near tokens x dim128 would require:

| Operand | Scratch MiB |
|---|---:|
| Selected near K8 | 128 |
| Selected near BF16 V | 256 |
| Selected near FP32 K-scale | 4 |
| Full near K/V/scale scratch | **388** |
| K8+scale only | **132** |

The previously observed first-call full Graph extra
reservation at this 56-head geometry was **590 MiB**.
590 + 388 = 978 MiB would already exceed the research
768 MiB Graph-extra cap *before allocator, shared
workspace or Graph pool overhead*. Do not create the
full K/V prepack candidate.

590 + 132 = **722 MiB**, leaving only **46 MiB nominal
headroom**; actual allocator reservation and whole-model
VRAM could still invalidate K-only even if numerical
parity and timing were positive. These are estimates
from DIFFERENT resource components, not a measured
per-process peak or permission to relax VRAM guards.

## One narrower materially distinct dataflow prototype

Prepack **only the ordered near K8 tokens and their
exact original FP32 per-token quantization scales** once
for each query centre, using the current *unsorted*
top-k IDX positions. Keep native BF16 V in its
unaltered memory and continue loading V with the
original IDX; no extra V materialization.

Existing source `_tile_kernel` is duplicated into
`tools/htrc_near_kpack_triton.py` only for this strict
research gate. The source clone must differ in exactly
four literal sites, automatically checked against the
canonical source:

1. research function rename;
2. near K8 address `(key_base+tok)` to
   `(key_base+j)`, selecting ordered contiguous scratch;
3. near KS address `key_base+tok` to `key_base+j`;
4. native BF16 V real-head base `head` to
   `(t//32)`, because the research variant represents
   each real head x centre as a pseudo-head.

Crucially: NO change to the far BL/BN64 traversal, four
original split intervals, G/LZ/VB/EXCL, unsorted IDX used
for V, original rowmax/topk, original near softmax
max/exp/sum and FP32 accumulators, original merge/tail,
dot precision, or BF16 source V. K8 bytes and FP32 KS
bits are copied unmodified, not requantized.

The exact source/caller geometry is **4 distinct BF16
heads, 32 centres x 64 rows, 15,275 tokens, exact near
512 leaves, 4 splits, BN64, STATS_BT16**, matching
the repo's existing far numeric gates. There are
128 pseudo-heads to match original 128 centres,
each with one tile and 8,192 packed K rows. The
original far states remain indexed by centre;
the V pointer is indexed by the original true head.
Packed K8+KS takes 132 MiB scratch plus a separately
allocated candidate PM/PL/PACC buffer.

The K8/scale pack is **not assumed cheap**, and even
zero memory-transaction reduction is possible; the
hope is to trade repeated indirect K8 loads/index
addressing inside the expensive Tile for coalesced
near K8 reads. It could easily be slower due to
per-centre K8 copies and extra launch. No speed
claim is authorized until a matched complete-call
candidate includes the pack and memory costs.

## Current gate — deliberately NUMERIC ONLY

`tools/benchmark_h3_near_kpack_numeric.py` first
runs the standard, unmodified real Graph vs eager
first-call check, then on TWO independent BF16 QKV
banks compares the entire unchanged baseline Tile
PM/PL/PACC vs packed-near Tile raw FP32 bits.
It separately compares the full original Merge output,
and independently checks bank B full Graph vs eager.
Source EXCL mask fingerprints must match previous
real data. Selected K/scale raw data are sampled and
compared directly, with whole-window output parity
covering all 128 centres.

This is one **fail-closed** numerical research probe,
not a new isolated speed sweep. It collects
no candidate CUDA-event timings, reports compiler
register/shared/spill metadata if generated, and
returns `PASS_NEAR_KPACK_NUMERIC_ONLY` or a terminal
`NO_GO_NEAR_KPACK_BIT_EXACT` (otherwise
`FAIL_CLOSED` infrastructure error).
Do not time a candidate if numerical parity fails.

One output file:
`D:\ComfyOutput\H3\math_probe\flat_near_kpack_numeric.json`.
Use `tools/run_h3_near_kpack_numeric_safe.ps1` with
`-ExactHead <new pinned SHA> -IClosedOtherGpuWorkloads`.
This runner uses disposable exact-source checkout,
CPU source-contract suite, mandatory quiet GPU and
Comfy port preflight. It does not alter model or
installed ComfyUI and emits only one persistent JSON.

**Next decision only if numeric PASS and resource
feasibility:** integrate K8 prepacking into ONE
optional complete 56-head Graph path; require
independent-capture A/B/B/A with pack cost included,
>=5% unprofiled full-call improvement, stable
controls and measured additional reserved/allocated
VRAM at or below the existing hard cap. The 132 MiB
estimate is NOT admission. Then separately run
real H3/model-resident VRAM and output-quality gates.
If not feasible, STOP; do not upsize scratch or
waive bit-exact requirements just to make a candidate run.

As of authoring: new numeric RTX GPU experiment,
full 56-head candidate, full model/video and
production promotion **UNRUN / NOT IMPLEMENTED**.
