# HTRC original Tile far/near two-pass resource experiment

Date 2026-11-16.
Repository `ASarych/HTRC`;
branch `experiment/flat-memory-to-speed-balanced-20261113`.
Parent HEAD `369b2858f1e92b771b1749d269f5ecbba23b2646`.

## Prior evidence and why this is different

Actual RTX 5070 Ti user-held 56-head synthetic, Balanced/512,
15,275-token Graph profile: Tile **41.456084 ms** device activities,
Stats **15.082391 ms**; original complete 56-head Graph about
**66.67 ms**. Per-window near-K8+scale prepacking was measured
in strict full Graph A/B/B/A: 67.148880 vs 92.372784 ms,
**37.564% slower** despite raw-bit perfect outputs, so that
mechanism is CLOSED. Prior split_stream VB (+41.94% slower)
and fused_vb (3.144x slower after independent-capture A/B/B/A)
also have terminal negative evidence at the specified 4-head
Tile32 shape. STATS_BT32 was numerically exact but under 5% speed
gate and is CLOSED. Do not resuscitate any of these.

This candidate is **not** a new K/V packing, Vbar regeneration,
inter-centre K/V multicast, or altered leaf selection. It uses
the existing **unmodified production source** `_tile_kernel`
and its already compiled-in `PASS=1` (far-only) and `PASS=2`
(loads FP32 partials, then exact-near) semantics. The current
stock launch is always PASS=0 (far then near within one CTA).
Research dispatch instead launches far and near kernels
**in the same CUDA stream with exactly the same input and
output pointers**. Their intermediate scratch is the existing
FP32 PM/PL/PACC, no K/V scratch or new maths.

This changes one dependency: the second CUDA kernel reads
the FP32 partials written by the first. CUDA same-stream
kernel ordering provides a global pass1->pass2 barrier; no
GPU cross-stream sync, new buffer, sorting, recompression,
or approximation is allowed. Register/shared-memory occupancy
might improve because far and near operand live ranges no
longer overlap, **but that is UNMEASURED**. Extra launch,
FP32 intermediate reads+writes and repeated Q operand
preparation could make this substantially slower.

## Work and memory arithmetic (hypothesis model, not observed kernels)

At 56 distinct heads, 15 full/partial windows, each
with 14 x 1024 and 1 x 960 padded query rows, the
padded query rows per head are **15,296** versus
15,275 actually valid. Original Tile's four splits
cover **8,192 exact-near tokens/query**, plus 954
real far leaves, padded to 960 far dot columns/query.
The nominal tensor-core QK and PV accounting (multiply
and add counted as two operations) is:

| Part | Nominal FLOP-equivalent |
|---|---:|
| near QK + PV | 3.592740 TF |
| far QK + PV | 0.421024 TF |
| total Tile QK + PV | 4.013764 TF |

This is an **operation-count model**, not measured
floating-point tensor-core throughput: INT8 QK and BF16
PV use different hardware efficiencies; some padded
lanes are masked. The observed **41.456084-ms** Tile
kernel family implies about **96.8 nominal TFLOP/s**
only under that convention, NOT a calibrated roofline.
At a *hypothetical* uniform 100 TF/s, 4.013764 TF
would require 40.14 ms, but the mixed-precision
pipeline and Tensor Core tiling make that neither a
physical lower bound nor a performance promise.

Inter-centre K/V reuse is not assumed a compute FLOP
saving: original Tile already shares each near key
across its 64 Q rows; original Stats BT16 shares K/V
over 16 centre MU rows. Nonuniform near IDX across
centres, 255 regs/thread and 49,664 B shared memory
in the observed original Tile make cooperative-centre
ownership/occupancy difficult. Previous
`research/H3_FLAT_KV_COOPERATIVE_ADMISSION_2026-11-14.md`
already closed RTX 5070 Ti TMA cluster multicast as
unsupported; do not retry it.

Per full 8-head, 16-centre Graph window, the existing
partial tensor has **8 * 16 * 4 splits * 64 rows *
(2 + 128) FP32 elements = 17,039,360 B = 16.25 MiB**.
A two-pass implementation would ADD one far-stage
write and one near-stage read of those partials
relative to fused PASS=0; it may incur roughly twice
that logical traffic per window, plus extra kernel
dispatch and Q rework. Cache residency and actual
device transactions are NOT inferred from these counts.
For 105 windows per full 56-head call, the *upper*
logical extra byte accounting is around 3.33 GiB
if all use the full 16-centre shape; the last window
has 15 centres, so actual is slightly lower.
No new scratch allocation is proposed.

## Executable gate: 4-head Tile32, numeric ONLY

`tools/htrc_two_pass_tile_adapter.py` temporarily intercepts the
original `tr._tile_kernel` launch **only inside**
`batch.launch_prepared_tile_batch`, replacing the single
PASS=0 dispatch with PASS=1 then PASS=2. No replacement
Triton kernel is authored. No changes to the original
`h3/` files, and the monkeypatch is always restored
even if compilation or launch fails. It refuses changed
source PASS semantics or unexpected research/kernel
options and grids.

`tools/benchmark_h3_two_pass_numeric.py` uses the
previously validated 4 independent heads x 32 centres
x 64 query rows, 15,275 tokens, Balanced/512,
4 splits, BN64, no tail in the Tile and the unchanged
Merge. Two independently generated BF16 QKV banks,
pinned selection-mask fingerprints, first-call stock
Graph/eager full-output parity and separate bank-B
native eager comparison are mandatory.

Each bank runs (a) original PASS=0 untouched,
(b) original PASS1->PASS2 using the same FP32
partial workspace, (c) the original Merge for
both. **Exact raw FP32 bits of PM, PL, PACC
and merged full window** are compared against
the unchanged control. Compiler registers, shared
memory, spill count are recorded for original,
pass1 and pass2. A source or input drift is fatal.
No CUDA Event timers, benchmark loops, candidate
full 56-head Graph captures or model weights
are involved in THIS numeric gate.

The one persistent result is:
`D:\ComfyOutput\H3\math_probe\flat_tile_far_near_pass12_numeric.json`.
Use the exact pushed SHA in the isolated
`tools/run_h3_two_pass_numeric_safe.ps1` runner.
It checks ComfyUI port 8188, other known GPU clients,
detached exact-HEAD checkout, Python compile and
CPU/source regression suites before GPU.

Success means `PASS_TWO_PASS_NUMERIC_ONLY`.
Raw-bit mismatch means `NO_GO_TWO_PASS_NUMERIC`,
with speed experiments PROHIBITED.
Infrastructure/source errors mean `FAIL_CLOSED`.
If numeric PASS, **do not yet claim useful speed**:
first review compiled resources. Only if either
pass has a plausible memory/occupancy benefit
compared with the stock 255regs/49,664B
and no pathological spills should one opt-in
**full 56-head Graph** A/B/B/A be attempted.
A valid speed claim must include both launches,
extra FP32 intermediate memory and 105 full Graph
windows, two-bank exactness, 7 samples x 4
independent captures, <=3% between-capture drift,
>=5% median and worst-pair speed improvement,
and the original 768-MiB VRAM guard.

At authoring, Python/Windows full source gate,
GPU numeric, full 56-head performance,
model-resident H3/video all **UNRUN**.
No production promotion or default change.


## 2026-11-16 — completed two-bank PASS1/PASS2: resource gate NO_GO

Uploaded raw evidence: `flat_tile_far_near_pass12_numeric.json`,
**4,644 bytes**, SHA-256
`98ad49e7aacc31d61006762607548a3ba31e982644bf20fa41b22880c98e7f0c`,
tested exact source
`6549e75117b5d2585357a4702f673fe781d5fc74`.
The report is user-held; source JSON was not modified or substituted.
`schema=htrc-tile-far-near-pass12-numeric-v1`;
`status=PASS_TWO_PASS_NUMERIC_ONLY`, `completed=true`,
`parity_admission=PASS_TWO_BANK_PM_PL_PACC_MERGE_BIT_EXACT`.
Two independent BF16 QKV banks and different IDX/EXCL fingerprints:
A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`,
B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`.
Original first complete Graph/eager and independent bank-B
complete Graph/eager both pass. On EACH bank:

- PASS0 original vs sequential original PASS1/PASS2 FP32
  **PM**: 0 bit mismatches / 32,768;
  **PL**: 0 / 32,768;
  **PACC**: 0 / 4,194,304.
- Original Merge result: 0 / 1,048,576 FP32 elements.
- Original standalone Tile+Merge vs original captured Graph
  window also bit-exact. Same original partial pointers reused,
  and no extra K/V scratch was allocated by the adapter.
- No speed timing in the report: `speed_capture_blocks=[]`;
  `candidate_speed_measurement=UNRUN_NUMERIC_ONLY`;
  full candidate 56-head Graph and real model/video both UNRUN.

Actual compiled resources match on both banks:

| Pinned 4-head Tile32 kernel | Reg/thread | Shared bytes/CTA | Spills |
|---|---:|---:|---:|
| Original PASS0 | 255 | 49,664 | 0 |
| Far-only PASS1 | 212 | 41,472 | 0 |
| Near-resume PASS2 | 255 | 41,216 | 0 |

Resource-only occupancy *upper bound*, four warps (=128 threads)
per CTA, nominal 65,536 32-bit registers and 102,400 bytes
shared memory per SM:

- PASS0: min(floor(65536/(255*128))=2,
  floor(102400/49664)=2) = **2 CTAs/SM**.
- PASS1: min(floor(65536/(212*128))=2,
  floor(102400/41472)=2) = **2 CTAs/SM**.
- PASS2: min(floor(65536/(255*128))=2,
  floor(102400/41216)=2) = **2 CTAs/SM**.

This is only a **resource-based ceiling**, not an observed
occupancy measurement or throughput prediction. Neither pass
crosses the previously documented threshold for 3 CTAs/SM:
<=170 registers/thread AND <=34,133 shared B/CTA,
before compiler allocation granularity. Near PASS2 still
needs all 255 registers; a lower shared-memory allocation
does not help its CTA count. There are no new spills.

PASS1 writes FP32 PM/PL/PACC and PASS2 reads them, whereas
fused PASS0 maintains the intermediate in registers.
For an 8-head, 16-centre full window, each such partial
state is **16.25 MiB**, so an extra write plus read adds
**32.5 MiB of logical traffic**, plus one more CUDA launch
and repeated Q operands. At the equivalent 56-head,
14 x Tile16 and 1 x Tile15 window per head group,
the source-only extrapolation is **3.318634 GiB** of
extra logical partial traffic for 105 windows; this
is not measured DRAM traffic and L2 caching may change
external-memory costs.

**DECISION: NO_GO_RESOURCE_ONLY** for the existing two-pass
structure at the measured 4-head Tile32 compiler shape.
The prior registered resource gate required a plausible
register/shared residency gain before a full 56-head
speed experiment. It did not occur. Thus do **not**
build/timing-chase a 56-head PASS1/PASS2 Graph A/B/B/A
on the basis of this result. **No slowdown has been
measured for the 56-head shape**, and no universal
performance claim is made. Numeric success remains
valid independently of resource rejection.

A distinct strategy for material acceleration must
reduce useful work or establish a better dataflow
than splitting a live softmax accumulator into
more global memory operations. Avoid retrying K8
prepack, Vbar split-stream/recompute, and previously
closed RB2/near-PV-acc/BT32 knobs.

### Ownership: quality is measured against dense, not Flat parity

`research/H3_FLAT_DENSE_QUALITY_PERCENT_2026-11-13.md`
explicitly defines the quality oracle as full FP32
dense attention over **all keys**, sampled evenly
across query positions, on two independent QKV banks.
Bitwise equality with another Flat mode is a *diagnostic*,
not a general admission gate for approximate algorithms.
Strict parity remains required for mathematics-preserving
implementation changes and current production defaults.

A practical next research direction is near-leaf budget,
not near-leaf reordering/copy: current public modes are
`fast=256` and `balanced=512`, both seq Tile64.
Exact source `h3/htrc_flat_directflow_live.py` enforces
`expected_leaves = {'fast':256,'balanced':512}[mode]`;
a 384-leaf Graph may not simply be patched into a
production mode or reused in an incompatible Graph
session. Before authoring any 384 variant, first
compare the existing **fast vs balanced** modes using
THE SAME native BF16 QKV pairs, **% rel-L2 vs dense
FP32** and tail per-query/per-head error, paired
complete-Graph latency and measured VRAM. Do not
invent a quality threshold or confuse synthetic
error with real-video perceptual acceptance.
User approval and real-model/video proof would still
be required before any different default.
