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
