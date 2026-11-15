# HTRC: dense versus packed-indirect far field data-copy gate

Date: 2026-11-15. Repository `ASarych/HTRC`.
Branch `experiment/flat-memory-to-speed-balanced-20261113`.
Exact parent HEAD `de5d8ec469c40e1c10915dfa9016d732a3ab26d0`.

## Why measure this — and what this is NOT

The actual NVIDIA RTX 5070 Ti far-exclusion geometry from two
independent BF16 QKV banks found zero completely excluded original
BN64 far blocks. The far-dot loop currently issues 1,920 nonempty
blocks per four-head 32-tile window. Split-preserving compaction has
a theoretical minimum of 1,056 / 1,049 blocks for the two banks,
45.0–45.4% fewer blocks; this is NOT a measured time reduction.

A separate GPU ordered-index packer already passed strict index,
split membership, ordering, full eager/Graph parity and suffix guards.
Its previous isolated median cost was 2.209 microseconds per window,
using only 526,336 output bytes. This cost came from a DIFFERENT run.

This slice tests the unresolved indirect DATA LOAD cost, **not** the
compacted far QK/PV attention computation, numerical rounding, full
attention latency, or real H3/video performance. No production code
is changed, and no previous negative occupancy or late-Q variants are
resumed. No claim of attention speedup is admitted by these tests.

## Narrow read-only dataflow experiment

New standalone, research-only files:

- `tools/htrc_far_gather_triton.py`: two small independent Triton
  copy probes, both using the identical `(128,4)` CTA grid and the
  original 4 split boundaries. `far_dense_copy` reads original
  sequential leaf rows from the materialized `G` int8, `VB` BF16,
  `LZ` float32, `SG` float32, `EXCL` int8 states. Every real leaf
  is copied, including leaf rows selected for exact near attention.
  `far_packed_copy` instead reads the already validated increasing
  `PACKED` leaf indices under `slot < COUNTS[center,split]` and
  indirectly gathers the same five fields only for eligible far
  leaves, preserving original ordering inside each split.
- `tools/analyze_h3_far_gather.py`: CPU-only reference layout,
  source leaf counts and ordered scalar content check.
- `tools/benchmark_h3_far_gather_cost.py`: identical prepared
  Graph QKV banks, original Graph/eager exact parity, actual GPU
  ordered index pack from the previous prototype, strict raw-byte
  equivalence of all five fields for two banks, then four independent
  captured copy probes in `dense / packed / packed / dense` order.
- `tools/test_h3_far_gather_cpu.py`: shape, ordering, masked-tail,
  invalid-count, fail-closed and exact-HEAD source contracts.
- `tools/run_h3_far_gather_safe.ps1`: ComfyUI/Blender preflight,
  disposable exact-HEAD clone, mandatory CPU tests, one JSON,
  no edits to the installed environment.

Only new files under `tools/` and `research/` are added.
`h3/htrc_flat_triton.py`, the working materialized Tile32, Stats
BT16, online softmax, merge, exact/near selection, default CUDA Graph
and production remain completely untouched.

### Fairness and strict interpretation

The two copy probes use the same 512 CTAs and output scratch *capacity*,
but they DO NOT perform equal amounts of work:

| Logical quantity, 4 heads × 32 centers | Dense reference | Packed candidate |
|---|---:|---:|
| Real source leaf rows read/copied | 122,112 | 56,576 |
| Original far split boundaries | 4 | 4 |
| CTA grid | (128,4) | (128,4) |
| Output capacity, scratch | 51,511,296 B | 51,511,296 B |
| Valid output rows | 122,112 | 56,576 |

Packed-to-dense logical leaf-load ratio is 46.33%. The candidate reads
an additional `PACKED` int32 and `COUNTS` int32. Both copy to fresh
scratch output on the GPU; this is intended to force the loads to
actually happen, not to reproduce the Tile's fusion, QK tensor-core
operations, L2 locality, achieved DRAM read transactions or softmax.
The output is NOT passed into attention and cannot serve as proof
of bit-exact `PM/PL/PACC` or an attention performance improvement.

All output capacities are identical, about 49.13 MiB per copy arm.
**Only valid prefixes are written. Uninitialized output suffixes
must not be read by validators or any future consumer.** The packed
index suffix is also uninitialized. The GPU kernel masks packed
index reads by actual count; the GPU data validator constructs safe
expected addresses from a previously verified CPU oracle, not from
uninitialized packed memory. The verified count itself is checked.

## Exact GPU gates and timing

Hardware only RTX 5070 Ti SM120; 15,275 BF16 QKV tokens, four
separate heads, balanced/512, 32×64 query rows, four far splits,
LPS=256, BN64; materialized `G/VB/LZ/SG/EXCL` with the real
prepared full-Graph selection.

1. Generate two independent QKV banks, run unchanged full
   Graph attention, check the first-call Graph-vs-eager exact gate,
   then independent bank-B full eager-vs-Graph parity.
2. Check each bank's selected IDX vs EXCL via the existing CPU
   reference; run the already validated ordered index pack.
   Check its exact valid prefixes and counts, not its suffix.
3. Run both new copy probes on each bank. Validate the produced
   `G`, `VB`, `LZ`, `SG` and `EXCL` bytes bit-for-bit against the
   real prepared state at each valid original index. In packed
   mode verify EXCL=0. Compare raw bytes even for BF16/FP32
   signed zero/NaN encodings (no float tolerance).
4. Fail closed on ANY content, shape, count, storage, or parity
   mismatch before taking CUDA event timing.
5. Four completely independent CUDA Graph captures, original
   `dense / packed / packed / dense` order. Each has five warmup
   replays, seven GPU-event samples, sixteen consecutive replays
   per event, timed samples divided by sixteen. Revalidate before
   capture, after warmups and after timing. All scratch/output
   allocations and correctness oracle construction are excluded
   from measured event timing.
6. Record actual compiled registers, shared bytes and spills
   when available; do not infer achieved occupancy. Record two
   arm medians and the ratio, clearly flag **this_is_not_equal_work**.

There is NO experimental attention candidate, NO compacted far-dot
kernel, NO `PM/PL/PACC` parity claim, and NO model/video inference.
A measured gather advantage would be a reason to PROTOTYPE a new
attention variant, not permission to adopt it in production.
A slower packed gather would constrain the potential savings but
would not automatically disprove a future fused compacted design.

### One JSON and failure handling

Expected output from the Windows exact-head runner:

`D:\ComfyOutput\H3\math_probe\flat_far_gather_cost.json`

`status=PASS_GATHER_COST_ONLY` means BOTH packed and dense copy
probes preserved raw source bytes and were measured; it is not a
speedup status. Any error yields a **single FAIL_CLOSED JSON** with
traceback and exit code 2. If CPU/source tests fail, GPU does not
start. Previous `flat_far_exclusion_probe.json` and
`flat_far_pack_cost.json` are not overwritten.

## Implementation status

CPU/source tests **12 PASS, 1 SKIP** locally. The one full-repository
source-contract check requires the user's exact detached checkout and
must run before GPU. `py_compile` PASS. Wrong-HEAD path tested locally
and fails closed without importing GPU dependencies. GPU compile,
Graph replay, gather parity on real G/VB and measured cost are
**UNRUN**. No current speed claim or production promotion.

## 2026-11-15 — actual RTX 5070 Ti packed-vs-dense far gather result

### Source and acceptance

User-uploaded original: `flat_far_gather_cost.json`;
**9,900 bytes**, SHA-256
`f1c93d17057f67ef7589a29cf8471ae4fcebc36b5da9b8f54f546da92aae3cef`.
Tested at exact source HEAD
`b564ccc112ef55369c88418b878a4761615a9de0`.
JSON schema `htrc-far-gather-cost-v1`,
`status=PASS_GATHER_COST_ONLY`, `completed=true`,
`probe=GPU_DENSE_VS_INDIRECT_FAR_FIELD_COPY_ONLY`.
Raw JSON stays outside repo; these lines record the uploaded
evidence, not a new hardware test.

Conditions: NVIDIA RTX 5070 Ti SM120, 15,275 BF16
QKV tokens, four distinct heads, 32 tiles, 4 aligned
far splits, BN64 and 512 exact near leaves.
Input: real prepared Graph `EXCL/IDX` and
`G/VB/LZ/SG` materialized states.

Both independent BF16 QKV banks PASS:
- Full first-call Graph/eager parity and independent
  bank-B complete eager/Graph parity.
- Ordered packed index prefixes, split membership,
  counts and no uninitialized-suffix reads.
- Five-field `G/VB/LZ/SG/EXCL` copy content
  **raw-byte bit-exact** in the dense reference
  and in the packed candidate.
- Selection fingerprints independently match earlier
  far exclusion and far pack tests:
  bank A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`;
  bank B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`.

The compared probes use the same 512 CTAs,
same input banks and scratch output allocation.
**Work is not equal**:
- dense: 122,112 valid source leaf rows copied,
  i.e., all 954 real leaves for each of 128 centers;
- packed: 56,576 valid far leaf rows copied,
  i.e., 442 selected non-near leaves per center;
- packed/dense logical row count: 0.463312369,
  a **53.6688% decrease in rows read and written**.
- dense and packed differ in index addressing and
  number of valid writes, so the timing ratio
  is NOT a pure measure of coalescing or GPU DRAM
  transaction count.

### Independent captured GPU-event result

Two dense and two packed independent CUDA Graph captures
in A/B/B/A order, seven samples per capture and
16 graph replays per GPU-event sample:

| Capture | Arm | Median ms | Median microseconds |
|---|---|---:|---:|
| 1 | dense control | 0.127851993 | 127.852 |
| 2 | packed indirect | 0.0184279997 | 18.428 |
| 3 | packed indirect | 0.0184359998 | 18.436 |
| 4 | dense control | 0.127486005 | 127.486 |
| **Mean of arm medians** | **dense** | **0.127668999** | **127.669** |
| **Mean of arm medians** | **packed** | **0.0184319997** | **18.432** |

Reported `direct_dense_over_packed_ratio=6.926486603`
for the **copy-only, unequal-work** task; the packed
copy has about **85.56% lower standalone kernel time**
than the dense copy. This is NOT an attention speedup,
not a measurement of fused gather overhead, not a
measurement of Tensor Core QK/PV, and not an isolated
cost of indirect addressing at equal byte count.

Compiled source data-copy probe resources:

| Resource | Dense control | Packed copy |
|---|---:|---:|
| Registers/thread | 112 | 95 |
| Shared bytes/CTA | 4,096 | 4,096 |
| Compiler spills | 0 | 0 |

Both arms allocated scratch output capacity of
**51,511,296 bytes = 49.125 MiB** to force actual
reads and writes; these are temporary test output
buffers and are not a demonstrated production
requirement. PACKED+COUNTS retains **526,336 bytes**.
The scratch and packed suffixes are uninitialized:
future consumers must read only valid-prefix slots.

### Decision: data-gather gate PASS, attention remains UNPROVEN

The earlier independent index-pack gate took
**2.209 microseconds per one-window pack** and this
new data-copy-only packed arm takes **18.432 microseconds**.
Those observations came from separate experiments,
so summing them is NOT a paired benchmark and cannot
be treated as a confirmed complete packed pipeline.
Nonetheless neither demonstrated a prohibitive
standalone cost for ordered indirect data access.

**PASS_GATHER_COST_ONLY** justifies a tightly scoped
far-attention NUMERICAL prototype, not a production
optimization or a presumption of speed gain.
A real packed far-field implementation must calculate
`QK`, `PV`, `PM/PL/PACC` using the original
softmax scaling, exact-near complement, and the four
original split boundaries.

The decisive risk is NUMERICAL: packing leaf columns
across BN64 blocks changes online-softmax reduction
association and Tensor Core operands. Even if the
input `G/VB/LZ/SG` bytes and leaf order are identical,
the FP32 partials may not remain bit-exact.
The **first GPU acceptance must compare PM, PL and
PACC bit for bit on BOTH independent QKV banks before
ANY candidate speed timing**. On mismatch, mark
`FAIL_CLOSED` with evidence and STOP; do not silently
relax to tolerance-based parity or change the working
attention math/production caller. Any future captured
A/B/B/A time claim additionally needs consistent
end-to-end complete attention speed, a memory budget
and model-resident proof.

Do NOT interpret 6.93× faster copy as 6.93× faster
attention, 53.7% fewer logical rows as measured 53.7%
lower physical DRAM, or the 45% theoretical block
reduction as actual compute savings. No new GPU run,
runtime update, video/model inference or source kernel
modification occurred while documenting this report.
