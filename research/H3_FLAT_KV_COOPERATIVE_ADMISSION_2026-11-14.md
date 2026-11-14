# HTRC Flat — cooperative-centre K/V reuse: architectural admission, NOT a kernel

Date: 2026-11-14. Based on experiment/flat-memory-to-speed-balanced-20261113
at 2c1e7d67cc3eedd5d33e2d26e764b2c437b37b6f. No changes to H3
runtime, ComfyNodes, Triton kernels, weights, exact leaves or default routing.

## Recovered evidence and the overlooked existing reuse

* The pinned materialized _stats_kernel runs BT=16 and BL=4. Each
  program loads a 16x128 K leaf and a 16x128 V leaf ONCE for a tile of
  **16 separate MU centres**, then computes (16x16) W, (16x128) G
  and (16x128) BF16 VB. The K/V operands are already shared across
  16 centres within a single stats CTA. This is source-level reuse,
  not an L2/DRAM measurement.
* The current Tile path has centre_group=1: one 64-query-row centre
  per Tile, with four far splits, BN=64 in the pinned PROD config,
  near K=256/512 leaves, INT8 G, BF16 VB and unchanged tail/merge.
  _tile_kernel's (head*tile,splits,1) launch already keeps each
  far split partial separate. FUSED_VB recomputed each VB leaf in
  each Tile CTA. It is bit-exact but 3.144x slower than materialized
  in the completed RTX 5070 Ti independent-capture ABBA
  (4.284656 vs 13.472448 ms). split_stream was also slower.
* The design target is 40,759 and 82,000 tokens, NOT a second
  optimization of the already rejected four-head 15,275-token case.
  A method that only improves the short case is not an admission.

## Ownership contract for a hypothetical multi-centre CTA

Three sequential stages, with barriers across separate kernels:

A. Stats: keep original centre geometry and exact G/LZ/SG.
   Possible optional output W[h,centre,leaf,16] in FP32 instead
   of VB[h,centre,leaf,128] BF16. W is the 16 normalized
   softmax weights already computed in original _stats_kernel.
   Keep EXACT original per-row max/exp/sum/divide and G rounding.
   The W hypothesis has NOT been implemented and has NOT passed
   a numerical GPU gate. Unlike storing BF16 W, storing FP32 W
   does not introduce a storage-rounding step; changing dot
   geometry or compiler schedules can still break bit-exactness.
B. Selection: preserve _rowmax_kernel, top-k IDX, EXCL and
   all selected 256/512 complete leaves. A barrier is mandatory:
   each centre's near set depends on scores over ALL key leaves.
   Neither the direct recompute nor staged-W approach can know
   which VB rows are excluded while initial stats are running.
C. Far consumer: conceptual CTA ownership
   (head, centre_block=floor(tile/B), far_split, query_row_group)
   for B in {1,2,4,8,16} consecutive centres and R in {1,2,4,8}
   disjoint equal-size query-row groups. The CTA reads the
   appropriate per-centre MU/G/LZ/EXCL and native V.
   Under W staging it also reads the FP32 softmax weights.
   Compute one leaf's per-centre VB and use the original
   far QK scores, exclusion, online m/l/acc recurrence and
   split order. Write distinct partial PM/PL/PACC addresses
   for every original (head, tile, split, query row).
   Then preserve exact near QK/PV, tail, and Merge.
   DO NOT share MU, EXCL, Q, scores or accumulator across
   different centres. A shared V leaf is legal; shared
   centre-specific softmax weights are not.

An ordinary Tile CTA cannot consume the result of another CTA
from its registers or shared memory. A producer/consumer split
requires global scratch and synchronization (like the slower
split_stream), or a verified CTA-cluster/TMA/DSM mechanism. No
cluster support or compiler occupancy has been proven on this
Windows RTX 5070 Ti stack. It is NOT implemented by merely
setting Triton's head/centre grid to a larger value.

## Source-level traffic model and byte accounting

The read-only, CPU-only model lives in
tools/estimate_h3_cooperative_kv.py, with seven tests in
tools/test_h3_cooperative_kv.py. It counts only source BF16
K/V reads and BF16 VB write+read, amortized per centre-leaf.
It deliberately omits the unchanged G/LZ/SG/IDX/EXCL/Q,
far softmax reductions, caches, L2 reuse, compiler-generated
traffic, CTA synchronization, Graph allocations and time.

One full leaf has 16*128*2 = 4096 bytes of BF16 K or V.
The reference 16-centre stats CTA reads both K and V once:
(4096+4096)/16 = 512 B per centre-leaf, then writes BF16 VB
256 B and Tile rereads VB 256 B. Thus a restricted modeled
reference is **1024 B per centre-leaf**.

An opt-in SCORE_ONLY stats pass reads only K, amortized
4096/16 = 256 B/pair. A naive direct far re-computation
must load K and V again, with ideal B-centre sharing.
With R independent query-row groups and U fraction of
leaves required by at least ONE centre in the group:

    direct_recompute = 256 + U * 8192 * R / B

This does NOT save the extra centre.MU dot/softmax work,
and it may load some K/V for excluded rows in the same pack.

Alternative (not implemented): store 16 FP32 W weights per
centre-leaf, exactly 64 B/pair, then read them for each R
query-row consumer and load only native V (not K).
This avoids repeating the centre.MU dot and leaf softmax:

    fp32_W_stage = 256 + 64 + 64*R + U * 4096*R/B

The write/read of W (64+64 at R=1) is explicitly included.
W uses one quarter the raw workspace of BF16 VB:
at 15,275 tokens, 4 heads, 32 tiles, 954 full leaves:
~7.45 MiB W vs ~29.81 MiB VB for one full Graph window.
Neither number is a measured model-resident VRAM peak.

Only after top-k can a consumer know each centre's far
set. With near fraction p=K/L, the BEST aligned-far-set
bound is U >= 1-p. For illustrative independent near
selections, U=1-p^B. This is an assumption, not a
measurement of actual correlations or memory traffic.
The worst-case bound is U=1 if any far leaves exist.

For R=1, B=8, the independent-selection illustration:

| Tokens | Full leaves L | Far fraction | Direct K/V + VB substitute | FP32 W staging | Reference |
|---:|---:|---:|---:|---:|---:|
| 15,275 | 954 | 46.33% | ~1273 B/pair | ~892 B/pair | 1024 B/pair |
| 40,759 | 2547 | 79.90% | ~1280 B/pair | ~896 B/pair | 1024 B/pair |
| 82,000 | 5125 | 90.01% | ~1280 B/pair | ~896 B/pair | 1024 B/pair |

At long context even B=8 DIRECT recomputation has a modeled
25% byte disadvantage on this subset, despite removing full
global VB. It also repeats tensor-core work. B=16 improves
this narrow source-byte model (~768 B/pair at U=1) but
requires a massive all-centre accumulator. That is not a
credible speedup claim. W staging with B=8 could have a
narrow modeled ~12.5% byte advantage, BEFORE the expensive
far consumer and any Graph occupancy/launch penalties.

## Occupancy/accumulator trade-off — the true blocker

One centre has 64 Q rows * 128 channels * 4 FP32 bytes:
**32 KiB of logical accumulator state**, before Q,
MU, scales, weights and scratch. A CTA owning all 64 rows:

| Centres/CTA B | Accumulator state |
|---:|---:|
| 1 | 32 KiB |
| 2 | 64 KiB |
| 4 | 128 KiB |
| 8 | 256 KiB |
| 16 | 512 KiB |

This is a logical data footprint, NOT reported registers,
shared-memory allocation or a hard NVIDIA device limit.
For a conservative 64-KiB *review filter* (not a hardware
cap), B>=4 cannot keep all accumulators in one CTA.
Splitting B=8 into four 16-row CTA groups lowers each
accumulator to 64 KiB, but V and W must then be loaded
by four different CTAs, eliminating the projected savings.
At R=4, B=8, W staging rises to ~2624 B/pair under
the long-context U=1 assumption. Existing Split/Tile
partial merging must also preserve exact reduction order.

A possible separate research route is a **verified
cluster-wide V multicast / producer-consumer** across
several query-row CTAs, with a per-centre row owner and
the original split accumulator order. This requires
device-level/driver/Triton support confirmation and
compiler-measured register, shared-memory, spill and
occupancy telemetry. Those preconditions are NOT met.
Never equate speculative multicast with proven HBM savings.

## Decision / acceptance

**NO_GO** for coding a naive B=2/4-centre fused recompute
kernel: it repeats an already existing 16-centre stats
K/V reuse and likely multiplies memory and compute.

**BLOCKED** for a B>=8 monolithic CTA or staged-W producer/
consumer: the source-byte model reveals a possible narrow
staging upside but all-row accumulator residency conflicts
with preserving K/V load reuse. Without an explicit
inter-CTA transport or a different exact accumulation
schedule, writing more Triton here is not justified.

Next minimal engineering input must be a concrete,
GPU-supported ownership/transport design: which CTA owns
every (head,centre,split,row) accumulator; how one V
leaf reaches multiple row CTAs without extra HBM loads;
how FP32 W can be staged and fed without changing exact
stats softmax/tf32 dot; and compiler register+SMEM+spill
evidence. Only THEN write an opt-in slice and require
two-bank full-output bit-exact, four-capture ABBA, and
separately isolated live VRAM. No real H3/video or production
promotion on this design document.

For now, retain materialized/Tile32/STATS_BT=16 and
the original eager fallback. No new GPU experiment
was run by this architecture slice.


## 2026-11-14 addendum — SM120 transport feasibility cross-check

Independent NVIDIA/CUDA/Triton source review: GeForce RTX 5070 Ti
is SM120; official NVIDIA CUTLASS GeForce SM120 examples explicitly
say TMA load **multicast is not supported** and use a 1x1x1
cluster shape. PTX cluster operations and TMA generally exist on
SM90+; that is NOT equivalent to hardware TMA multicast. Triton
issue #10973 (OPEN on 2026-11-14) reports SM120 cluster-op
lowering failures; the earlier #10963 is closed and superseded.
Neither parsing num_ctas=2 nor importing Triton establishes a
working producer/consumer cluster.

Therefore **NO_GO for CTA-cluster TMA multicast on RTX 5070 Ti**.
The prior possible cluster-multicast design above is retained
as a historical hypothesis, but is NOT an admissible local
engineering route. A non-multicast distributed-SMEM experiment
would need a separate working compiler/runtime and measured
transport cost; it is NOT implemented or admitted.

The remaining non-cluster possibility is independent CTAs with
L2 reuse of V, but cache hits are NOT guaranteed and cannot be
inferred from source loads. For B=8/R=4, the source-only traffic
model gives optimistic minimum variable-source cache hit
fractions of 78.125% for FP32 W staging and 81.25% for direct
K/V recomputation, under the deliberately favorable assumption
that the original materialized reference incurs all its
1024 B/pair as external-memory traffic. Actual reference L2
hits and kernel/Graph overhead make this only a best-case
necessary crossover, NOT an expected speedup. No model or
video result, profiler counters, or CUDA transport test exists.

See H3_SM120_TRANSPORT_ADMISSION_2026-11-14.md and the read-only
tools/probe_h3_sm120_transport.py inventory. No CUDA kernel or
production routing changes occurred.
