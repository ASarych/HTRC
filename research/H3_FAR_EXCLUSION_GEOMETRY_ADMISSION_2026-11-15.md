# HTRC Tile32: far exclusion geometry before any compacted kernel

Date: 2026-11-15. Branch: `experiment/flat-memory-to-speed-balanced-20261113`.
Implementation base: `6dd70da8f9361d989668f8b1323f2c54257a484c`.

## Reason to measure before modifying arithmetic

The previous NVIDIA RTX 5070 Ti study rejected four resource/row variants
(`late_q`, `stage1`, `rb2`, `rb2_late_q`) and the existing near PV `dot(acc=...)`
variant. All passed strict numerical parity but produced no Tile-only speedup.
In particular, obtaining resources permitting three resident CTA per SM did
not make RB2 faster. Further occupancy sweeps have no current justification.

The present `_tile_kernel` computes far QK dot per 64-leaf block and only
then applies `EXCL` to suppress near-selected leaves from the far result.
`EXCL` is constructed by `excl.zero_(); excl.scatter_(1, ids, 1)` after
top-k, and `IDX` carries the exact leaves processed by near-field attention.
The far kernel still evaluates full Tensor Core dot blocks when some far
leaves are masked out, but a fast skip of an ENTIRE block is possible only
when every valid leaf in that block is excluded. A more powerful compaction
could require new indexing, reordering, extra preprocessing and less coalesced
loads; it must not be implemented based solely on a theoretical FLOP count.

## Read-only empirical probe

New standalone research-only files:

- `tools/analyze_h3_far_exclusion.py`: pure CPU shape/invariant and block
  accounting; no torch dependency. The aligned `lps` formula matches
  `h3.htrc_flat_triton._split`.
- `tools/profile_h3_far_exclusion.py`: borrows the already established
  materialized full-Graph first-call path on RTX 5070 Ti (SM120); two
  independently generated BF16 Q/K/V banks; 15,275 tokens, four heads,
  Balanced/512, Tile32. Checks full eager-vs-Graph numerical parity on the
  first bank and separate bank B eager parity, and checks distinct outputs.
  Extracts actual prepared Graph `states["EXCL"]` and `states["IDX"]`
  from the full 2,048-query-row window after replay. The two banks must
  yield different mask fingerprints so stale state is not accepted.
- `tools/test_h3_far_exclusion_cpu.py`: mask/IDX equality, shape, tail,
  split-alignment, exact leaf budget, all-near and sparse-selection
  synthetic cases, source contracts and fail-closed report contract.
- `tools/run_h3_far_exclusion_safe.ps1`: exact-head disposable checkout,
  ComfyUI/Blender guard and mandatory local CPU/source suite before CUDA.

For each bank, fail closed if EXCL is not a binary mask, if selected leaf
indices are duplicated/out of range, if EXCL differs from IDX, or if the
pinned tensors have unexpected shape. Only the 4x32x954 EXCL and
4x32x512 IDX tensors are transferred to CPU for mask analysis;
full attention outputs are ALSO copied to CPU for two-bank parity.
Raw QKV tensors are not transferred to CPU by the probe.

The diagnostic counts actual far-leaf blocks using four split CTAs and
`BN=64`. With 954 full leaves, `lps = 256`, a valid tile iterates
4+4+4+3 = **15 far blocks** and there are 128 tile centers. The report
records, per QKV bank:

- total far blocks and how many are **fully excluded** (potentially
  skippable without leaf compaction);
- a histogram of 0/1-15/16-31/32-47/48-63/64 eligible far leaves in
  each far dot block;
- selected near leaves (512) vs eligible far leaves (442) per center;
- an *optimistic lower bound* on blocks after split-preserving packing
  of only eligible leaves. Packing is NOT implemented; it would add
  index construction, gathers and possibly uncoalesced memory access;
- per-head counts, precise mask SHA-256 and two-bank consistency.

This is **not** an NCU DRAM measurement, a candidate kernel, or a proof
of speedup. Existing first-call Graph/eager code may internally time its
own unchanged operator; the new probe performs **no candidate Tile A/B/B/A**.
A theoretical 50% reduction of far blocks cannot be equated with a 50%
reduction in the full attention call or even in Tile runtime.

## Decision after receiving the one JSON

If the measured number of completely excluded far blocks is small,
**do not implement a branch that skips whole 64-leaf blocks**: it cannot
remove significant far work. A packed-far prototype would need a separate
measurement of permutation/gather cost and a new bit-exact GPU numerical
admission before any independent-capture A/B/B/A. If many whole far blocks
are excluded, a minimal opt-in early-exit block-skip design can be
considered, preserving exactly the same PM/PL/PACC and merged output.

Do not modify the current default materialized Tile32, Stats BT16,
quality budgets, exact leaf selection, CUDA Graph session, dispatch,
ComfyUI or production. No new H3 model/video run.

## Safe execution and status

Exact-head PowerShell runner invokes `tools/profile_h3_far_exclusion.py`
with `--expect-head`, and produces exactly **one user-facing JSON**:

`D:\ComfyOutput\H3\math_probe\flat_far_exclusion_probe.json`

Success: `status=PASS_FAR_EXCLUSION_GEOMETRY` with two audited banks;
errors: `FAIL_CLOSED` with traceback in the same JSON. A CPU preflight
failure refuses GPU before writing a new report. The probe is confined
to the branch and does not promote any production optimization.

Local tests: **8 PASS, 1 SKIP** (full-repository source-check requires the
user's exact checkout). `py_compile` PASS. Mismatched/unavailable HEAD
smoke returned one `FAIL_CLOSED` JSON with exit 2 and no CUDA call.
The repository's full source gate is mandatory on Windows; the related
source literals are verified against exact GitHub blob contents.

**Actual user GPU geometry result UNRUN.**


## 2026-11-15 — observed 2-bank far-exclusion geometry

Evidence file uploaded by user: flat_far_exclusion_probe.json;
5,416 bytes, SHA-256
7ac70141c73188977f909249f1adce8ab7c76ac3f9bdd261ab383efa5d50f3c3.
Exact source HEAD: 5586c61f8da3bbc65875f15f457afb893a306b4e.
Raw JSON remains user-owned; this repository records an
auditable result summary, not a substituted simulation.

### Acceptance and scope

The actual RTX 5070 Ti SM120 report has schema
htrc-far-exclusion-geometry-v1,
status=PASS_FAR_EXCLUSION_GEOMETRY, completed=true,
all_mask_id_invariants_passed=true, two independent
BF16 QKV banks with different selection-mask SHA-256.
Both first-call and bank-B complete Graph-vs-eager
bit-exact tests passed. The prepared state came from
native materialized full-Graph selection.

Each bank: 15,275 tokens; 4 heads x 32 tile centers
=128 centers; 954 full 16-token leaves/center;
512 near-selected leaves/center; 442 eligible far
leaves/center; 4 original far splits; BN64;
_split(954, 4, 64)=256. Actual far loop iterations
per center: 4 + 4 + 4 + 3 =15.
No new attention kernel and no candidate GPU timing:
ab_ba_speedup=UNRUN and h3_model_or_video=UNRUN.

### Observed block geometry

| Measured value | Bank A | Bank B |
|---|---:|---:|
| Current far 64-leaf dot blocks | 1,920 | 1,920 |
| Fully excluded blocks | **0** | **0** |
| Blocks with some eligible far leaves | 1,920 | 1,920 |
| Eligible far leaves across 128 centers | 56,576 | 56,576 |
| Total 64-column dot lanes including padding | 122,880 | 122,880 |
| Real full leaves excluded into near, fraction | 53.6688% | 53.6688% |
| Blocks with 16-31 far leaves | 1,346 | 1,359 |
| Blocks with 32-47 far leaves | 574 | 561 |
| Blocks with 0, 1-15, 48-63 or 64 far leaves | 0 | 0 |
| Idealized per-split compacted block count | 1,056 | 1,049 |
| Difference versus current blocks | 864 | 871 |
| Idealized block-count reduction | 45.000% | 45.365% |

Selection-mask fingerprints were distinct:
bank A bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32;
bank B 05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264.
Within each bank every IDX leaf matched one EXCL=1 leaf;
no out-of-range, duplicated or stale selected-leaf
indices were admitted.

Per-head idealized compacted blocks were:
bank A 264/261/268/263; bank B 264/261/262/262.
Their sums reconcile to 1,056 and 1,049.
The block lower bound is computed as the sum of
ceil(far_eligible_in_each_original_split / 64)
per center, preserving original split boundaries.
It does not encode pack/gather overhead or any
working GPU implementation.

### Decisions and important caveats

**Simple far block skip: NO_GO.** Neither bank had
one wholly excluded BN64 block. Adding a conditional
skip for zero-useful-leaf blocks in the current kernel
would skip exactly zero actual blocks in this measured
configuration. No new kernel is warranted for it.

**Cross-block far compaction: HYPOTHESIS ONLY.** There
are enough sparsely populated blocks to justify
considering packing indices (16-47 useful leaves
per original block, 29.4667 mean); but the 45% figure
is ONLY an arithmetic lower bound on dot-block count.
It is not a measured reduction in GPU instructions,
physical DRAM/L2 traffic, Tile time or full attention
time. Packed indirect loads, index construction,
synchronization, dynamic group sizes, less-coalesced
access and unchanged near-field work incur costs.

More importantly, moving leaves across BN64 group
boundaries can change online-softmax max/exp/sum
update grouping, BF16 dot inputs and FP32 reduction
association. Thus **bit-exact PM/PL/PACC versus the
current materialized Tile is NOT established and may
be fundamentally difficult to preserve**. This
strict gate cannot be waived merely because
selection membership is identical; numerical and
performance admission are different claims.

If developing a packed-far candidate later:
1. First benchmark index-pack *construction overhead*
   separately, with exact split membership and ordered
   indices for BOTH QKV banks. Do not present index
   packing alone as an attention speedup.
2. Before any Tile performance timing, require full
   strict numerical PM/PL/PACC parity under actual
   indirect gathers. If reordering fails bit-exact
   parity, STOP; do not quietly switch to a tolerance
   threshold or change the quality contract.
3. Only after numeric acceptance and evidence that
   packing overhead is plausibly amortized should
   independent-capture complete-attention A/B/B/A
   and model-resident VRAM be considered.

Do not implement a zero-block skip, RB4/RB8, or a
production compacted Tile based on this report.
The existing materialized RB1 / STATS_BT16 / BN64 /
2-stage kernel remains unchanged. No new GPU
experiment or production write occurred in this
documentation-only result slice.
