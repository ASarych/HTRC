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
