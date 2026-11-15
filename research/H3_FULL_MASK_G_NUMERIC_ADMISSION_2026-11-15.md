# HTRC: full Tile32 mask-G parity with original Merge (no speed test)

Date: 2026-11-15. Research branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Implementation base: `e6a2dceb51c63fa1d9cc07b78b713dbf45da053c`.

## Motivation and previous verified evidence

The preceding RTX 5070 Ti masked far-only experiment at
`57b6c48667f3d0cafc141e5816acc0df43e23c06`
returned `PASS_MASKED_FAR_NUMERIC_ONLY`. `mask_g`, `mask_vb`
and `mask_both` exactly reproduced original far-only `PASS=1`
FP32 PM/PL/PACC on two independent BF16 QKV banks.
However `mask_g` used 255 registers/thread, zero compiler spills;
`mask_vb` and `mask_both` reported 12 and 8 spills, respectively.
The unmasked far-only research `mirror` passed its reference parity.

Earlier cross-BN64 far packing changed FP32 PL/PACC and was
rejected as `NO_GO_FAR_BIT_EXACT`. Here there is NO compacted
index list, NO change in BN64 grouping, and NO change in the
number/order of far and near Tensor Core dot operations.

The unresolved strict question is whether mask-G retains
bitwise equality when the complete original far+near Tile
is fused in ONE kernel, and whether its downstream original
Merge (including the 11 exact tail tokens) also remains exact.
The result must not be inferred from the far-only result.

## Research-only implementation

New files (only under tools/ and research/):

- `tools/htrc_full_mask_g_triton.py`: one independent
  research-only full far+near Tile clone with `MASK_G=0/1`.
  Only the far **G** load changes its mask from the original
  valid-column mask to `cm & (EXCL == 0)`; excluded lanes
  receive zero. VB load remains unchanged. The kernel retains
  original BN64 far dots, four original split boundaries,
  INT8 QK operand quantization, BF16 PV, FP32 online-softmax,
  all 512 exact near leaves and near Q8/K8/V semantics.
  It writes complete (unnormalized) `PM/PL/PACC` partials.
  It never edits production `h3/htrc_flat_triton.py`.
- `tools/benchmark_h3_full_mask_g_numeric.py`: exact two-bank
  prepared-Graph source validation, original full Tile PASS=0
  control through existing `launch_prepared_tile_batch`,
  independent research mirror, then mask-G candidate.
  Both candidates use the *same existing*
  `launch_prepared_merge_batch` with `tails=session['tail']`
  and `merge_rows=32` (11 exact tail tokens at 15,275 keys).
  First compare original reconstructed Tile+Merge against
  existing captured Graph window output, then compare all
  FP32 partial bit patterns AND merged output bits.
- `tools/test_h3_full_mask_g_numeric_cpu.py`: fixed geometry,
  near/far ordering, unchanged masks and near loads, original
  source contract, wrong-HEAD fail-closed and runner checks.
- `tools/run_h3_full_mask_g_numeric_safe.ps1`: Windows
  ComfyUI/Blender guard, disposable pinned HEAD checkout,
  mandatory CPU tests before CUDA, exactly one output JSON,
  recognized numerical NO_GO rather than false installation error.
- This research admission document.

Reuses existing, already admitted source utilities from the
same exact Git checkout for prepared masks, parity diagnostics,
canonical Graph/eager checks, baseline Tile and original Merge.
No extra changes to these sources and no copied implementation
of the Merge or two-pass far/near calculations.

## Pinned source and numerical protocol

- NVIDIA RTX 5070 Ti, SM120, 16 GiB VRAM.
- Two separately generated BF16 QKV banks, seed 42026.
- Shape: 15,275 tokens, four heads, 128 dims; full
  attention on unchanged baseline Graph for source admission.
- Exact candidate scope: **one 2,048-query-row window**,
  four heads x 32 tiles x 64 rows, materialized VB and
  original STATS_BT16 selection. The 15,275-token full
  attention result is only used to establish canonical
  eager/Graph correctness and the source state.
- Balanced/512 near leaves, 954 complete 16-token leaves,
  exact 11-token tail, four original leaf splits, far BN64,
  near BN64, RB1, original two stages, four warps, INT8 G/Q/K,
  BF16 VB/V and exact merged tail.
- EXCL/IDX must match for every head/tile, and each QKV bank
  must retain its earlier actual selection-mask SHA-256:
  A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`;
  B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`.
  Any mismatch fails closed before candidate admission.
- Re-execute the **unchanged original full Tile** into separate
  FP32 partial buffers and Merge the 11-token tail. Check its
  merged output against the active Graph-window output itself.
  A baseline mismatch is an invalid oracle, not a candidate NO_GO.
- First run `MASK_G=0` research `mirror`. Require bit-for-bit
  equality of FP32 PM, PL, PACC AND original Merge output.
  If mirror fails, do NOT admit a masked candidate; retain
  the mismatch diagnostics on both QKV banks.
- Only after a passing mirror on a bank, test `MASK_G=1` using
  the same BN64, near and Merge paths. Check every FP32 raw
  bit (including signed zero), not `torch.equal` or tolerance.
  Record per-field mismatch counts, first mismatch location,
  max finite absolute difference and compiler registers,
  shared memory and spills for both original and candidates.
- Both QKV banks and the original first-call and separate
  bank-B full eager/Graph gates must pass. Only then can
  the test claim a full-window numerical `PASS`.

The source `h3/` kernel, launcher, Merge, production
attention dispatcher, model weights and ComfyUI remain
completely unchanged. No Torch CUDA event timing or
A/B/B/A is permitted in this numerical slice.

## Outcome states and limitations

The script produces exactly ONE file:

`D:\ComfyOutput\H3\math_probe\flat_full_mask_g_numeric.json`

- `PASS_FULL_MASK_G_NUMERIC_ONLY`: mirror, candidate and
  canonical Merge all bit-exact on BOTH QKV banks. Next
  independent GPU slice may measure full **window** Tile-only
  performance, subject to compile resource and spill review.
- `FAIL_CLOSED`, `parity_admission=NO_GO_FULL_RESEARCH_MIRROR`:
  research unmasked clone failed numerical equality. Stop
  before performance; diagnose clone scheduling/rounding.
- `FAIL_CLOSED`, `parity_admission=NO_GO_FULL_MASK_G_BIT_EXACT`:
  mirror exact, masked candidate differs in either partials
  or Merge. Do not promote or measure candidate speed.
- Any source SHA, geometry, Graph parity, CUDA compile,
  shape or runtime exception is an ordinary `FAIL_CLOSED`
  with traceback and exit 2, not a numerical pass/NO_GO.

All statuses preserve `speed_measurement=UNRUN_NUMERIC_ONLY`,
`no_cuda_event_timing=true` and `speed_capture_blocks=[]`.
No benchmark result or model/video claim is possible here.
Even a one-window full numerical PASS is NOT proof for all
windows, all 56 heads or the model-resident denoising process.

## Implementation test status (before any user GPU run)

- Local CPU/source contracts: 7 PASS, 1 SKIP; the skip
  requires the exact full HTRC checkout and is **mandatory**
  in the Windows disposable runner before GPU.
- Python syntax / `py_compile` for new modules: PASS.
- Wrong HEAD smoke: PASS, one `FAIL_CLOSED` JSON with exit 2;
  no CUDA/Triton import prior to source SHA gate.
- New RTX 5070 Ti full far+near and Merge numerical run:
  **UNRUN**. Actual compiler resources for this full masked
  variant are unknown until the target GPU runs.
- GPU performance, complete multi-window masked attention,
  model/video output and production promotion: **UNRUN**.


## 2026-11-15 — observed RTX 5070 Ti full-window mask-G numeric PASS

Evidence supplied by user: `flat_full_mask_g_numeric.json`,
7,540 raw bytes, SHA-256
`27e937473afe495d1446cf97ed4f88d2b1e748e0be4663cb383adf2fac78b04c`.
Executed at exact source HEAD
`352b4b5992d2d209a6b76a556aac6abe9d1eedd0`.
Raw JSON remains user-owned. This section records the measured
facts and strict next-gate decision.

Result: `schema=htrc-full-mask-g-numeric-v1`,
`status=PASS_FULL_MASK_G_NUMERIC_ONLY`, `completed=true`,
`parity_admission=PASS_FULL_WINDOW_PARTIALS_AND_MERGE_ONLY`,
`mask_g_full_numeric_bit_exact=true`,
`mirror_full_numeric_bit_exact=true` and
`baseline_window_graph_parity=true`.

Target and scope: NVIDIA RTX 5070 Ti SM120, two independent
BF16 QKV banks, 15,275 key tokens; pinned four heads,
32 tiles and one 2,048-query-row materialized window.
The original full attention was used to establish first-call
and bank-B eager/Graph parity. The candidate was validated
on **one window only** with Balanced/512 near selection,
954 full leaves, four BN64 far splits and 11 tail tokens
merged by the existing `launch_prepared_merge_batch`.
The candidate has not run through all 56 model heads,
other query windows or real model/video denoising.

Each bank's `EXCL/IDX` mask fingerprint matches the
preceding far numeric studies:
- Bank A: `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`.
- Bank B: `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`.

### Exact full-window numerical results

The unchanged original one-pass `_tile_kernel(PASS=0)` plus
unchanged Merge was independently verified against the
prepared Graph window output, with **0/1,048,576 output
bit mismatches** on each bank. The unmasked research
`mirror` also matched original partials and Merge on
both banks before the mask-G candidate was evaluated.

On **each** independent bank the mask-G candidate
achieved exactly zero raw FP32-bit mismatches:

| Field | Elements checked per bank | Bank A | Bank B |
|---|---:|---:|---:|
| PM partials | 32,768 | 0 mismatches | 0 mismatches |
| PL partials | 32,768 | 0 mismatches | 0 mismatches |
| PACC partials | 4,194,304 | 0 mismatches | 0 mismatches |
| Final original Merge (11-token tail) | 1,048,576 | 0 mismatches | 0 mismatches |

No error tolerance was applied. Both individual-bank
first complete and separate bank-B eager/Graph parity
gates also PASS, and the banks have distinct selection
masks. The original far BN64 groups, exact near order,
four split boundaries and original Merge are retained.

Actual compiled Tile resources on BOTH banks:

| Kernel | Registers/thread | Shared bytes/CTA | Spills |
|---|---:|---:|---:|
| Original full Tile | **255** | **49,664** | **0** |
| Unmasked research mirror | 254 | 49,664 | 0 |
| Full mask-G Tile | **255** | **49,664** | **0** |

This means the full mask-G research kernel had **no
measured additional register/SMEM allocation compared
with the full baseline**, and no compiler spills.
It does **not** establish equal runtime or achieved
occupancy; the generated machine code may still
schedule masked memory requests differently.

### Decision: numerical PASS, authorize only a separate performance gate

`timing_scope=NUMERIC_ONLY_NO_CANDIDATE_TIMING`,
`speed_measurement=UNRUN_NUMERIC_ONLY`,
`no_cuda_event_timing=true`,
`speed_capture_blocks=[]`,
`production_promotion=PROHIBITED_RESEARCH_ONLY`,
H3 model/video UNRUN.

**PASS the pinned full-window numerical gate for mask-G.**
It is now reasonable to design a **research-only**
independent-capture Tile A/B/B/A on the same true prepared
QKV source states, with original full Tile and unchanged
mask-G full Tile, both protected by the existing two-bank
strict numeric gate. Do not conflate original full Tile
resource counters with the earlier far-only resource figures.

Require deterministic independent captures and repeated
measurements, complete per-arm PM/PL/PACC rechecks, explicit
compiler register/SMEM/spill accounting and a pinned identical
2,048-row window. Use the previous study's **>=5% consistent
Tile-only speedup criterion** before considering expanded
full attention coverage. Distinguish Tile-only event timing
from full call, preload/selection and model/video inference.

If mask-G is slower or inconclusive on A/B/B/A, report
NO_GO_TILE_ONLY or INCONCLUSIVE rather than presenting a
theoretical 53.7% reduction in eligible G leaf loads as
a speedup. No pack buffers are needed for this method;
Tensor Core far-dot block count remains 1,920.

The present result is a documentation-only update: no
new GPU experiments, production modifications, or source
kernel edits were performed while recording it.
