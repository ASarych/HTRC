# HTRC: mask excluded G/VB loads, preserving original far BN64 grouping

Date: 2026-11-15. Branch: `experiment/flat-memory-to-speed-balanced-20261113`.
Exact implementation base HEAD: `e79866dd2638d7ce2f51ca4ac477c78253437fb8`.

## Rationale from measured gates

The 2026-11-15 far-exclusion probe found **zero** wholly excluded far blocks
among 1,920 original BN64 blocks (4 heads x 32 tile centers, 4 splits).
Each center has 954 real full leaves: 512 in near and 442 useful far.
Previous split-preserving **cross-original-BN64 compaction** produced
correct indices (2.209 microseconds/window) and cheap packed-only data
copies, but the new compacted far QK/PV **failed strict FP32 partial parity**
on both independent BF16 QKV banks: PM matched while PL/PACC did not.
Do not test its speed or resurrect cross-BN64 packing under the bit-exact
contract. More occupancy variants also have established NO_GO results.

### Different, bounded hypothesis

Do not reorder any original leaf, split or BN64 dot group. Instead use
**the existing EXCL mask to avoid reading G and/or Vbar for lanes whose
far contribution is later removed by `sc = where(EXCL == 0, sc, -inf)`**.
The QK and PV tensor-core operations, online FP32 softmax reductions,
`PM/PL/PACC` storage and original 512-CTA grid remain mathematically
unchanged. Only actual leaf-memory read mask changes; no sparse
Tensor Core speedup is asserted.

For excluded far columns, the original far branch computes a QK score
that is then set to negative infinity, yielding zero softmax probability.
The candidate fills masked G/VB inputs with zero and retains the same
64 physical column slots. This is a different mechanism from packing
columns together. A GPU bit-for-bit test is STILL REQUIRED because
compiler scheduling and invalid-input arithmetic may change.

### Implementation — no changes to h3/

Five new research-only files:

- `tools/htrc_far_masked_loads_triton.py`: independent far-only PASS=1
  reproduction of the fixed-shape existing INT8 QK, BF16 PV and FP32
  online-softmax. The original 64-column blocks and four far splits
  are unchanged. Two Triton constexpr flags, `MASK_G` and `MASK_VB`,
  change **only** the respective leaf load mask from `cm` to
  `cm & (EXCL == 0)`. LZ, EXCL, SG and all score, max, exp, sum and PV
  operations remain in their previous relative order. No modifications
  or imports in normal HTRC production dispatch.
- `tools/benchmark_h3_far_masked_loads_numeric.py`: original production
  `_tile_kernel(PASS=1)` control vs independent dense research clone,
  plus masked-G-only, masked-VB-only and masked-both variants. The cloned
  UNMASKED kernel is checked first to detect errors or compiler drift
  from rewriting the far-only source. **If this mirror does not match
  baseline bits, NO masked variant can be admitted**, even if its bits
  happen to coincide by accident.
- `tools/test_h3_far_masked_loads_numeric_cpu.py`: CPU mask geometry,
  exact fixed block boundaries, AST/source contracts, failure-path smoke,
  and mandatory original source-contract test on Windows full checkout.
- `tools/run_h3_far_masked_loads_numeric_safe.ps1`: exact SHA disposable
  checkout; ComfyUI/Blender guards; CPU tests before GPU; exactly one
  output JSON and robust handling of expected scientific negative.
- This decision document.

Pinned conditions: NVIDIA GeForce RTX 5070 Ti 16GB, SM120; 15,275
BF16 QKV tokens, 4 heads, 32 64-row tiles, Balanced/512, materialized
VB, INT8 G and Q, BN64, 4 original far splits, 4 warps, 2 stages,
RB1, original STATS_BT16 prepared Graph selection. QKV banks are
independent and must reproduce the source mask SHA-256 fingerprints:

- Bank A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`
- Bank B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`

## Exact correctness-first GPU protocol

1. Run original complete Graph attention on BOTH QKV banks, verify
   first-call strict Graph/eager parity, full bank-B eager parity and
   different outputs/masks. Verify EXCL equals the selected IDX list.
2. Run the unchanged source `_tile_kernel(PASS=1)` as an original far
   baseline into separate FP32 PM/PL/PACC buffers for each bank.
3. Run the new unmasked clone (`MASK_G=0`, `MASK_VB=0`), then G-only,
   VB-only and both masked variants; compare **raw 32-bit FP32 bit
   patterns** for each PM, PL and PACC field against the original.
4. Record per-field mismatch count, first multidimensional mismatch
   index, total elements and maximum finite absolute difference.
   This is diagnostic, not a new tolerated numerical threshold.
5. Record compiler registers/thread, shared bytes/CTA and spills for
   each variant, but no CUDA event timing and no benchmark speed claim.
6. Return statuses:
   - Unmasked mirror differs: `FAIL_CLOSED`,
     `parity_admission=NO_GO_RESEARCH_DENSE_MIRROR`;
   - Clone exact, every mask variant fails exactness: `FAIL_CLOSED`,
     `parity_admission=NO_GO_MASKED_FAR_BIT_EXACT`;
   - Clone exact, at least one masked variant exact on BOTH banks:
     `PASS_MASKED_FAR_NUMERIC_ONLY`, listing the exact admissible
     variants. Substatus distinguishes BOTH vs a passing subset.
   Each outcome is recorded in ONE JSON with the strict data.
7. Even a far-only numeric PASS is NOT a full-attention approval:
   near-field and merge parity remain UNRUN, and no performance
   inference is permitted. A separate full near+far numerical gate
   is required before any captured Tile A/B/B/A.

Existing output file names and earlier result JSONs are untouched.
Single new output:

`D:\ComfyOutput\H3\math_probe\flat_far_masked_loads_numeric.json`

The runner accepts an expected negative exit-code 2 when the report
contains two-bank diagnostic proof. A launch/runtime/verification error
is ordinary FAIL_CLOSED and should not be mistaken for an observed
scientific negative. No model checkpoint/video inference is run.

## Preflight and current status

Local CPU tests: **8 PASS, 1 SKIP** (full source contract needs the exact HTRC checkout). Python `py_compile` PASS. Wrong-HEAD `FAIL_CLOSED` smoke PASS; no CUDA imported.
RTX 5070 Ti new kernel compile/numerical outcome: **UNRUN**.
The numeric-only experiment has **no speed measurement**.
No changes to h3/; only new tools/ research/ files, no production or
ComfyUI modifications. Do not promote this untested candidate.
