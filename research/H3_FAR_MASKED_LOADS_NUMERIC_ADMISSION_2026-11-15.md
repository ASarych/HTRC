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

## 2026-11-15 — measured far-mask PASS, resource-pressure caution

Evidence: user-uploaded `flat_far_masked_loads_numeric.json`,
**9,858 raw bytes**, SHA-256
`252e4680532685cf4a20d2bac7db5a577226e93f0ef95eba7f8240d1f1b1cbc1`.
Pinned exact source HEAD:
`57b6c48667f3d0cafc141e5816acc0df43e23c06`.
The raw uploaded JSON remains user-held; this commit
records the exact source fingerprint and result fields.

Status: `PASS_MASKED_FAR_NUMERIC_ONLY`,
`completed=true`, schema
`htrc-far-masked-loads-numeric-v1`, admission
`PASS_BOTH_MASKED_FAR_ONLY`.
Both independent BF16 QKV banks passed full original
first-call eager/Graph bitwise parity and separate
bank-B full eager/Graph bitwise parity. Their
`EXCL/IDX` selection hashes match earlier probes:
A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`,
B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`.
Both banks used the original 1,920 far BN64 blocks,
identical split boundaries and 512 near-selected leaves.

### Strict two-bank far-only raw-bit results

Control: unchanged original `_tile_kernel(PASS=1)`.
Research mirror: identical BN64 and arithmetic, both
load masks disabled. Three masked options retain
the exact same BN64 dot/online-softmax groups.

All rows below mean **zero raw FP32-bit mismatches**
on **each** QKV bank, not merely a floating point
tolerance or an approximate quality comparison.

| Candidate | Bank A PM/PL/PACC | Bank B PM/PL/PACC | Registers/thread | Shared B/CTA | Compiler spills |
|---|---|---|---:|---:|---:|
| Original PASS1 | Reference | Reference | 212 | 41,472 | 0 |
| Research `mirror` | **PASS / PASS / PASS** | **PASS / PASS / PASS** | 204 | 41,472 | 0 |
| `mask_g` | **PASS / PASS / PASS** | **PASS / PASS / PASS** | 255 | 41,472 | 0 |
| `mask_vb` | **PASS / PASS / PASS** | **PASS / PASS / PASS** | 255 | 41,472 | **12** |
| `mask_both` | **PASS / PASS / PASS** | **PASS / PASS / PASS** | 255 | 41,472 | **8** |

For each candidate on each bank, the test inspected
32,768 PM elements, 32,768 PL elements and
4,194,304 PACC elements; every mismatch count = 0
and every maximum finite absolute difference = 0.
The mirror requirement passed independently and
is not inferred from candidate parity.

### Resource and performance interpretation

The masked loads removed reads of some far leaf
rows for which `EXCL=1`. The observed number
of selected near leaves implies a potential 53.6688%
reduction of **logically enabled G/VB leaf loads**,
but this experiment did NOT measure issued GPU
memory transactions, L2/DRAM traffic, Tensor Core
instruction counts or complete attention runtime.

All three masked variants use 255 registers/thread,
up from 212 for original PASS1. The `mask_vb` and
`mask_both` variants also have reported compiler
spills (12 and 8 respectively). The increased
register pressure can negate the traffic reduction;
spills are an adverse resource signal, NOT a
measured slowdown. Even `mask_g` (no spills)
may be slower because 255 registers/thread can
limit scheduling. Shared-memory cost is unchanged.
Do not infer a speedup from either strict numerics
or the preceding copy-only far-gather study.

Numerical admission is **far-only**. The user JSON
explicitly says `full_masked_attention=UNRUN`,
`speed_measurement=UNRUN_NUMERIC_GATE_ONLY`,
`no_cuda_event_timing=true`,
`speed_capture_blocks=[]`,
model/video UNRUN and production promotion prohibited.

### Next admissible experiment

Prioritize **`mask_g` first** as the only new
no-spill option. Keep `mask_vb` and
`mask_both` available as numerically accepted
research variants; do NOT pronounce performance
NO_GO without data. Before GPU timing:

1. Check full far+near `PASS=0` against unchanged
   original full Tile `PM/PL/PACC` on both banks,
   including exact selection and Merge output.
   Any two-pass diagnostic must first prove that
   the unchanged split `PASS=1`/`PASS=2`
   reconstruction itself matches one-pass baseline;
   otherwise it cannot serve as a valid parity oracle.
2. If full numeric parity holds, perform independent
   CUDA Graph A/B/B/A for `mask_g` against original
   on identical prepared states, including a resource
   and spill gate and at least 5% consistently
   improved Tile-only timing criterion.
3. Only if Tile timing improves, investigate the
   complete attention call, pack/prepare overhead,
   VRAM and actual H3 model. No production write
   or checkpoint inference until these independent
   gates pass.

This result was documented only. No new GPU execution,
new kernel edit, runtime change or production promotion
occurred during this recording slice.
