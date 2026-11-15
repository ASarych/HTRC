# HTRC: ordered far-index pack cost — strictly research only

Date: 2026-11-15. Branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Exact implementation parent HEAD:
`fd16b0f3f261e27b11282a281894cbe1dd99d40a`.

## Evidence and bounded question

The actual RTX 5070 Ti `flat_far_exclusion_probe.json` from
`5586c61f8da3bbc65875f15f457afb893a306b4e` showed:

- Two independently generated BF16 QKV banks, full Graph/eager
  strict parity PASS and exact `EXCL`/`IDX` mask membership PASS.
- Four heads × 32 tile centers, 954 full far leaves/center,
  512 near-selected, hence 442 eligible far leaves/center.
- `BN=64`, 4 original far splits of aligned `LPS=256`.
  Each center's baseline far loop makes 15 dot-block iterations.
- Per bank there are 1,920 far dot blocks, **zero** fully
  excluded blocks. Skip-empty-block logic would skip nothing.
- An idealized *split-preserving* packing would require
  1,056 or 1,049 far BN64 dot blocks respectively, a
  theoretical block-count reduction of 45.0–45.4%.
  This is NOT attention speedup or measured DRAM savings.

The new narrow question: how much **additional GPU time and
buffer memory** does it cost to create an ordered, contiguous
list of far-eligible leaf indices per existing tile/split,
using the REAL Graph-prepared selection masks?

Do not implement attention gathering, reordered G/VB storage,
new dynamic far dot loops or a quality/softmax change yet.
These have a distinct, unresolved **bit-exact PM/PL/PACC**
challenge from changing BN64 online-softmax association.

## Standalone prototype only, not an attention kernel

Files in this slice (all new, no modification of current HTRC):

- `tools/htrc_far_pack_triton.py`: **one research GPU kernel**.
  Grid `(128,4)` for flattened (head × tile, original split).
  Each CTA reads up to 256 binary `EXCL` flags. A Triton
  `tl.cumsum` of `EXCL == 0` assigns unique contiguous
  offsets to eligible far leaves, ordered by increasing
  ORIGINAL leaf index. Valid positions `[0,count)` are
  written to `PACKED`; `count` is separately written to
  `COUNTS`. A mask guards split 3's 70 padded leaf slots.
- `tools/analyze_h3_far_pack.py`: torch-free CPU reference
  packing and exact GPU output-prefix validation. Calls
  the previously admitted `summarize_exclusions` so `IDX`
  must match the prepared `EXCL` mask exactly.
- `tools/benchmark_h3_far_pack_cost.py`: standalone
  strict two-bank Graph source gate and GPU pack timings.
  Import of Triton is deferred until GPU execution,
  allowing wrong HEAD to fail closed without CUDA.
- `tools/test_h3_far_pack_cpu.py`: CPU reference, padding,
  split, invalid-count, stale-prefix, source and runner tests.
- `tools/run_h3_far_pack_safe.ps1`: same proven disposable
  Git checkout, exact HEAD, ComfyUI/Blender guards,
  CPU tests before GPU, one JSON.

**No changes to** `h3/htrc_flat_triton.py`, the working
materialized Tile32, Stats BT16, near/far math, top-k,
CUDA Graph dispatch, ComfyUI, model runtime or production.

### Output semantics and size

- `PACKED`: contiguous int32 `[4,32,4,256]`, capacity
  131,072 elements = **524,288 bytes** (0.5000 MiB).
- `COUNTS`: contiguous int32 `[4,32,4]`, 512 elements
  = **2,048 bytes** (0.00195 MiB).
- Total allocated output capacity **526,336 bytes**
  (0.50195 MiB) per one four-head 2,048-query-row window.
  No G/VB value tensor is copied or repacked here.
- **IMPORTANT:** `PACKED[...,count:256]` is deliberately
  **uninitialized** and MUST be masked with `COUNTS` by any
  future consumer. The prefix is completely overwritten
  on every replay. This avoids an additional memset kernel
  from falsely inflating the isolated index-construction
  cost. The CPU validator intentionally ignores the suffix.
- The original split boundaries are preserved. Counts may
  vary per center/split. No dynamic reordering across splits.

## Exact GPU protocol

Target: actual NVIDIA RTX 5070 Ti SM120 (16 GiB), pinned
15,275 BF16 QKV tokens, four heads, balanced/512,
32×64 query rows and materialized leaf states.

1. Create two different independent seeded BF16 QKV banks.
2. For each bank, run the **existing unmodified** complete
   Graph attention first call, verify existing exact eager
   parity, and locate its original Graph-prepared
   `[4,32,954] EXCL` and `[4,32,512] IDX` tensors.
3. Check binary EXCL, unique/in-range IDX, exact top-k mask
   equality, and distinct bank output/mask fingerprints.
4. For each bank, run new pack index kernel once, copy its
   `PACKED` + `COUNTS` arrays to CPU **outside timing**.
   Compare each of 512 valid packed prefixes against a
   strict CPU reference list with the same original split
   membership and increasing leaf order. Check count,
   expected 56,576 valid far indices overall, and the
   idealized packed-block geometry. A mismatch FAILS CLOSED.
5. Check the bank B **full** eager vs Graph output independently
   before taking ANY packed-kernel timing sample.
6. Only then create four **independent CUDA Graph captures**
   of the same index pack on bank B with separate output
   buffers. Each gets five warmups, seven CUDA event samples
   of **16 consecutive replays per sample** (event milliseconds
   divided by 16), and output prefix checks before capture,
   after warmups, and after timing. This measures the
   incremental pack-kernel GPU event cost for the ONE window.
7. Return four capture medians and their median, plus two-bank
   correctness and output bytes. Do not compare directly
   with older independent Tile timing as an end-to-end speedup.
   CPU source generation, full Graph and eager parity gates,
   transfer-based validation, and input QKV preparation are
   **not included** in the timed index-kernel cost.

The pack kernel never reads G/VB/K/V and is NOT an attention
candidate. Therefore: `no_packed_attention_kernel=true`,
`no_attention_candidate_timing=true`, and H3/video UNRUN.

### Fail-closed and one report

Exactly one result path per user run:

`D:\ComfyOutput\H3\math_probe\flat_far_pack_cost.json`

The previous `flat_far_exclusion_probe.json` is preserved.
Success status: `PASS_PACK_COST_ONLY` (correct indices and
measured packing cost only; NOT a packed-attention speed PASS).
Any invalid mask, incorrect index/count, source SHA mismatch,
CUDA/Graph error or missing Triton returns `FAIL_CLOSED` with
`completed=false` and traceback in the same JSON. CPU preflight
failure refuses GPU before running the benchmark.

The exact pinned HEAD runner is:

`tools/run_h3_far_pack_safe.ps1 -ExactHead <40_SHA> -Rounds 7`

## Status at implementation

CPU reference and source tests: **11 PASS, 1 SKIP** locally;
the one full-source HTRC contract test requires the detached
user checkout and remains mandatory before GPU. Python
syntax/compile PASS. False-head smoke with a temporary Git
repository returned one `FAIL_CLOSED` JSON and exit code 2,
without importing CUDA or Triton.

Actual RTX 5070 Ti Triton kernel compile/Graph timing:
**UNRUN**. No measured pack overhead yet. Do not infer
packed attention numeric compatibility or performance.
No production promotion, no paid inference.
