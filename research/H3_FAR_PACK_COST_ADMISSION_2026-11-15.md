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


## 2026-11-15 — measured GPU pack index cost on RTX 5070 Ti

User-uploaded \`flat_far_pack_cost.json\`, **5,564 raw bytes**,
SHA-256 \`9dfc5430e66f1a0ad6b434df2bac670d698f9b0ca911304e6fd0d01181595dca\`.
The source was run at exact HEAD
\`e3b49725409ecc1258ef7d36a06f68f8fd6a04cf\`.
The raw JSON is not committed; this section logs the source
fingerprint and its verified measurements.

### Strict result gates

- \`schema=htrc-far-pack-cost-v1\`;
  \`status=PASS_PACK_COST_ONLY\`, \`completed=true\`.
- \`probe=GPU_ORDERED_FAR_INDEX_PACK_COST_ONLY\` with 15,275
  tokens, 4 heads, 32 tiles, 4 splits, far BN64, 512 near
  leaves; source: true prepared Graph \`EXCL/IDX\` from
  two independent BF16 QKV banks.
- Bank A selection mask SHA-256
  \`bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32\`.
  Bank B
  \`05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264\`.
  These equal the earlier actual exclusion-geometry source
  masks, not one replicated or stale bank.
- Both banks: \`valid_prefixes_bit_exact=true\`,
  \`split_membership_and_order_exact=true\`,
  \`padded_suffix_not_read=true\`; each packs **56,576**
  valid original far-leaf indices. Bank A reproduces the
  optimistic 1,056 far block lower bound; bank B 1,049.
  The original far workload has 1,920 blocks per bank.
- First-call and separate bank-B full Graph/eager parity
  both PASS, distinct QKV/mask banks PASS.
- \`no_packed_attention_kernel=true\`,
  \`no_attention_candidate_timing=true\`,
  \`production_modified=false\`; model/video UNRUN.

### One-window pack kernel event timings

Four separate CUDA Graph captures on bank B, each with
7 CUDA event samples of **16 complete graph.replay() calls
per sample**, then divided by 16. This is a pack-only
kernel cost with prepared mask inputs and output buffers
resident; it does NOT include mask/top-k generation,
compilation, Graph capture, CPU validation/transfers, or
the compacted attention candidate.

| Capture | Median GPU-event time per pack (ms) | Per pack (microseconds) |
|---|---:|---:|
| 1 | 0.002208000 | 2.208 |
| 2 | 0.002184000 | 2.184 |
| 3 | 0.002212000 | 2.212 |
| 4 | 0.002210000 | 2.210 |
| **Median of four capture medians** | **0.002209000** | **2.209** |

In each capture the first event was a high outlier
(approximately 7.802 / 8.230 / 11.270 / 7.434 microseconds
per pack), whereas the remaining events were near
2.16–2.22 microseconds. Each capture median is robust to
this one outlier. No outliers were manually deleted from
the reported medians.

Pack output allocation: 524,288 bytes of int32 \`PACKED\`
capacity + 2,048 bytes of int32 \`COUNTS\`, total
**526,336 bytes = 0.50195 MiB** for this single window.
The suffix of \`PACKED\` past \`COUNTS\` is UNINITIALIZED
and must always be masked by a future consumer.

### Decision and scope

**PASS_PACK_COST_ONLY**: a correct ordered original-split
far-index packer is inexpensive *in this one-window isolated
Graph measurement*. For scale only, 2.209 microseconds
is about 0.57% of the approximately 0.385-ms previous
standalone baseline Tile window from a DIFFERENT
experimental run; this is not a paired time ratio and
not a measured full-attention overhead.

This positive narrow result removes one candidate
feasibility concern (index-pack construction cost) but
does NOT establish that compacted far attention is
faster, numerically bit-exact, or profitable across
all windows, heads, selection/gather memory traffic or
real model-resident inference.

**Next research gate, NOT yet implemented here**:
measure the actual indirect \`G/VB/LZ/EXCL/SG\` gathers
against coalesced baseline loads on identical EXCL/IDX
states, keeping the same original split membership and
valid-prefix masks. Collect resource and bytes/timing
cost, not an invented attention speedup. The crucial
subsequent gate is exact \`PM/PL/PACC\` parity of the
far online-softmax computation. Moving leaf columns
across original BN64 blocks changes the accumulation
association and may violate the fixed bit-exact contract.
Do not relax parity or silently promote an approximate
result. If exactness is impossible at acceptable cost,
close this compaction direction without modifying
production.

No additional GPU experiment, new kernel edit, runtime
modification, paid inference, or production promotion
was performed when recording this uploaded result.
