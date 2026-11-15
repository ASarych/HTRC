# HTRC: ordered compacted far QK/PV numeric gate (not a speed experiment)

Date: 2026-11-15. Research branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Base branch commit: `a804f3bb1366ca2c5bce8e532548ba375f4e6d5c`.

## Why this implementation is bounded

User RTX 5070 Ti experiments on the pinned four-head, 15,275-token,
Balanced/512 materialized Tile32 window have established:

- 512 exact near leaves/center; 442 far leaves/center out of 954
  real full leaves; 4 original split partitions, 64 far leaf lanes/dot.
- There are 1,920 dense far dot blocks per 128 tile centers, and **zero**
  entirely excluded blocks. Simple whole-block skipping was NO_GO.
- Ordered per-split packed indices correctly represent the 56,576 far
  leaves of one window: theoretical **1,056 / 1,049** packed blocks
  on the independent banks. That is a 45–45.4% block-count lower bound,
  NOT performance.
- Pack-only GPU Graph event median **2.209 microseconds/window**.
- Pack-guided five-field data copies on a separate test: dense 127.669
  vs packed 18.432 microseconds/window. The different valid row counts
  make this an *unequal-work copy test*, **NOT** a 6.93x attention gain.

New question: when those leaves are packed into new consecutive BN64
groups, does the genuine **far QK, PV and online softmax** produce the
**identical FP32 bit patterns** to the existing `_tile_kernel`?

### Strict two-bank implementation

Two new research files and their safe runner/CPU contracts:

- `tools/htrc_packed_far_numeric_triton.py`: experimental Triton
  `packed_far_partial` kernel. Reads actual `PACKED/COUNTS`, original
  `Q/MU/G/LZ/VB/SG`, computes identical INT8 query quantization and
  INT8 QK, scales scores via `SG` and `LZ`, performs the same online
  max/exp/FP32 sum, and BF16 PV dot. It writes **far-only**
  `PM/PL/PACC` into separate scratch buffers. No production import.
- `tools/benchmark_h3_packed_far_numeric.py`: creates two independent
  seeded BF16 QKV banks, checks the unchanged full first-call Graph
  versus eager and separate bank-B eager parity; extracts real
  materialized state from the same `(tiles=32,valid_rows=2048)` window.
  It audits `EXCL/IDX`, checks the known bank mask hashes, builds ordered
  packed indices with the previously successful GPU pack kernel, and
  validates **only initialized packed prefixes and counts** against the
  CPU oracle. It never reads uninitialized suffix values.
- The baseline is the **original unchanged**
  `h3.htrc_flat_triton._tile_kernel` with `PASS=1`, preserving far-field
  math and the original BN64 groupings. `PASS=1` skips near, so both
  sides are genuinely **far-only** before the exact-near loop.
- Candidate is the independent new packed-far kernel, same 512 CTA grid
  `(128,4)` and 4 original leaf splits, materialized STATS_BT16,
  `RB1`, 4 warps, two stages, BF16 PV, INT8 QK, 64 query rows.
- For every field `PM`, `PL` and `PACC` and **each of two independent
  banks**, compare `float32.view(int32)` values bit-for-bit, not
  `torch.equal` of floats (which would equate signed +/- zero).
  JSON records bit-mismatch count, total elements, location of first
  mismatch, and maximum finite absolute difference for each field.
- A numerical mismatch is `status=FAIL_CLOSED`, `completed=false`,
  `parity_admission=NO_GO_FAR_BIT_EXACT`, exit code 2. Both bank
  diagnostics are preserved; the Windows safe runner recognizes a
  valid scientific negative **without reporting a speed success**.
- If both bank far partials unexpectedly match bit-for-bit:
  `status=PASS_FAR_NUMERIC_ONLY`, `parity_admission` indicates that
  the whole far+near PASS=0 / merge remains untested.
  **No GPU timing is performed even on this positive far-only gate**:
  the next independent slice must first establish complete near+far
  partial parity before an A/B/B/A is justified.
- GPU/compiler/runtime errors, source SHA mismatch or stale selection
  masks are also FAIL_CLOSED. No permissive float-error threshold.

### Why zero-tolerance is difficult

Original dense online-softmax processes valid leaves in original BN64
column groups, with excluded scores masked to negative infinity after
QK. The packed candidate moves valid leaves across **original BN64**
group boundaries; this changes which terms enter each `tl.max`,
`tl.sum`, and BF16 `tl.dot` PV group. Even if all source inputs and
split membership are bit-exact, FP32 reduction and Tensor Core rounding
association need not be. This prototype is expected to be falsifiable
at the numerical gate and is **not presumed to pass**.

A NO_GO at this gate means the present cross-block packing cannot be
substituted under the existing strict equality contract. It does not
prove that *all* sparse attention methods are impossible. Do not relax
the gate, silently promote approximately equivalent math, or revive
prior failed RB/late-Q/stage1 experiments. Preserve the working Tile32
Baseline (Stats BT16, materialized VB, RB1, two stages).

### Execution and safety

`tools/test_h3_packed_far_numeric_cpu.py`: geometry/membership,
original-vs-compacted BN64 grouping, static kernel/runner
source checks, fail-closed wrong-HEAD smoke, and mandatory full-repo
source contract in the user's exact checkout.

`tools/run_h3_packed_far_numeric_safe.ps1`: disposable exact-HEAD
checkout, ComfyUI/Blender guards, CPU suite before CUDA, one JSON.
One user-facing output only:

`D:\ComfyOutput\H3\math_probe\flat_packed_far_numeric.json`

Old gather and index reports are preserved; real H3/model/video,
complete compacted attention and speed A/B/B/A all UNRUN.
No edits to `h3/`, CUDA Graph production dispatcher, ComfyUI,
model weights, or online quality contract.

## Implementation test status

- CPU-only local suite: **9 PASS, 1 SKIP** (skip requires complete
  exact-head HTRC source, mandatory on Windows runner).
- Local Python `py_compile` for all new Python files: PASS.
- Wrong-HEAD smoke in an initialized temporary Git repo: PASS;
  `FAIL_CLOSED` JSON, exit 2, no CUDA/Triton import.
- Actual RTX 5070 Ti candidate compile, two-bank numeric parity,
  far-only GPU partial comparison and complete attention: **UNRUN**.
- No performed GPU timing, no observed speedup or production promotion.
