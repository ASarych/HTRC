# HTRC full Tile32 mask-G — independent-capture A/B/B/A admission

Date: 2026-11-15. Branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Parent HEAD at implementation: `a1b731aa56d06b511ed18e8549b8414dd8db0f79`.

## Reason for this narrowly scoped GPU experiment

Two independent RTX 5070 Ti BF16 QKV banks previously established strict
raw-FP32-bit identity of all 4,194,304 PACC, 32,768 PM, 32,768 PL and
1,048,576 original-Merge output values **per bank** for the complete
`mask_g` Tile against the unmodified one-pass materialized Tile.
The original merged output was also verified against the native Graph
window. The candidate has 255 registers/thread, 49,664 B shared/CTA,
zero spills, identical to the original full-Tile compiler resources.

Unlike rejected cross-BN64 packing, mask-G preserves the 1,920 original
BN64 far dot blocks, four split boundaries, the entire near pass,
softmax accumulation order and the 11 exact tail tokens. It simply
avoids some excluded G loads in an otherwise unchanged far loop.
No previously reported copy-only ratios are projected onto Tile time.

**This slice may measure Tile-only speed only AFTER repeating all exact
numeric source gates on TWO independent QKV banks and verifying an
unchanged compiler resource envelope.** It does not promote a kernel
or benchmark complete video inference.

## Research-only files

- `tools/benchmark_h3_full_mask_g_tile_abba.py`: reuses the existing
  proven candidate `htrc_full_mask_g_triton.full_mask_g_partial` and
  `benchmark_h3_full_mask_g_numeric` helpers. It does not implement
  another attention, partial-Merge, or memory-gather kernel.
- `tools/test_h3_full_mask_g_tile_abba_cpu.py`: pure CPU speed decision
  tests, resource gates, SHA/fail-closed source gates, one-JSON runner
  contract, and exact-repository baseline source checks.
- `tools/run_h3_full_mask_g_tile_abba_safe.ps1`: isolated exact-HEAD
  Windows runner with ComfyUI/Blender preflight, the previously admitted
  full-mask-G CPU suite and new A/B/B/A CPU suite *before CUDA*, and a
  single output report.
- This research admission document.

There are **no changes to h3/**, CUDA Graph production dispatcher,
ComfyUI installation, model checkpoint, source selection, GPU kernels,
existing Python numeric probes, or baseline configuration.

## Pinned GPU input and repeated strict numeric gates

Target: NVIDIA RTX 5070 Ti SM120 (16 GiB), 15,275 BF16 QKV tokens,
heads=4, 32 tiles x 64 query rows, source full-Graph materialized
Balanced/512 / STATS_BT16, far BN64, near BN64, four splits,
11 exact tail tokens and original head-major state buffers.

1. Create two independent seeded QKV banks and obtain both real
   native Graph outputs and prepared `EXCL/IDX/G/VB/LZ/SG/Q/K/V` states.
   Verify original first-call complete eager/Graph bit parity and
   the separate bank-B complete eager-vs-Graph result.
2. Verify each bank's `EXCL/IDX` invariants, exactly 1,920 original
   far dot blocks, zero entirely excluded blocks, and pinned
   bank A/B mask SHA-256 fingerprints:
   - A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`
   - B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`
3. Re-execute the unchanged original one-pass complete Tile using
   `batch.launch_prepared_tile_batch`, then original `Merge` with
   11 tail tokens; verify its output is raw FP32-bit identical to
   the corresponding native Graph window.
4. On **both** banks, re-execute the research unmasked mirror and
   `mask_g` full-Tile kernels. Compare every raw FP32 PM/PL/PACC bit,
   AND the output of unchanged original Merge. An error or mismatch
   stops the experiment **before capturing any timed Graph**.
5. Require original full Tile compiler metadata pinned to 255
   registers/thread, 49,664 shared bytes/CTA and 0 spills; require
   `mask_g` to use no more registers or shared bytes and to have
   zero spills. Any missing or changed metadata -> `NO_GO_RESOURCE_GATE`
   with no speed measurements, even if numeric parity passes.

## Independent-capture timing — ONE 2048-query-row window

Four independent CUDA Graph captures **A/B/B/A** using identical
original prepared bank-B Graph state:

A = original full Tile; B = research full `mask_g` Tile.
Each capture gets separate 4-head partial output buffers; 5 warmup
Graph replays; 7 CUDA Event samples (configurable 7..12), with
**16 Graph replays per sample** and per-Tile ms = elapsed_ms/16.
The only timed operation is the **full Tile kernel**. Existing Merge,
source preparation, top-k/selection, graph capture/compile, CPU transfers,
partial allocations, warmup and all bit-exact checks are **outside** the
recorded CUDA event window. This is not a complete attention measurement.

Before capture, after warmups and after timing, revalidate each arm's
PM, PL, PACC **and original merged output** against a frozen original
bank-B partial/merge reference. A mismatch is a fail-closed numeric
NO_GO and invalidates any speed claim.

The decision takes the two independent capture medians for each arm:

- **PASS_TILE_ONLY_SPEED_GATE:** stable inter-capture spread <=3% for
  both arms, median-of-two time reduction >=5% AND the slowest
  candidate capture is >=5% faster than the fastest control capture.
- **NO_GO_TILE_ONLY:** both arms stable but the strict 5% criterion
  fails, including slower or negligible improvement cases.
- **INCONCLUSIVE_TILE_ONLY:** either arm's two capture medians differ
  by more than 3%; do not present this as a speed improvement.
- **NO_GO_RESOURCE_GATE:** unacceptable compiler resource drift;
  nothing is timed.
- **FAIL_CLOSED:** missing exact HEAD/source, invalid Graph/selection,
  or any two-bank or replay numeric parity mismatch. No performance
  admission is permitted.

Even a Tile-only PASS is **not** permission to promote to production:
the next independent gate would need native whole-attention A/B/B/A,
VRAM/peak memory and real 56-head H3 model-resident measurements.
Do not turn a Tile-only ratio into a model/video speedup. No new
attention math or alternative approximation threshold is authorized.

## One JSON and execution status

Windows runner:
`tools/run_h3_full_mask_g_tile_abba_safe.ps1 -ExactHead <SHA> -Rounds 7`

Only user-facing report:
`D:\ComfyOutput\H3\math_probe\flat_full_mask_g_tile_abba.json`

The report includes source SHA, both QKV source bank masks and numeric
parity, real per-arm compiler resources, all four independent captures,
all seven individual event samples per capture, pre/post numeric checks,
control and candidate medians, inter-capture spread, worst-pair ratio
and a conservative status. Prior experiment JSONs are untouched.

Implementation-time CPU unit tests, exact-HEAD static checks and
negative-path smoke are required before CUDA. At authoring time, the
RTX 5070 Ti GPU A/B/B/A, actual full Graph model/video and production
integration are **UNRUN**; there is **no measured Tile speedup** yet.
