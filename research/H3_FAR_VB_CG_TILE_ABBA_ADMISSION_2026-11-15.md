# HTRC: isolated far Vbar L1-bypass cache policy, full Tile32 A/B/B/A

Date: 2026-11-15. Research branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Starting exact HEAD: `03abe76755eab307144ec4e9121a31de88d53a98`.

## Evidence and bounded hypothesis

Existing accepted RTX 5070 Ti SM120 measurements on 15,275 BF16
tokens, four heads, materialized Balanced/512, Tile32, 2,048 query rows
and STATS BT16:

- The unchanged Tile uses 255 registers/thread, 49,664 shared B/CTA,
  no compiler spills and about 15.57% achieved occupancy in the
  earlier 12-node Nsight replay. Nsight Tile median duration
  422.96 us and median DRAM throughput 21.73% of peak were
  **profiling replay statistics**, not normal CUDA Graph time.
- Among sampled Stats/Rowmax/Tile nodes, Tile had 81.8% of the
  sampled L2 TEX read sectors, but those sectors are **not L1-hit-rate
  data**. The Tile is not proven L1- or DRAM-bandwidth limited.
- Original 2,048-row full Tile Graph A/B/B/A control median
  383.251 us; research mask-G candidate 369.470 us (3.5958%
  time decrease), stable numerical PASS but **NO_GO_TILE_ONLY**
  against the pre-registered >=5% admission bar.
- Cross-original-BN64 leaf packing was **NO_GO_FAR_BIT_EXACT**.
  Stage1/2 and RB2/late-q resource/occupancy-only probes were NO_GO.
  None is resurrected by this experiment.

**New independent question**: does routing far BF16 Vbar reads
through L2 with `tl.load(..., cache_modifier=".cg")`, bypassing L1,
yield a meaningful full Tile speedup compared with the original
compiler/default load policy? This changes cache *hints only*, not
the number/order of far Tensor Core dots or the original softmax.
It may be slower if L1 reuse is important or if the compiler's
scheduling changes. This is an isolated cache-policy hypothesis,
**not** a combined mask-G/cache experiment or a memory-savings
claim.

Triton documents load `cache_modifier=""` (default) and `".cg"`:
https://triton-lang.org/main/python-api/generated/triton.language.load.html
The user environment from previous inventory was Triton 3.6.0;
actual installed `".cg"` support still needs CUDA compilation.
No TMA multicast or inter-CTA cluster support is implied.

## Research-only files and exact code boundary

- `tools/htrc_full_far_vb_cg_triton.py`: separate copy of
  the already numerically admitted original far+near full-Tile
  research mirror. Exactly ONE load, the far BF16 `VB` load, gets
  `cache_modifier=CACHE_VB`, with compile-time values `""` or
  `".cg"`. G/LZ/SG/EXCL and the complete exact-near K/V/K8
  loads have no new cache hints. The original BN64 ordering,
  INT8 QK, BF16 PV, FP32 partials and all 512 near leaves remain
  unmodified. The existing original Merge and 11-token tail
  are used directly, not copied.
- `tools/benchmark_h3_far_vb_cg_tile_abba.py`: separate
  isolated benchmark based on the admitted mask-G A/B/B/A
  harness. All source/Graph/eager, selection-mask and exact
  numerical gates are retained. Three arms: unchanged
  original control, research `mirror` with default
  cache_modifier, and research `far_vb_cg` with `".cg"`.
  Only control and far_vb_cg are timed; mirror is a prior
  correctness oracle. No model/video inference.
- `tools/test_h3_far_vb_cg_tile_abba_cpu.py`: synthetic
  A/B/B/A decision and resource rules, exact cache-hint
  source contract and wrong-HEAD smoke, without importing
  GPU on an invalid SHA; full-source gate must run in
  exact-HEAD checkout.
- `tools/run_h3_far_vb_cg_tile_abba_safe.ps1`: detached
  exact-SHA Windows clone, ComfyUI/Blender guard, two mandatory
  CPU test suites before GPU, exactly ONE final JSON.
- This decision document.

All changes are research-only: **no changes to h3/**,
ComfyUI, CUDA Graph production dispatcher, model checkpoints,
baseline K/V/Q state, selected near leaf set or main.
Do not change a production kernel merely to try the cache hint.

## Strict numerical and resource gates BEFORE ANY timing

Fixed request: RTX 5070 Ti SM120; 15,275 tokens;
4 BF16 heads, 32 tiles x 64 rows, Balanced/512;
4 splits, far BN64, near BN64; materialized VB;
STATS_BT16, 11 tail tokens.

1. Two separately seeded BF16 QKV banks must reproduce
   first-call full original eager/Graph strict bit parity,
   separate bank-B full eager/Graph parity and **different**
   source selection masks. Mask SHA-256 fingerprints must
   match earlier accepted runs:
   - A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`
   - B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`
2. Independently recompute unchanged full Tile PM/PL/PACC
   and call original `launch_prepared_merge_batch`.
   Its merged output must match the native captured Graph
   source window bit-for-bit (including the 11-token tail).
3. Research default `mirror` must match ALL FP32 partial
   bits and original Merge on BOTH banks. Then research
   `far_vb_cg` must match exactly the same fields and
   final Merge on BOTH banks. Do not use approximate
   float tolerance or `torch.equal` for partial-bit checks.
4. Require actual Triton resource metadata: original control
   255 registers/thread, 49,664 shared B/CTA and no spills;
   candidate no more of either resource and zero compiler
   spills. Any mismatch -> `NO_GO_RESOURCE_GATE`, no time.
5. Only after these gates, use real bank-B prepared state
   and four **independent CUDA Graph captures** in strict
   order `control/far_vb_cg/far_vb_cg/control`.
   Five warmups per capture; seven samples (7..12 allowed)
   of 16 Graph replays per event. Revalidate PM/PL/PACC
   and original merged output **before capture, after warmups,
   and after GPU timing**. A mismatch invalidates speed.

## Admission and reporting

Timing is ONE complete 2,048-row Tile kernel window only.
CUDA event excludes preparation, original Merge, GPU output
validation, compilation/capture and model/video denoising.

- `PASS_TILE_ONLY_SPEED_GATE`: independent arm capture
  medians stable within 3%, both robust median reduction
  and worst cross-capture pair show **>=5%** speed gain.
- `NO_GO_TILE_ONLY`: stable measurements but speed criterion
  not met; report even a positive <5% speed gain as NO_GO.
- `INCONCLUSIVE_TILE_ONLY`: capture median spread >3%.
- `NO_GO_RESOURCE_GATE`: numeric PASS but compiled resource
  envelope not admitted; zero timed capture blocks.
- `FAIL_CLOSED`: invalid source, GPU, full numerical parity,
  captured output or other uncaught runtime error. Never claim
  completed performance on a FAIL_CLOSED result.

SINGLE output path, not overwriting earlier research JSONs:

`D:\ComfyOutput\H3\math_probe\flat_full_far_vb_cg_tile_abba.json`

The JSON must expose exact tested HEAD, both source masks,
two-bank checks, compiled register/shared/spill metadata,
all four capture blocks and raw timing samples if admitted,
the speed decision and `not_a_full_operator_speedup=true`.
No full attention or 56-head model promotion from one Tile.

## Authoring-time status

Tile-only is the sole timed scope; this research is not production.

Candidate source, conservative benchmark, CPU/source contracts
and Windows runner authored. Static cross-source audit checks
cache hint confined to far VB, exact source SHA guards, arm
ordering and one JSON output: **PASS_STATIC_REVIEW**.

Actual installed-Python syntax/unit suite: **UNRUN** until
the user's exact-head Windows checkout executes it. In this
authoring environment HTRC repository is connector-only and
the user GPU/Triton runtime is unavailable. This is a
limitation, not a claim that the tests passed.

New GPU compile, first-call new mirror and far VB .cg
numeric parity, resource gate, CUDA Graph A/B/B/A timing,
full H3 model/video and production promotion: **UNRUN**.
No new GPU measurement or change to working production.
