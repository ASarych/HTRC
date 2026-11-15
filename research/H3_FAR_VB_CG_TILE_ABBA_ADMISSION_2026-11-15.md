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

## 2026-11-15 — measured RTX 5070 Ti far-VB .cg Tile A/B/B/A: INCONCLUSIVE

Source: user-uploaded `flat_full_far_vb_cg_tile_abba.json`,
**23,533 bytes**, raw SHA-256
`30086057d292a1ce8dc46ef75ebee794292a18d782e25d82b96e35c106938692`.
Exact tested source HEAD:
`4f2749f3e93be8975cc790ac3e3f9ff30e04542c`.
The original JSON is not checked into Git; these are extracted
verified measurements and provenance. Report:
`schema=htrc-far-vb-cg-tile-abba-v1`,
`status=INCONCLUSIVE_TILE_ONLY`, `completed=true`,
`numeric_gate=PASS_TWO_BANK_FULL_TILE_MERGE_BIT_EXACT`,
`timing_scope=ONE_WINDOW_TILE_KERNEL_ONLY_NO_MERGE`.

Target: RTX 5070 Ti SM120, four BF16 heads, 15,275 tokens,
one 2,048-query-row window (32 tiles, 64 rows each),
Balanced/512, four far BN64 splits and exact 11-token tail
in the unchanged Merge. First-call full Graph/eager and
independent bank-B Graph/eager source checks PASS.
Both independent selection mask fingerprints are identical
to earlier validated reports:
A `bd56312c61e4c5f950c1a1dfea718faed0ac50c7aa18c0d3c862cd156dee4a32`;
B `05dc41bf73bcc98204739f708a86b6c8e39343f24adb39fa9b952c195a341264`.

### Exact numerical and compiled-resource results

On BOTH independent QKV banks:
- Unchanged original full Tile + original Merge matches
  captured Graph output bit-for-bit.
- Research unmodified `mirror` matches original
  PM/PL/PACC and merged output bit-for-bit.
- Research `far_vb_cg` also matches original PM/PL/PACC
  and merged output bit-for-bit.
- For each bank, 32,768 PM, 32,768 PL, 4,194,304 PACC
  and 1,048,576 Merge FP32 values were checked; **zero raw
  bit mismatches** for each candidate field.
- All four independent captures passed pre-capture,
  post-warmup and post-timing same-field bit-exact
  validations. The performance inconsistency is not
  due to an observed numerical mismatch.

| Compiled kernel | Registers/thread | Shared B/CTA | Compiler spills |
|---|---:|---:|---:|
| Unchanged full control | 255 | 49,664 | 0 |
| Research default mirror | 254 | 49,664 | 0 |
| Research far VB .cg | 254 | 49,664 | 0 |

`resource_gate.admitted=true`: the candidate did not exceed
the reference's registers/SMEM and did not spill. The report's
`no_additional_registers_or_shared_memory=false` field is
misleadingly named: the actual implementation sets it by
**strict equality** of both register counts and SMEM.
It is false because 254 is **one fewer** register than 255,
NOT because the candidate increased resource usage.
This is metadata semantics, not a resource-gate failure;
do not retroactively rewrite the user JSON.

### Independent Graph GPU event timings

Four independent captures in exact A/B/B/A order, each
with 5 warmup replays, 7 samples of 16 Graph replays.
All times below are reported ms per one full Tile only.
Merge and preparation are excluded.

| Capture | Arm | Median ms | Median microseconds |
|---|---|---:|---:|
| 1 | Control | **0.507575989** | 507.576 |
| 2 | far_vb_cg | 0.374404013 | 374.404 |
| 3 | far_vb_cg | 0.374381989 | 374.382 |
| 4 | Control | **0.383578002** | 383.578 |

The two control captures differ by **32.3266677%**
using the pre-registered `max/min - 1` spread;
the allowed maximum was 3%. Far VB .cg capture spread
was only **0.00588274%**. Their individual samples also
show excursions: control-1 ranges from 0.459758 to
0.559120 ms; control-4 has several 0.427–0.433 ms
samples despite a 0.383578 ms median. The .cg captures
also have isolated ~0.425–0.431 ms samples. This is
observed variance, not proof of another workload,
thermal throttling, GPU clocks or any specific cause.

The report's raw median-of-capture-medians arithmetic
gives control 0.445576996 ms, candidate
0.374393001 ms, and an apparent **15.97568885%**
time reduction. **DO NOT call this an accepted speedup**:
the control is unstable and artificially high in
that aggregation. The conservative `worst_cross_capture_
time_reduction_pct` is only **2.39168806%**,
below the independent >=5% gate. Relative to the
faster control-4 median, candidate's two-capture
median corresponds to around **2.40%** reduction,
but selectively using that control is only context,
NOT a replacement for an admissible stable comparison.

### Decision and next work

**INCONCLUSIVE_TILE_ONLY.** Exact full numerical
admission PASS, resource admission PASS, timing gate
NOT PASSED. Do not promote the far VB .cg cache hint,
interpret the arithmetic 15.98% as demonstrated, or
extrapolate any Tile timing to all attention windows,
56 heads, H3 denoising or video quality. The Graph
captures measured only this one isolated Tile;
`timed_full_attention=false` and `h3_model_or_video=UNRUN`.

The large and temporally varying control latency means
the benchmark environment warrants inspection before
using ANY future small speed effect for admission.
First investigate reproducibility/thermal-clock/
background GPU interference **as hypotheses only**,
with instrumented measurements if this matters,
instead of rerunning until a favorable pair appears.
Do not retroactively drop capture1 or outliers to
turn this pre-registered INCONCLUSIVE into PASS.

If the cache-policy hypothesis is revisited, require
an independently designed environmental stability
gate and paired baseline/control measurements before
any fresh GPU A/B/B/A. Maintain the same >=5% robust
threshold, raw bit-exact two-bank checks, exact-head
source pinning, single JSON and unchanged production.
The existing validated `mask_g` +3.596% Tile-only
result remains separately documented **NO_GO**;
there is no evidence to add the two effects or claim
a combined improvement.

This commit records the uploaded experiment ONLY;
no new CPU/GPU execution, code fix, production change,
model/video run or fresh speed measurement was done.
