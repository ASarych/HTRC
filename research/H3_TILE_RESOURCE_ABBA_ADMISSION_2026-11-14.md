# HTRC Tile32 — isolated register / shared-memory A/B/B/A admission

Date: 2026-11-14. Research branch:
`experiment/flat-memory-to-speed-balanced-20261113`, based on
`d911060d6852c93d2b891f57fa2e317eadaff777`.

## Why now

The user's RTX 5070 Ti Nsight report
`flat_materialized_resource_ncu.json` was verified on
`01996061cec72ba252a3634892c9c642ab90c78c`: 12 CUDA Graph
kernel nodes and strict eager parity PASS. For balanced/512,
15,275 tokens and four heads, Tile under NCU replay had
422.96 us median, achieved occupancy 15.57%,
255 registers/thread, 50,688 shared bytes/CTA and a
register/SMEM occupancy limit of two resident CTAs.
DRAM peak percentage 21.73% and SM peak 64.54% do not prove
that HBM traffic is the main bottleneck.

To allow 3 instead of 2 128-thread Tile blocks on this SM120
(65536 registers/SM and 102400 shared bytes/SM), both resource
ceilings must be met *simultaneously*: <=170 registers/thread
and <=34133 shared bytes/CTA before allocation granularity.
Even 3 CTAs do not prove greater throughput.

## Independent research variants

The existing batch launcher has a new **optional** keyword
`research_resource_variant`. Omission produces the exact
old Tile kernel launch arguments and default compilation;
all original Graph/eager paths omit the keyword.

Two mutually exclusive candidates are allowed, with a hard
source-level shape gate pinned to four heads, 32 tiles,
64 rows/tile, 15,264 aligned keys, 512 exact leaves,
4 splits, BN=64, 4 warps, and original `num_stages=2`:

- `late_q`: only sets the existing `_tile_kernel`
  `LATE_NEAR_Q=1` compile-time option, which moves preparation
  of near-field Q operand to after the far-field loop. Still
  `num_stages=2` and `RB=1`.
- `stage1`: only changes `num_stages` from 2 to 1. The original
  `LATE_NEAR_Q=0` and `RB=1` remain unchanged.

No candidate modifies Stats, Rowmax, top-k, near-field leaf
selection, quality budget, BF16/INT8 representation, or the
near/far accumulation formula. This is **not** a fused Vbar,
streamed VB, or TMA/multicast candidate; those remain NO_GO.
The `late_q+stage1` combination and RB2 have NOT been admitted
until the separate one-knob cases are understood.

## GPU protocol (only after CPU gate)

`tools/benchmark_h3_tile_resource_abba.py` uses existing
`flow.graph_attention_bthd` with two independently generated
BF16 Q/K/V banks at 15,275 tokens, four heads and Tile32.
The first real synthetic call must pass mandatory complete
output eager/Graph parity. Bank B is checked again against
fresh independent eager Flat. Bank A and B must differ.

For one original full 32-tile window at each QKV bank,
call the same existing `batch.launch_prepared_tile_batch`
with control and selected candidate against identical
prepared leaf stats and K/V state. Compare **PM, PL and
PACC bit-for-bit** before any timing. Record compiler
`n_regs`, `n_spills`, and allocated shared bytes per CTA.
These values come from the returned Triton compiled-kernel
metadata, not fabricated from an analytic model.

Then, on bank B, perform four independent **Tile-only CUDA
Graph captures** in A/B/B/A order (control, candidate,
candidate, control). Each block has five warmup replays,
then seven warmed CUDA-event samples by default, with a
bit-exact PM/PL/PACC check before capture, after warmups,
and after timing. A candidate speed signal requires >=5%
median-of-block-medians improvement AND both independent
candidate capture-block medians strictly below both control
block medians.

**Important:** This does NOT time complete attention and
must not be compared directly to the earlier ~4.285-ms
complete-call control. It does not establish H3/video
speedup, VRAM residency, or production suitability. Even
if kernel-only ABBA wins, full-attention independent-capture
ABBA and model-resident memory acceptance are still needed.
New compiler variants can change arithmetic association;
any PM/PL/PACC mismatch is a strict FAIL, not a tolerance.

## Execution, fail-closed contract

`tools/run_h3_tile_resource_abba_safe.ps1` builds a disposable
detached exact-HEAD checkout, checks the user has closed
ComfyUI and Blender, runs CPU tests, and calls the one-JSON
GPU script. No GPU profiling is launched if CPU tests fail.
No H3 checkpoint, paid inference, video output or production
CUDA kernel changes occur.

Single output:
`D:\ComfyOutput\H3\math_probe\flat_tile_resource_abba.json`

The user may choose `-Candidate late_q` or `-Candidate stage1`
for an independent result; the fixed JSON path is intentionally
reused to avoid multiple duplicate files. Old successful
Nsight DRAM/resource reports remain untouched.

**At commit:** local CPU helper tests PASS; exact-repo source
gate and all GPU tests UNRUN until the user's Windows run.
Do not report speedup before the resulting JSON is inspected.


## 2026-11-14 — completed local late_q A/B/B/A: NO_GO

User-uploaded flat_tile_resource_abba.json produced on RTX 5070 Ti.
Source HEAD 8e491e2af6f46f2d849270c593f011274198470c.
SHA-256 of the uploaded JSON bytes:
a4b7d5d86a605660901d37304c07f6a058b1efe3fee1efc0f726256817812b98
(4,224 bytes). Raw JSON is not committed to GitHub; this is
an auditable summary of the user-provided artifact.

Acceptance gates from the actual JSON:
- status=PASS_NUMERIC_TILE_ONLY, completed=true.
- candidate=late_q, source_head exact; 15,275 tokens,
  four independent BF16 heads, 32-tile windows, 512 near leaves.
- Both independent Q/K/V banks: PM/PL/PACC **bit-exact** to
  default materialized control. Bank A/B output differs.
- First-call complete Graph-vs-eager gate PASS and additional
  bank-B eager parity PASS.
- Four separately captured Tile-only CUDA Graph blocks
  in control, late_q, late_q, control order, seven warmed
  timings per block and five extra warmups.

Measured (compiled Triton kernel metadata; no Nsight in this run):

| Field | Control | late_q |
|---|---:|---:|
| Registers/thread | 255 | 255 |
| Shared bytes/CTA | 49,664 | 41,472 |
| Compiler spills | 0 | 0 |
| Optimistic CTA residency per SM | 2 | 2 |
| Per-variant block median-of-medians, ms | 0.391184002 | 0.395295992 |

Independent block medians, ms:
- Block 1 control: 0.391328007
- Block 2 late_q: 0.395231992
- Block 3 late_q: 0.395359993
- Block 4 control: 0.391039997

Shared decreased **8,192 bytes/CTA (16.50%)**, but
register count remains **255/thread**. Both register and
shared-memory limits independently remain at two CTAs/SM.
No occupancy increase was measured or admitted.

control_over_candidate_ratio=0.9895976928: the
candidate took ~1.051% longer in the Tile-only test,
and both independent candidate captures were slower
than either control capture. Neither the at-least-5%
speed criterion nor its both-captures consistency gate
passed. **Final late_q = NO_GO_TILE_ONLY**.
This is a negative performance result, NOT a failed
numerical test and NOT a change in output quality.

Keep existing LATE_NEAR_Q=0 default and do not
re-run late_q or promote it into production. The
standalone stage1 opt-in is already implemented
but **UNRUN**; evaluate it independently via the
same exact-HEAD safe runner with -Candidate stage1.
Do not blend late_q+stage1, alter exact near leaves,
or infer full-model speed from this single kernel.
