# HTRC Tile32: independent leaf-stats CTA batch-size experiment

Date: 2026-11-14. Repository: `ASarych/HTRC`,
`experiment/flat-memory-to-speed-balanced-20261113`.
This is an **isolated stats-kernel research slice**, not a
ComfyNodes or production promotion.

## Measured parent baseline: do not include VAE

The user's six-step output-producing H3 Flat Graph A/B on
RTX 5070 Ti 16 GiB established:

| Full attention, 294 calls | Tile16 | Tile32 |
|---|---:|---:|
| GPU event sum | 130.48 s | **113.28 s** |
| Mean of five warmed 49-block GPU cycles | 20.33 s | **18.41 s** |
| CUDAGraph replays per attention at 40,759 tokens | 280 | **140** |
| Peak PyTorch CUDA reserved | 14,368 MiB | 14,624 MiB |

Tile32 gained 13.2% **in full attention GPU-event sum**.
These events include the original quantized QKV, INT8
compact pack, HTRC Graph and native out_proj. They do
**not** establish acceleration of one particular Triton
kernel. Blender activity, QKV offload and other concurrent
GPU loads can affect run-to-run comparisons. VAE and video
saving are **not used** for this kernel decision.

The stable default remains Tile32 with the original
`_stats_kernel` geometry: `BT=16, BL=4`. Every
window's 32 centres are handled by two centre-block CTAs
per head. The candidate `BT=32` uses one centre-block
CTA per head. For 40,759 keys, 2,547 full leaves and
`BL=4`, this reduces total `_stats_kernel` CTA
instances per 56-head attention from 1,426,880 to
713,440 across 20 windows. Each CTA may do more work,
use more registers or spill: **fewer CTAs are not
necessarily faster**.

## Implementation

`h3/htrc_flat_head_batch.py` accepts research
`APL_HTRC_TILE32_STATS_BT=32` ONLY with a capture-safe
native/direct BF16 Q, materialized VB, existing KV cache,
preallocated workspace, 8-head group and `tile_count<=32`.
No exact-leaf count, `G/LZ/SG/VB`, selection,
`torch.topk`, K8, near-field, far-field, Tile or Merge
math changes. `BT=16` is the **unchanged default**.

`h3/htrc_flat_directflow_live.py` rejects BT32 unless
the call uses `tiles=32`, `compare_only=True`,
the explicit genuine-H3-style 8-head Q/K/V producer,
and safe MU/TopK capture. The selected BT is included
in the captured CUDAGraph session key to prevent
replaying a BT16 graph in BT32 tests or vice versa.

### Exact synthetic CUDA A/B: one output JSON

`tools/benchmark_h3_tile32_stats_bt_ab_gpu.py`:
- Exact shape: `40,759` tokens, 8 unique heads,
  BF16 Q/K/V (two independently generated banks),
  balanced/512 exact leaves, Tile32.
- Identical original CUDA Graph Q/K/V producers for both
  modes, no changed QKV weights or attention shortcuts.
- Separate BT16 and BT32 Graph captures; both banks
  must produce bitwise identical **full attention outputs**
  between modes, and each mode must replay bank B
  deterministically. A/B outputs cannot accidentally
  remain unchanged when the source bank changes.
- Warmed whole-attention GPU events (three rounds,
  first capture excluded) compare real CUDA Graph
  runs, with fixed source and the same 8 heads.
- For one representative full query window per mode,
  run the **same HTRC kernels in eager dispatch** and
  collect separate GPU event intervals for
  `leaf_stats`, `rowmax_topk_exclusion`,
  `tile_kernel` and `merge_kernel`, plus Q/K
  staging intervals. This is **kernel-stage attribution
  outside Graph**, NOT stage timing within the Graph.
  The eager output window must be bit-exact to the
  captured Graph output on bank B.
- The first-call DirectFlow reports are temporary and
  deleted; **one** persistent artifact remains:
  `D:\ComfyOutput\H3\math_probe\flat_tile32_stats_bt_ab.json`.
  On failure it is marked `FAIL_CLOSED` with traceback,
  not `PASS_NUMERIC`.
- No H3 checkpoint, VAE, Blender, ComfyUI model hooks,
  or full video are loaded.

The small pure-Python AST source/CTA-count gate in
`tools/check_h3_tile32_stats_bt_cpu.py` runs before
CUDA in the isolated script
`tools/run_h3_tile32_stats_bt_ab_safe.ps1`.
It uses an exact pinned checkout cloned from the
user's HTRC source, checks the exact SHA and never
moves the installed ComfyNodes directory. It rejects
an open ComfyUI port 8188 **and active Blender**,
so external GPU/VRAM use does not trivially confound
the paired kernel benchmark. PyTorch's per-arm peak
allocator delta is recorded; absolute memory peaks
still carry allocator-cache and previous-output
retention effects across arms. Do not treat those
as independent whole-model VRAM measurements.

### Decision after GPU result

1. If GPU source/capture, both-bank numerical parity,
   or stage-output parity fails: **NO_GO**. Keep BT16.
2. If BT32 is slower: **NO_GO** regardless of smaller
   CTA count. No H3 video.
3. If BT32 is reproducibly faster by a useful margin
   on 40,759 tokens and does not cause a VRAM regression:
   consider an **independent 49-block single denoise
   forward** opt-in as the next acceptance gate.
   Only after that should a six-step real H3 test be
   discussed. Do not change `main` or stock fallback.

Status on user's RTX 5070 Ti:
**NEW CPU/GPU TESTS UNRUN**; no BT32 speedup or
numerical parity claim.

## 2026-11-14 17:58 — first Windows preflight failure and regression fix

The user executed the isolated exact-HEAD CUDA A/B after its
CPU source/CTA gate succeeded. It **failed before any BT16
CUDA kernel or Graph capture**:

```text
phase=BT16_capture
RuntimeError: H3 Direct-Flow is inference-only
graph_attention_bthd -> torch.is_grad_enabled()
```

Root cause was **our benchmark harness**, not the Tile32
statistics kernel. `tools/benchmark_h3_tile32_stats_bt_ab_gpu.py`
called the strict HTRC output-producing Graph without entering
`torch.inference_mode()`. The DirectFlow guard correctly
rejected grad-enabled execution before any GPU timing was
recorded. Prior static 20/20 checks failed to cover this
runtime precondition.

Fixed by decorating the complete GPU benchmark `emit()` entry
with `@torch.inference_mode()`, rather than changing HTRC
Graph policy or disabling grad globally for unrelated code.
`tools/check_h3_tile32_stats_bt_cpu.py` now parses the
actual benchmark's Python AST and enforces the exact
`torch.inference_mode()` decorator before GPU test startup.

No Flat math, Tile16/Tile32 defaults, capture cache, H3 model
weights, or ComfyNodes path changed. The corrected CPU/GPU
tests are **UNRUN** in this environment. No GPU speed result
is inferred from this failed harness launch.

## 2026-11-14 — first completed 8-head CUDA A/B: BT32 not faster

User-supplied RTX 5070 Ti report `flat_tile32_stats_bt_ab.json`
from exact HEAD `dc8c08d437e1849b83638cfb0dbc69bca7091284`:
`PASS_NUMERIC`, 40,759 tokens, balanced/512, two independent
BF16 Q/K/V input banks; both modes bit-exact across every output row.
No H3 checkpoint, denoise, VAE, video, or real production promotion.

- BT16 warmed full-attention GPU events: 37.2983, 37.2873,
  37.3243 ms; median **37.2983 ms**.
- BT32 warmed GPU events: 58.4037, 53.6107, 37.4611 ms;
  median **53.6107 ms**, ratio BT16/BT32 = 0.6957.
- One eager full-window `leaf_stats` GPU interval:
  **0.618496 ms** (BT16) vs **0.619552 ms** (BT32).
  Tile kernel intervals 0.949792 vs 0.947744 ms; these
  are not per-Graph timings and do not explain the two slow
  BT32 attention samples.
- Arm-baseline allocated-peak deltas: 1123.20 / 1123.13 MiB.
  Cross-arm PyTorch caches, retained bank outputs, and Graph
  private pools prevent interpreting absolute peaks as an
  independent total VRAM comparison.

Decision: keep **STATS_BT=16** and Tile32 default. Do not run
an H3 model/video test of BT32 based on these data. The
BT32 37.46 ms final sample is too close to BT16 to rule
out capture/order/warmup-related variability.

### Opt-in independent-capture ABBA follow-up

The CUDA harness now supports `--paired-abba --warmup-replays 5
--rounds 7`. It performs BT16, BT32, BT32, BT16 in one
process with **fresh CUDAGraph captures each block**, two
independent input banks and bit-exact full outputs both
across BT variants and repeat captures. Each block gets five
additional untimed full-attention replays AFTER capture,
then seven warmed timed replays. No timing includes capture.
ABBA skips eager kernel-stage attribution entirely, so the
four capture blocks do not inherit extra diagnostic allocations.
The ordinary two-mode A/B still reports separate eager stages.

The single JSON includes capture order, raw GPU-event
samples, block medians, and a speed decision. An
**exploratory** BT32 speed signal requires >=5% gain in
the median of block medians AND both BT32 block medians
strictly below both BT16 medians. This is a threshold for
further checking only; NOT permission to promote to H3.
Allocator baselines are still not isolated by process.

On the user's Windows checkout (from exact pushed HEAD):

```powershell
& "D:\HTRC-graph-group-ab-20261113\tools\run_h3_tile32_stats_bt_ab_safe.ps1" -ExactHead "<exact pushed HEAD>" -PairedABBA -Rounds 7
```

The safe runner checks source, ComfyUI/Blender inactivity,
Python AST CPU admission, and stores ONE persistent report
at `D:\ComfyOutput\H3\math_probe\flat_tile32_stats_bt_abba.json`.
The previous A/B JSON is left intact. The original non-ABBA
runner behavior remains unchanged.

New Python CPU/source gate, actual CUDA ABBA, and real H3
are **UNRUN** as of this commit. No kernel/default/runtime
math or installed ComfyNodes source was modified.

## 2026-11-14 — ABBA methodology correction (pre-GPU run)

Review of the new ABBA harness found that the old eager
per-window stage attribution still ran **between capture
blocks**, after the GPU timings. Though not directly timed,
it could retain allocator buffers and alter subsequent
capture/Graph-cache conditions. Prior HTRC group-policy
research measured precisely this kind of contamination.

ABBA now **skips the eager attribution entirely**, returning
`eager_stage_attribution_status=SKIPPED_ABBA_ISOLATION`,
and JSON `null` for stage intervals and eager/Graph
window parity. Full attention parity for two banks,
cross-variant bit-exactness and repeated-capture parity
remain mandatory. The previous non-ABBA benchmark still
runs its eager stage attribution unchanged.

An AST CPU source admission now rejects unguarded
`prepare_batched_head_states`, `launch_prepared_tile_batch`,
and `launch_prepared_merge_batch` calls within the
benchmark `emit` body. This is source-level verification,
not a CUDA result. New CPU script and CUDA ABBA remain
**UNRUN**; the existing BT16 default stands.
