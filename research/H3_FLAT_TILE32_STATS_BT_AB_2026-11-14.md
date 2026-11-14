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
an open ComfyUI port 8188.

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
