# Universal batch Tile+R32 exact Tail Merge: final-output gate

Date: 2026-11-11. Branch experiment/h3-flat-56head-tile-20261111.
Continues on top of previous confirmed 2.292x Tile-only streamed
12/12/12/12/8-group RTX 5070 Ti result with 487MiB measured peak.
BEST parent source remains e47cc7a validated tail-in-Merge R32.

New opt-in Triton constexprs for existing `_merge_tail_kernel`:
  BATCH_TILES_PER_HEAD (default 0), BATCH_TAIL_ROWS (default 0).
Source default compiles as before; when active, head ID derives
from flattened tile ID and shifts K/V tail pointers to that
head's private 1..16 tail tokens.

`launch_prepared_merge_batch` consumes prepared Tile PM/PL/PACC
and Q, plus tails [H,ntail,D], to emit final normalized fp32
attention outputs in one Merge kernel launch per head GROUP.
It handles no-tail by delegating to the existing Merge kernel.
All H values are permitted, with no H-specific code variants.

New `tools/benchmark_h3_flat_tile_merge_streamed.py` stages one
group at a time under current free VRAM and 1024MiB test budget:
- pinned BEST per-head Tile + per-head R32 Tail Merge
- candidate one batched Tile + one batched Merge per group
- complete final attention output parity, rel-L2 <=1e-4
- independent original whole Flat output crosscheck for each
  distinct raw captured head in the first group
- eager CUDA-event A/B for BOTH Tile and Merge
- optional --cuda-graph comparison (GPU test UNRUN)
- actual peak PyTorch allocated MiB, OOM-halving retry

Important: stats/selection/topk still constructed per head and
are excluded from CUDA-event timings; this is not yet end-to-end
whole Flat or H3 generation. Captured raw heads 1/16/27/55
repeat under explicit --repeat-available-heads proxy flag.

Windows PowerShell:

    cd "D:\ComfyNodes-flat-split-20261111"
    git fetch origin
    git switch --detach origin/experiment/h3-flat-56head-tile-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat_triton.py h3\htrc_flat_head_batch.py tools\benchmark_h3_flat_tile_merge_streamed.py
    & $py -B tools\benchmark_h3_flat_tile_merge_streamed.py --heads 56 --repeat-available-heads --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 3 --batch 2

Output: `D:\ComfyOutput\H3\math_probe\flat_tile_merge_streamed_*.json`.
GPU run UNRUN. No production changes, BEST not promoted.

