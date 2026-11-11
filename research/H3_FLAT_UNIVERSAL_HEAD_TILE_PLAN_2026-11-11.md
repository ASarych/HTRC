# Universal Flat Tile head count + VRAM-aware group planner

Date: 2026-11-11. Research branch: experiment/h3-flat-56head-tile-20261111.
BEST mathematical baseline remains e47cc7a310936547ab6c7d97bd07526eb6a135fe.

New h3/htrc_flat_head_batch.py exports a SINGLE prepared Tile launcher.
Head count enters solely through Triton grid (H*tiles,split,1), not as
a compile-time specialization. Tile count/key count/data type remain
compile-time address/layout parameters. Works for any positive H.
Only the validated INT8/BF16 seq/sink0/TILE-ONLY path is admitted.
User-visible default H3 runtime remains unchanged.

The pure planner takes currently FREE GPU bytes, safety reserve,
buffer dimensions and optional max group. It chooses the largest
group predicted to fit, supports uneven last groups, and fails if
even one head does not fit. It estimates Q/MU/K8/VBF16/KS/G8/VB/
LZ/SG/EXCL/IDX/PM/PL/PACC plus conservative multiplier and fixed
temp allowance. Prediction is NOT validated lower VRAM use.

GPU benchmark must stage one group at a time and measure real
torch.cuda.max_memory_allocated, then validate parity vs pinned
BEST and full-group timing. Current script still excludes stats/
selection/topk/merge from CUDA event timing.

CPU-only planner tests: tools/test_h3_flat_head_batch_plan.py.
GPU validation of new launcher and streaming memory: UNRUN.
No Actions, no production changes, no promotion to BEST.

