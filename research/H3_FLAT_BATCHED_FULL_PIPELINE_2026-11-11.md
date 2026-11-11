# First all-stage grouped Q/K/V -> final Float32 Flat attention A/B

Date 2026-11-11, research branch experiment/h3-flat-56head-tile-20261111.
Source mathematical BEST e47cc7a310936547ab6c7d97bd07526eb6a135fe.
Candidate combines previously validated arbitrary-head Tile+R32
Merge with NEW head-addressed leaf_stats and flattened rowmax/topk.

New stats kernel constexprs BATCH_TILES_PER_HEAD, BATCH_KEY_ROWS,
BATCH_TILE_BLOCKS_PER_HEAD default to 0, so legacy single-head
kernel remains compiled as before. Padding mask prevents cross-head
writes for incomplete 16-tile stats blocks. One grid handles H
head groups, not separate compiled kernels per head count.
Selection already accepted arbitrary flat Q tiles: one rowmax
launch on H*tiles and one batched torch.topk/scatter.
Combined with previously validated Tile and Merge, there are
four main grouped computational stages (stats, select, topk,
Tile, Merge; topk remains PyTorch).

tools/benchmark_h3_flat_full_pipeline_batch.py replays the same
captured CPU Q/K/V head group under two paths:
1) pinned original per-head full state preparation and per-head
   Tile plus R32 exact-tail Merge;
2) candidate batched input gather, per-head K8 quantization
   vectorized, one grouped stats, one rowmax, batched topk,
   one Tile, one R32 Merge.
Both return final normalized float32 attention outputs. Checks
rel-L2 <=1e-4 against pinned control per group and independent
full original flat_attention for unique real captured heads.

Primary measure is synchronized wall time, INCLUDING CPU data
gather/H2D, state allocation, quantization, stats/select/topk,
and Tile+Merge. This addresses the earlier 100+ms prep wall
bottleneck. It is NOT a CUDA-Graph or actual H3 denoiser
benchmark. Per-call allocator life and first JIT warmup are
explicitly differentiated.

Default grouped budget1024MiB, reserve256MiB, expected H12 groups;
OOM halves group size. Report measures actual incremental PyTorch
allocated peaks. The default file still has only 4 distinct
raw heads, optionally repeated 14 times for 56 logical jobs.

Suggested Windows test:

    cd "D:\ComfyNodes-flat-split-20261111"
    git fetch origin
    git switch --detach origin/experiment/h3-flat-56head-tile-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat_triton.py h3\htrc_flat_head_batch.py tools\benchmark_h3_flat_full_pipeline_batch.py
    & $py -B tools\benchmark_h3_flat_full_pipeline_batch.py --heads 56 --repeat-available-heads --tiles 16 --mode fast --budget-mib 1024 --rounds 2 --warmups 1

Output D:\ComfyOutput\H3\math_probe\flat_full_prep_batch_ab_*.json.
GPU run UNRUN. No product behavior change or BEST promotion.

