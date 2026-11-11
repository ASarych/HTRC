# Streaming Tile head-group VRAM gate — RTX 5070 Ti

Research date: 2026-11-11. Branch: experiment/h3-flat-56head-tile-20261111.
Tool: tools/benchmark_h3_flat_tile_streamed_groups.py.

This benchmark is the first to REALLY stage one dynamically chosen
head group at a time on GPU; unlike the prior group-size sweep it
does not prebuild all 56 logical heads into GPU memory at once.
Group size comes from live free bytes, user reserve, conservative
per-head prepared state estimate, and optional max group cap.
True CUDA OOM halves the group size, frees temporary memory and
retries without treating unrelated errors as memory failures.

Each group comes from the same trusted native saved Q/K/V (or
explicit repeated four-head proxy), computes leaf states and
topk exclusions using the original pinned preparation, then
compares 1 grouped Tile kernel launch to separate per-head
Tile launches from pinned BEST e47cc7a. Numerical PM/PL/PACC
parity, spills, registers, CUDA Event medians, and actual
torch.cuda.max_memory_allocated are collected for every group.

This is still only a Tile-stage result. Preparation CUDA kernels
are executed but timed separately as synchronized wall time;
merge, tail and full H3 generation are not measured. The sum
of group Tile medians is explicitly NOT whole-operator throughput.

Example PowerShell:

    cd "D:\ComfyNodes-flat-split-20261111"
    git fetch origin
    git switch --detach origin/experiment/h3-flat-56head-tile-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m unittest discover -s tools -p test_h3_flat_head_batch_plan.py
    & $py -m py_compile h3\htrc_flat_head_batch.py tools\benchmark_h3_flat_tile_streamed_groups.py
    & $py -B tools\benchmark_h3_flat_tile_streamed_groups.py --heads 56 --repeat-available-heads --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 3 --batch 2

Output JSON: D:\ComfyOutput\H3\math_probe\flat_tile_streamed_groups_*.json.
GPU results UNRUN when authored; no production changes.
Native 56-distinct-head trace not available in stored capture.
CUDA Graph plus full-operator merge and 56-head end-to-end speed
are separate acceptance gates before promoting BEST.

