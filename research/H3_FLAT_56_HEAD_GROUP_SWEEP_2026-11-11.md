# H3 Flat: 56-head Tile launch group-size sweep (research, GPU UNRUN)

Date: 2026-11-11. Branch: `experiment/h3-flat-56head-tile-20261111`.
BEST remains `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.
User already measured at 16 tiles/head and 56 logical repeated heads:
- Eager 56 pinned launches 6.44446ms vs one Tile launch 2.48078ms, **2.59776x**.
- CUDA Graph replay 4.16928ms vs one Tile 2.733992ms, **1.52498x**.
- PM/PL/PACC bit-exact, 255regs, 0 spills, 49,664B shared.
- 4 distinct heads repeatedly cycled; **NOT 56 distinct native heads**.
- Full set of prepared states+outputs ~1,956.574MiB, not model-resident.

## Hypothesis

GPU launch cost scales with 56 separate head calls and underfilled
per-head chunk grids. But very large head fusion may hurt cache
locality, live-state size and graph replay variance. Try batch
groups of **1,2,4,7,8,14,28,56 heads per launch** before
optimizing production data movement. At 16 tiles/head, launch
counts per 56 jobs are 56,28,14,8,7,4,2,1 respectively.

`tools/benchmark_h3_flat_tile_56head_group_sweep.py` reuses
native Q/K/V captured rows and the same prepared per-head
leaf stats and selections. For each size it performs exact
partial parity against pinned BEST, GPU resource checks,
and paired A/B/B/A CUDA Event measurements both eagerly and
under CUDA Graph. No accidental comparison against tl.range(1).

This sweep **still prepares all 56 logical heads simultaneously**
for controlled Tile timing. The per-group memory figures are
hypothetical reductions if a future implementation stages
only that group, not measured savings in this script.
Separate full-operator/VRAM-constrained benchmarking is needed
before accepting a new cumulative BEST HEAD.

Windows PowerShell:

    cd "D:\ComfyNodes-flat-split-20261111"
    git fetch origin
    git switch --detach origin/experiment/h3-flat-56head-tile-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile tools\benchmark_h3_flat_tile_56head_group_sweep.py
    & $py -B tools\benchmark_h3_flat_tile_56head_group_sweep.py --heads 56 --repeat-available-heads --tiles 16 --modes fast --groups 1,2,4,7,8,14,28,56 --rounds 5 --batch 4

Output JSON: `D:\ComfyOutput\H3\math_probe\flat_tile_56head_group_sweep_*.json`.
Run as a standalone process without the large H3 checkpoint.
Graph capture failures will be reported under each group.
Any candidate parity failure blocks promotion; the existing BEST
remains unchanged. GPU results UNRUN when authored.

