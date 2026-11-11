# HTRC: compact query quant cache imported as separate research variant

HTRC branch: experiment/flat-query-quant-cache-20261111
Parent: a8fe43e7aab8d9f951af7b4cba8fbd54f7a61e5c
Source ComfyNodes branch: 8b3a5d4e7cd96e27f2244546aa6d3b6782c4fa18

This work is isolated from the batched-head experiment branch.
It reuses the byte-identical HTRC pinned BEST source on parent a8fe.
The original numerical reference ComfyNodes/571a7a3 was added
as a code snapshot under research/baselines/comfynodes_571a.
All three original H3 source files are preserved unchanged, and
the baseline helper loads them without needing ComfyNodes checkout.

Opt-in KERNEL research: rowmax split0 caches the exact
INT8 query quantization dq8/sdq, and optionally q8/sq.
Tile reuses it across split CTAs; additional cache per
256-tile Q chunk ~2.06MiB or ~4.13MiB.
Selection max/stride1/group1/INT8 dot/fused only.
No production activation, no head batching in this branch.

The source was committed but has **NO GPU PARITY OR SPEED PASS**.
Run --heads 1,16 --modes fast,balanced on the saved native
H3 block00 call01 capture; compare pinned e47 source using
local HTRC a8 blob-equivalent control. Fail closed on
candidate compile/parity regressions.

PowerShell:

    cd D:\HTRC
    git fetch origin
    git switch --detach origin/experiment/flat-query-quant-cache-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat.py h3\htrc_flat_triton.py tools\benchmark_h3_flat_query_cache.py
    & $py -B tools\benchmark_h3_flat_query_cache.py --heads 1,16 --modes fast,balanced --rounds 6 --batch 4

The default trusted native capture remains local under
D:\ComfyOutput\H3\math_probe; use --capture if relocated.
Do NOT mix the query-cache experiment with unverified
batched preparation. When a candidate passes, evaluate its
combined improvement on a measured cumulative BEST.
