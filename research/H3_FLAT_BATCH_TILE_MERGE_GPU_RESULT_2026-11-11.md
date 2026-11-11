# H3 Flat: first final-output GPU evidence for arbitrary-head Tile + Merge

Date 2026-11-11. User JSON: `flat_tile_merge_streamed_20261111-193808.json`.
RTX 5070 Ti 16GB. Tested experimental checkout
`cd9db42c6e576b143e6df24b42526ba3f393023a`.
BEST pinned cumulative mathematical control:
`e47cc7a310936547ab6c7d97bd07526eb6a135fe`.

## Workload / evidence scope

One saved H3 attention capture, native distinct raw head IDs
1, 16, 27, 55 (4 different heads) repeated 14 times to
56 **logical head jobs**. This is explicitly a repeated proxy;
not a same-block record of 56 distinct native model heads.
Fast; 16 tiles x 64 = 1024 Q rows per head, K=40752
full leaf rows + 7 exact tail K/V rows; 4 split-CTAs per Tile.
Effective GPU grouping budget=1024 MiB, reserve=256MiB,
safety factor1.5, chosen group counts 12+12+12+12+8.

## Paired eager CUDA events: Tile+Merge only

| Head job span | H | BEST separate Tile+R32 Merge (ms) | One grouped Tile+Merge (ms) | Factor |
|---|---:|---:|---:|---:|
| 0–12 | 12 | 1.958704 | 0.594336 | 3.2956× |
| 12–24 | 12 | 1.968624 | 0.596840 | 3.2984× |
| 24–36 | 12 | 1.964536 | 0.597328 | 3.2889× |
| 36–48 | 12 | 2.171024 | 0.597136 | 3.6357× |
| 48–56 | 8 | 1.285680 | 0.404312 | 3.1799× |

Sum of the five paired **group median GPU Tile+Merge** times
is 9.348568 vs 2.789952ms = **3.35080×**; this is not
measured 56-head end-to-end attention throughput.
All 5 groups final output exactly matched pinned individual
Tile+Merge (relative L2=0; max abs=0, all finite).
Independent `flat_attention` reference checks for raw native
heads 1/16/27/55 also exactly match (relative L2=0).

## Resources and memory

Compiled grouped Tile: 255 regs, 0 spills, 49664B shared.
Compiled grouped tail Merge: 199 regs, 0 spills, 16384B shared.
Actual max incremental PyTorch allocation observed per group:
453.850MiB; H12 batch group. Last H8 used 330.709MiB.
Both returned final output and 7-row tails included.
No reported CUDA OOM, graph testing disabled (cuda_graph=null).
Not tested simultaneously with large H3 checkpoint.

## Preparation is the next bottleneck

Per-group synchronized prep wall time (ms):
992.4925 [cold JIT], 105.3750, 104.7458, 107.0530, 70.6194.
Do not interpret cold JIT as steady-state GPU kernel time.
Stats, selection, torch.topk, input-copy and quantization
are outside the GPU Tile+Merge event interval.

**Decision:** research staged Tile+Merge output accuracy PASS.
Keep pinned BEST unchanged until a whole-preparation+attention
end-to-end GPU/host A/B, bounded VRAM, and actual distinct native
head coverage are validated. Next high-value implementation:
flatten leaf_stats and rowmax_select over heads, batch topk
and compare full Q/K/V -> final outputs against pinned BEST.

