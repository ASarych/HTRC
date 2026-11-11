# H3 Flat — compact INT8 query quantization reuse from rowmax to Tile

Date: 2026-11-11. GPU target: RTX 5070 Ti 16 GiB.
Branch: `experiment/h3-flat-query-quant-cache-20261111`.
Baseline code: `e47cc7a310936547ab6c7d97bd07526eb6a135fe`
with `tile_tail_merge=True`, R32, original Python `range` near loop.
No inherited tl.range(1) default regression and no full QG cache.

## Motivation

Prior full QG reuse `tile_select_score_cache` kept 80MiB FP16 or
160MiB FP32 per chunk and was 18-33% slower. Also its inherited
tl.range(1) default had a ~3-5% slowdown vs pinned old-range source.
Those are **negative controls**; no speedup may be claimed from them.

With INT8 far-field G and INT8 near K, the rowmax selection
calculates per-query `dq8` and FP32 row scale `sdq`. Each of four
split Tile CTAs independently quantizes the same `dq` again.
Each Tile CTA also independently quantizes exact near `q` into
`q8` and scale `sq`. These are redundant per-query operations.

## New isolated opt-in

`kern.tile_select_query_cache` values:
- `none` (default): exact pinned code path; no new buffers
- `dq`: rowmax split0 writes the already computed dq8/sdq to
  small global buffers; each Tile split reuses them
- `both`: also compute/cache exact q8/sq once in rowmax split0;
  remove its duplicated Tile quantization

All saved quantized values use exactly the original formula
(amax, scale, `(x >= 0 ? x+0.5 : x-0.5)`, int8 cast).
Write is restricted to rowmax program `sp == 0`, and Tile
launch occurs after rowmax/topk on the same stream. The
cached buffers have one writer per query row; no new GPU kernel.

Applicability is deliberately narrow: max selection, stride1,
group1, tile_rb1, fused Tile, `dot=bf16`, `store=int8`,
nonzero near leaves, no `late_near_q`. Unsupported opt-ins raise
an explicit error rather than applying an incorrect cache.

Per 256 tiles, 64 query rows, dim128:
- dq8 uint8-sized int8 + sdq float32: ~2.06 MiB extra
- both (plus q8 + sq): ~4.13 MiB extra

## GPU benchmark (UNRUN)

`tools/benchmark_h3_flat_query_cache.py` compares:
- source original 571a7a3 (numerical reference only)
- pinned clean baseline e47cc7a, tail-in-merge R32
- new code with cache disabled (`new_none`): compiler regression gate
- new code with `dq` and `both`

For native Q40759 head1/16 fast/balanced, each entry captures
compiled Tile/rowmax register count, spills, shared memory;
numeric parity vs pinned and original (rel-L2 <=1e-4); and
paired interleaved batched full-head CUDA-event medians.
If a candidate fails compilation/parity, it is saved to the
JSON and the test ends with a nonzero exit.

Windows PowerShell:

    cd "D:\ComfyNodes-flat-split-20261111"
    git fetch origin
    git switch --detach origin/experiment/h3-flat-query-quant-cache-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat.py h3\htrc_flat_triton.py tools\benchmark_h3_flat_query_cache.py
    & $py -B tools\benchmark_h3_flat_query_cache.py --heads 1,16 --modes fast,balanced --rounds 6 --batch 4

Report: `D:\ComfyOutput\H3\math_probe\flat_query_cache_ab_*.json`.
All GPU compilation/timing is UNRUN in authoring environment.
Production branch and defaults unchanged. If this also loses,
switch attention to removal of select/topk/split memory traffic
rather than expanding the new cache.

