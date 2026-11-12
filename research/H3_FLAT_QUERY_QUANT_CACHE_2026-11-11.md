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


## RTX 5070 Ti GPU A/B completed — 2026-11-12: PASS_NUMERIC / MARGINAL PERFORMANCE, no integration

The owner executed the ComfyNodes original experimental HEAD
`8b3a5d4e7cd96e27f2244546aa6d3b6782c4fa18` with saved
native H3 Q40759 head IDs 1 and 16, both `fast` and `balanced`,
6 paired rounds / batch4. Exact report archived here as
[one normalized JSON](evidence/flat_query_cache_ab_20261112.json).
The historical `UNRUN` notation above describes the pre-GPU state.

**All 4 cases pass**: `pinned_R32`, `new_none`, `cache_dq`, and
`cache_both` each have `max_abs_vs_pinned=0`, `rel_l2_vs_pinned=0`,
finite outputs, and no compilation/parity failures. This does NOT
establish equality with native dense attention: rel-L2 against the old
exact original is around 4-6e-7.

| Head/mode | new_none speedup | cache_dq speedup | cache_both speedup |
|---|---:|---:|---:|
| 1 / fast | 0.99660x | **1.01036x** | 1.00760x |
| 1 / balanced | 0.99911x | **1.00895x** | 0.99838x |
| 16 / fast | 0.99872x | **1.00698x** | 1.01015x |
| 16 / balanced | 0.99993x | **1.00672x** | 0.99832x |

The `dq` variant saves about **0.67%-1.04%** of full-head
synthetic/captured-input attention time. `both` is inconsistent,
slower on two cases. Select kernel grows from 80 registers
(`pinned_R32`/`new_none`) to 87 (`dq`) or 127 (`both`).
Tile stays at 255 registers. All tested kernels have **zero spills**.
Cache staging is ~2.06 MiB (`dq`) / 4.13 MiB (`both`) per
256-tile chunk. No 56-distinct-head model-resident A/B, denoising,
video quality or whole Direct-Flow CUDA Graph timing was performed.

**Decision:** mark `PASS_NUMERIC / PERF_TOO_SMALL_TO_INTEGRATE`.
Keep this opt-in experiment as a numerical control; do **not**
merge into mutable Direct-Flow or production on this evidence.
No repeat GPU test needed for the same configuration.
Prefer experiments cutting the dominant stats / selection /
dispatch cost before revisiting this ~1% micro-optimization.
