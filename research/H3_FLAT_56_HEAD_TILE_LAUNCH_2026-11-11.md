# RTX 5070 Ti H3 Flat: one Tile launch across all 56 heads

Date: 2026-11-11. Branch: `experiment/h3-flat-56head-tile-20261111`.
Clean BEST parent: `e47cc7a310936547ab6c7d97bd07526eb6a135fe`
(validated tail-in-Merge, 32-row, ordinary Python-range near loop).
Original 56-head mode is **untouched**. No production activation.

## Question and limitations

Current `FlatRunner.run(jobs)` iterates heads serially. This A/B
tests whether a **single grid of 56 heads for Tile** beats 56
individual Tile kernel launches on identical saved H3 data.

NCU for the previous Tile reported 16.7% theoretical, ~16.2%
achieved occupancy with 255 regs/thread and ~49.7KiB shared/block.
It also reported ~70.7% scheduler cycles with no eligible warp.
Those figures are *not* GPU idle percentage: one 16k-Q
Tile already issued 1,024 CTAs (more than enough for 70 SMs).
Register/shared residency caps remain under head batching.
Increased grid size may still amortize launches and reduce
underfilled SM waves for small per-head chunks.

## Independent pinned and batched code

Patch adds compile-time `BATCH_TILES_PER_HEAD` and `BATCH_KEY_ROWS`
to `_tile_kernel`. When zero (default), the original program ID,
near K/V/KS addressing and numerical operations remain intact.
When active, program-id tile identifies its head and uses
`key_base=head*BATCH_KEY_ROWS` for the per-head K/V/KS.
Q, G, VB, LZ, SG, selection/exclusion and PM/PL/PACC are
flattened `[heads * tiles, ...]` in head-major order.
Only one Tile kernel launch (grid heads*tiles, split, rb).

Benchmark `tools/benchmark_h3_flat_tile_56head_launch.py`
requires 56 distinct native raw heads for a true native-head test.
The previous default capture `attention_block_00_call_01.pt`
actually contains only **4** raw heads. An explicit
`--repeat-available-heads` flag repeats those real captured heads
to form 56 logical jobs and test launch throughput only. It cannot
prove correctness/performance across 56 distinct native heads.
Uses the exact
same bf16 K/V, INT8 near-key quantization, leaf stats,
per-tile selection+torch.topk and exclusions to build input
states for each head; both paths consume those *same* tensors.
Control loads exact pinned BEST source into a separate module
namespace and launches its Tile once per head. Batched Tile
executes the same far/near math in one grid with corrected
per-head K/V pointer offsets.

Primary parity: all three split partial outputs PM, PL, PACC
from pinned control vs fused Tile, threshold rel-L2 <=1e-4
and finite results. Compiled regs/spills/shared recorded.
Paired A/B/B/A timings include eager CUDA Events and
CUDA Graph replay if graphs capture successfully.

## Test scope and VRAM

DEFAULT: all 56 heads, modes fast, chunks 16 and 64 *tiles per
head*, 64 queries per tile, full 40,752 key rows per head.
At chunk16 each head contributes 64 Tile CTAs (4 splits).
Combining heads changes 56 x 64-CTA GPU launches into one
3,584-CTA launch. At chunk64, 256 CTAs per head vs one
14,336-CTA grid. These are NOT full Q=40759 query lengths:
the probe uses first 1,024 or 4,096 query rows of every head
for practical 16GB GPU staging, with native 40,752 K keys.
Benchmark reports actual allocated GPU input/partial buffer bytes.

The preparation, `leaf_stats`, `select`, `torch.topk`,
`merge_tail` and output handling are **outside timing**.
Therefore even if Tile wins, this is not a complete 56-head
operator in one kernel or proof of end-to-end gain.
One full Q length contains 637 query tiles and would require
multiple 56-head Tile chunk launches at tested chunk size.

Run from separate experimental Windows worktree:

    cd "D:\ComfyNodes-flat-split-20261111"
    git fetch origin
    git switch --detach origin/experiment/h3-flat-56head-tile-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat_triton.py tools\benchmark_h3_flat_tile_56head_launch.py
    # First: 56-work-item batching PROXY from the four captured heads
    & $py -B tools\benchmark_h3_flat_tile_56head_launch.py --heads 56 --repeat-available-heads --modes fast --tiles 16 --rounds 6 --batch 4

    # True 56-head validation: only with a same-block capture of >=56 distinct heads
    # & $py -B tools\benchmark_h3_flat_tile_56head_launch.py --capture "D:\path\to\real_56_head_capture.pt" --heads 56 --modes fast --tiles 16 --rounds 6 --batch 4

Output JSON: `D:\ComfyOutput\H3\math_probe\flat_tile_56head_launch_ab_*.json`.
Strongly recommend running standalone without the 14GiB H3 model
already resident in VRAM. If memory is insufficient, use `--tiles 8`
to reduce scratch before attempting larger tile chunks.
All GPU tests UNRUN; production remains unchanged.

## Next if Tile is promising

Measure end-to-end 56 heads including prepared state, selection,
top-k, Tile, merge and transfers. On the real model, bound VRAM
by chunking across query tiles. If Tile fusion beats BEST on total
multi-head time and fits the 16GB budget, integrate into the new
BEST HEAD and re-run full-head numerical/CUDA Graph gates.

No promise that batching increases occupancy or video throughput.

## 2026-11-11 input-capture correction

The previous example command failed before GPU launch because the pinned
capture has only **four** raw heads. The new CLI refuses to pretend a
four-head file is a native 56-head trace. Explicit
`--repeat-available-heads` repeats them to create 56 logical jobs and
marks JSON `input_head_kind=repeated_captured_head_proxy` and
`is_repeated_capture_proxy=true`. This is a valid **launch-pressure
microbenchmark**, not native 56-distinct-head validation. For the latter,
use `--capture` with one unmodified same-block 56-head Q/K/V capture.

First run `--tiles 16` standalone to limit 16GiB VRAM pressure; move
to `--tiles 64` only after testing the smaller shape. GPU UNRUN.
