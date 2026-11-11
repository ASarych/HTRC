# H3 Flat: first successful 56-logical-head single Tile launch A/B

Date: 2026-11-11. Hardware: NVIDIA RTX 5070 Ti 16GB.
Source local GPU JSON: `flat_tile_56head_launch_ab_20261111-190205.json`.
Experimental test commit: `6d113089693e42aa7bcd14d181c5c946d63aa247`.
BEST pinned control commit: `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.

## Scope

Saved same-block `attention_block_00_call_01.pt` contains only raw heads
1, 16, 27, 55. The test explicitly used `--repeat-available-heads` to
cycle those 4 heads 14 times into **56 logical work items**.
`is_repeated_capture_proxy=true`. This is NOT a real distinct 56-head
capture or evidence of actual H3 model end-to-end improvement.

Fast mode, 16 tiles × 64 rows = 1,024 Q per logical head;
40,752 K rows per logical head, 256 exact leaves, 4 split CTAs per
tile. 56 independent Tile launches with 64 CTAs each (896*4 =
3,584 CTAs total) vs one flattened 3,584-CTA launch.

## Paired CUDA-event timings

| Measurement | 56 separate pinned Tile launches | One head-flattened Tile launch | Speedup | Time reduction |
|---|---:|---:|---:|---:|
| Eager CUDA Event median | 6.444460 ms | 2.480780 ms | **2.597756x** | **61.50%** |
| CUDA Graph replay median | 4.169280 ms | 2.733992 ms | **1.524979x** | **34.43%** |

Six A/B/B/A rounds with `batch=4` GPU launch repetitions per event.
Individual samples show eager batched outliers up to 4.60ms,
and graph batched samples ~2.50-2.79ms. Eager candidate median
was faster than graph candidate median; do NOT assume the
graph result reflects the performance ceiling. Repeat later.

## Correctness and resources

`partial_max`, `partial_den`, `partial_acc` outputs match the pinned
control bit-for-bit: rel-L2=0 and max_abs=0 for all three.
Both original and single-batch `_tile_kernel` compile at
255 registers/thread, 0 spills, 49,664 B shared, 4 warps.
This result tests Tile with TAIL=0, as required for R32
`tail_in_merge` but does NOT execute Merge.

GPU prebuilt input states occupy 1,729.074 MiB; both sets of
intermediate output buffers another 227.5 MiB. Total explicitly
accounted = **1,956.574 MiB** (~1.91GiB), excluding allocator,
graph private pools, and temporary preparation. This is measured
with the large H3 checkpoint not present in VRAM.

## Interpretation

The test supports fewer GPU kernel launches / larger combined
launch grids for Tile on small per-head query chunks.
It does not establish improvement in per-SM occupancy, which
remains constrained by 255 regs and 49.7KiB shared per CTA.
The old profile's 70.7% no-eligible-warp scheduler cycles is
NOT device idle time and not an estimate of possible speedup.

**Tile microbenchmark PASS, full H3 runtime acceptance PENDING.**
Do not promote experimental head batching to BEST just yet.
Next check should sweep batch group sizes (1,2,4,7,8,14,28,56),
including eager+CUDA Graph and exact partial parity, to measure
the diminishing returns and approximate staging memory.
Then build a bounded-VRAM full attention 56-head A/B that
includes leaf stats, selection, topK, Tile, Merge, transfers,
and if available actual native 56-distinct-head capture.

No changes to production defaults or H3 model inference.

