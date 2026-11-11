# H3 Flat streamed Tile groups: RTX 5070 Ti GPU result

Date: 2026-11-11. Report from user:
`flat_tile_streamed_groups_20261111-192833.json`.
Candidate tested `602f3be601f7a1e09ae4bef3a886a23633bd0076`;
pinned cumulative BEST `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.

## Exact workload and planner

NVIDIA GeForce RTX 5070 Ti, one saved H3 block0/call1 capture.
Only **four distinct** raw head indices 1/16/27/55; repeated
14 times to **56 logical head jobs** (not 56 unique H3 heads).
`fast`, 16 tiles/head (=1024 Q), 40752 full key rows/head.
`free_mib_before=15039`, artificial effective head-group budget
1024MiB, reserve 256MiB, fixed-temp 128MiB, safety 1.5.
Planner chose 12 heads/group and five contiguous groups
12+12+12+12+8. No CUDA OOM retries or other failures.

## GPU result, per group, Tile-only paired medians

| Job indices | H | Pinned separate Tile (ms) | One grouped Tile (ms) | Speedup | Observed incremental PyTorch alloc peak (MiB) |
|---|---:|---:|---:|---:|---:|
| 0:12 | 12 | 1.152872 | 0.553896 | 2.081x | 487.095 |
| 12:24 | 12 | 1.351168 | 0.557808 | 2.422x | 487.095 |
| 24:36 | 12 | 1.368184 | 0.554120 | 2.469x | 487.095 |
| 36:48 | 12 | 1.359864 | 0.554368 | 2.453x | 487.095 |
| 48:56 | 8 | 0.725888 | 0.378728 | 1.917x | 329.197 |

Sum of the five per-group GPU event medians:
`pinned=5.957976ms`, `grouped=2.598920ms`, **ratio=2.29248x**.
This is NOT measured end-to-end 56-head attention latency:
stats/selection/topK preparation, tail Merge and data movement
are not included in the event timer. It is not a CUDA Graph A/B.

Parity: **5/5 group PASS**, `PM/PL/PACC` exactly equal
(rel-L2=0.0, max-abs=0.0). Compiled resources unchanged:
255 regs, 0 spills, shared 49664B, 4 warps.

**Real measured peak** PyTorch allocated for largest streamed
group=487.095MiB (of which permanent prepared state=370.516MiB).
Last H8 group peak=329.197MiB (state=247.011MiB).
Before every group PyTorch allocated=0MiB; all 56 jobs complete.
Previous full H56 prebuilt Tile test reported ~1956.574MiB for
all states and both partial workspaces at once, so staging
drastically reduces the research benchmark's working set.
These peaks are PyTorch allocation counters, not total device
reserved memory/other processes, and no 14GB model co-residency.

## Preparation timings are dominant and **not** GPU Tile times

Per-group prep synchronized wall ms: 1007.374 (first,
includes initial GPU compilation/warmup), 111.154, 104.579,
108.449, 74.185. Their simple total is ~1405.742ms,
but first-group JIT warmup makes this unsuitable as a steady
production estimate. Later-group preparation wall times remain
large compared with ~0.55ms Tile events. Optimization must
now cover stats/select/topK/merge/materialization, not just Tile.

## Decision

**Accept** this as a native GPU validation of the *streamed
12-head Tile grouping strategy on repeated-head proxy data*.
**Do not promote to BEST yet**: output is partial, no final
tail Merge, no all-stage or real-model video measurement and
no native 56-unique-head capture. Cumulative pinned BEST remains
`e47cc7a310936547ab6c7d97bd07526eb6a135fe`.

Next: implement head-aware batch Merge with each head's own exact
tail K/V rows (7 here), compare final output against pinned BEST;
then build bounded-VRAM whole-operator 56-head A/B.

