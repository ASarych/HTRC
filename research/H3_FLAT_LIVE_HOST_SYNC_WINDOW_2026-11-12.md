# H3 Flat live backend: per-window host stall and query-window size (2026-11-12)

Branch `experiment/flat-live-host-sync-20261112`, based on
`experiment/flat-directflow-real-h3-20261112@c8b22570`.
Raw report: [`flat_live_window_overhead_20261112.json`](evidence/flat_live_window_overhead_20261112.json).
Tool: `tools/benchmark_h3_flat_live_window_overhead.py`.

## Scope

RTX 5070 Ti, synthetic random BF16 Q/K/V, 18,723 tokens, 56 distinct heads, dim 128,
real `flat_attention_bthd` eager path, fixed `fast` math (256 exact leaves, centre
group 1). **No model weights resident, no video, no quality claim.** No Flat kernel,
leaf count, centre group, selection or precision was changed.

## Finding 1: one implicit CUDA synchronization per query window

`prepare_batched_head_states.gather` built its head index with
`torch.tensor(positions, device=cuda)`. That is a pageable host-to-device copy; it
waits for everything already queued on the GPU. With `query_staging="gather"` (the
default of `flat_attention_bthd`) it ran once per window and head group: 114 times per
attention call at 16 tiles. `torch.cuda.set_sync_debug_mode("warn")` reported exactly
those 114 sites, plus one per group in `prepare_gpu_head_kv_cache` and in `_tail`.

Fix in this branch: consecutive native head positions are taken with a slice, so no
index tensor is uploaded. Non-consecutive positions keep the old `index_select` path.

| `gather`, 16 tiles, budget 1024 MiB | whole-call wall |
|---|---:|
| base `c8b22570` (three separate runs) | 168.5 / 179.5 / 179.5 ms |
| this branch (two separate runs) | 116.9 / 118.3 ms |

Output is bitwise identical. The base-commit runs were console measurements made
before this tool existed; only the branch numbers are in the JSON.

The two per-group uploads (`prepare_gpu_head_kv_cache` in `htrc_flat_head_batch.py`,
`_tail` in `htrc_flat_model_shadow.py`) are still there: 6 + 6 per call at this
geometry. They were not changed or measured separately.

## Finding 2: the 16-tile window makes the call host-bound

Each window/group iteration costs roughly 0.9 ms of host time regardless of how much
GPU work it carries. Window state (G/VB/LZ/SG/partials) is small next to the K/V
cache, so a wider window fits in the same budget and needs fewer iterations.

Numbers from the JSON (this branch, median of 5, `wall_ms` = call + final synchronize):

| staging | tiles | budget MiB | groups x windows | wall ms |
|---|---:|---:|---:|---:|
| gather | 16 | 1024 | 6 x 19 = 114 | 118.33 |
| gather | 32 | 1024 | 8 x 10 = 80 | 89.57 |
| gather | 64 | 3072 | 4 x 5 = 20 | 78.28 |
| native_torch_mu + native_direct | 16 | 1024 | 6 x 19 = 114 | 115.70 |
| native_torch_mu + native_direct | 32 | 1024 | 8 x 10 = 80 | 78.52 |
| native_torch_mu + native_direct | 64 | 1024 | 14 x 5 = 70 | 87.09 |
| native_torch_mu + native_direct | 32 | 1536 | 5 x 10 = 50 | 73.18 |
| native_torch_mu + native_direct | 32 | 2048 | 3 x 10 = 30 | 67.02 |
| native_torch_mu + native_direct | 64 | 2048 | 6 x 5 = 30 | 70.17 |

All nine cases are bitwise equal to the first one. `peak_extra_mib` in the JSON
includes one live 256 MiB output tensor from the previous round; compare cases with
each other, not with earlier per-call memory figures.

In earlier console runs on the base commit, native staging at 16 tiles measured
101.3 / 104.8 ms; here 115.7 ms. Treat the 16-tile native figure as 101-116 ms.

Only the 32-tile / 2048 MiB case returns to the host before the GPU finishes
(`host_return_ms` 55.25 vs wall 67.02). Everywhere else the host is the limit.
The lowest walls, 66-70 ms, agree with the mutable-QKV CUDA Graph probe
(14.0 ms per 11-head group, about 71 ms for 56 heads): that is the GPU floor of the
unchanged kernels on this shape.

## What this does and does not show

- Shown: on this synthetic shape, removing the per-window stall and using 32 tiles
  with native staging takes one attention call from 168-180 ms to about 75-79 ms at
  the same 1024 MiB budget, bitwise identical.
- Not shown: any model-resident or video timing. The planner derives group capacity
  from free VRAM, so groups may be smaller with weights resident. The real-run
  figure of 169.08 ms was recorded at Q=0.3 settings and is not a like-for-like
  baseline.
- The earlier `Direct Q / Direct QKV = NO_GO_PERFORMANCE` decision was taken on sums
  of GPU event medians, which do not see host stalls. Whole-call wall with native
  staging is not slower here. That decision should be re-measured on whole-call wall.

## Next gate (UNRUN)

Model-resident A/B of one real attention call and of steady-state denoise steps:
base vs this branch at `APL_HTRC_FLAT_TILES=16` and `32`, gather and native staging,
with peak VRAM. Then compare against the Direct-Flow graph gate on the same call.
