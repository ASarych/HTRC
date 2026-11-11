# HTRC migration: Flat multihead and INT8 preparation research — 2026-11-11

## Source and exact provenance

Source branch: ASarych/ComfyNodes at eb9a39b5907e8f8126623ed62d2bee43c4b2de90.
Destination parent: ASarych/HTRC experiment/flat-tail-merge-flatq-20261111
at a8fe43e7aab8d9f951af7b4cba8fbd54f7a61e5c.

The three pinned HTRC source files in ComfyNodes/e47cc7a310936547ab6c7d97bd07526eb6a135fe
are Git-blob-identical to HTRC/a8fe43e7aab8d9f951af7b4cba8fbd54f7a61e5c:
- h3/htrc_flat.py: 3bbafe0db8e9031f0b105adc2814cd44a75acc2d
- h3/htrc_flat_triton.py: c8bc44e22aaedaf79891129dc9d8485e6c23921e
- h3/htrc_flat_runner.py: 4dea14eed738c62c1499965d17d041f30b6827f7

The migrated benchmark baseline helper maps the legacy ComfyNodes
BEST SHA e47cc7a to this local HTRC commit for git show. No
ComfyNodes checkout is needed to recover the same control source.
The source Git histories and commit SHAs remain different.
The archived HTRC original is main 15983d3589b49926b8bac9b48f64d1feb3bef521.

## Code migrated

- h3/htrc_flat_triton.py: opt-in batched stats, Tile, per-head exact
  tail Merge and fused BF16 K->INT8/FP32-scale quantization.
- h3/htrc_flat_head_batch.py: any positive head count, current free
  VRAM-based head-group planner, grouped state preparation, Tile/Merge.
- tools/benchmark_h3_flat_exact_baseline.py and
  tools/benchmark_h3_flat_split_pass.py: pinned control loading.
- tools/benchmark_h3_flat_tile_56head_launch.py,
  tools/benchmark_h3_flat_tile_56head_group_sweep.py,
  tools/benchmark_h3_flat_tile_streamed_groups.py,
  tools/benchmark_h3_flat_tile_merge_streamed.py,
  tools/benchmark_h3_flat_full_pipeline_batch.py: evaluation chain.
- tools/test_h3_flat_head_batch_plan.py: CPU-only planner tests.

The existing HTRC h3/htrc_flat.py, h3/htrc_flat_runner.py,
h3/htrc_flat_q.py, tools/sweep_h3_flat_kernels.py, theory, main
and the user-added RT addendum were not overwritten.
No checkpoint or private QKV capture files are included.
Default capture paths in scripts refer to local H3 recordings.

## Existing GPU evidence transferred from source (NOT rerun here)

RTX 5070 Ti, four distinct raw head captures (1,16,27,55)
repeated to 56 logical head jobs; not 56 unique model heads.

- Tile-only 56 jobs one launch vs 56 launches: 2.598x eager,
  1.525x CUDA Graph. For VRAM-streamed 12+12+12+12+8,
  sum of per-group Tile medians 5.958 to 2.599 ms (2.292x).
  PyTorch allocated peak ~487.095MiB for group 12.
- Tile+per-head exact R32 tail Merge output: bit-exact for
  all groups against pinned BEST, four distinct full Flat oracles.
  Sum of group CUDA-event Tile+Merge medians 9.349 to 2.790 ms
  (3.351x); max incremental PyTorch allocated peak 453.85MiB.
- Full CPU QKV->final attention batched pipeline: bit-exact
  final outputs but sum of group synchronized host wall medians
  510.640 ms pinned vs 585.737 ms batched, 14.71 percent
  slower. Peak group H12 allocated 453.313 -> 785.522MiB.
  DO NOT promote this preparation implementation into BEST.
- New fused Triton K8 + CPU BF16 staging is GPU UNRUN: must
  compile, check every native-head K8 byte, FP32 scale,
  final attention output and wall/VRAM before considering it.
- Earlier compact dq8/sdq query-cache experiment remains a
  separate GPU UNRUN research item.

None prove GPU compatibility in the new HTRC checkout,
full query 40759 performance, model residency, native 56
distinct head capture or whole denoising/video speed.

## Local test on RTX 5070 Ti

    cd D:\HTRC
    git fetch origin
    git switch --detach origin/experiment/flat-head-batching-quant-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m unittest discover -s tools -p test_h3_flat_head_batch_plan.py
    & $py -m py_compile h3\htrc_flat_triton.py h3\htrc_flat_head_batch.py tools\benchmark_h3_flat_full_pipeline_batch.py
    & $py -B tools\benchmark_h3_flat_full_pipeline_batch.py --heads 56 --repeat-available-heads --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 2 --warmups 1 --quant-mode triton_cpu

Report under D:\ComfyOutput\H3\math_probe\, filename
flat_full_prep_quant_ab_*.json.
GPU run UNRUN since migration. Production is unchanged.
