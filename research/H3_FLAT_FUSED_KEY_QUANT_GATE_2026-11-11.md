# H3 Flat: fused K8 and reduced H2D staging A/B after full-prep regression

Date 2026-11-11. Branch `experiment/h3-flat-56head-tile-20261111`.
Pinned cumulative BEST source: `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.
Last GPU result: `flat_full_prep_batch_ab_20261111-195815.json`.
That whole Q/K/V-to-final-output batched candidate (SHA 353a00f)
was **14.71% slower** than pinned BEST (sum of five group
median HOST wall times 585.737 vs 510.640ms) while remaining
bit-exact for 56 repeated logical jobs and 4 independent full
Flat controls. Peak allocated H12: 785.522MiB batched vs
453.313MiB BEST. Tile+Merge itself won, preparation lost.

## Hypothesis to test — not confirmed

Original batched prep collects H heads' K/V then materializes
whole-group FP32 `kf=kbf.float()` and quotient `kf/KS`.
This can add 100s MiB of transient GPU tensors. It also sends
K/V as full-precision tensors to GPU before BF16 conversion.
With only four native heads repeated to 56, all repetitions
must still use distinct logical K/V buffers for this benchmark.

## New opt-in backend

One Triton `_quant_key_int8_rows` launch per group reads BF16
K directly and computes per-token FP32 scale, INT8 K. No global
FP32 K or quotient buffer. Uses precise `tl.div_rn` and PTX
nearest-even `cvt.rni.s32.f32` intended to mirror PyTorch
round(). Actual parity must be proven on RTX 5070Ti before
timing; the test compares **every** INT8 byte and FP32 scale
against pinned PyTorch quantization for one full native K head.
Any mismatch, GPU compile error, or numerical output error
stops acceptance and saves a JSON diagnostic.

An independent optional CPU BF16 cast before H2D avoids
holding temporary full-head-group original-precision K/V
tensors on GPU. The four candidate settings are:

| Flag | Quantization | BF16 conversion |
|---|---|---|
| torch_gpu | original PyTorch (unchanged control) | GPU (unchanged) |
| torch_cpu | original PyTorch | CPU before H2D |
| triton_gpu | fused Triton K8 | GPU after H2D |
| triton_cpu | fused Triton K8 | CPU before H2D |

**Default remains torch_gpu**, previous behavior unchanged.
Both cumulative BEST source and grouped Tile/Merge code untouched.

## Whole-input A/B gate

`tools/benchmark_h3_flat_full_pipeline_batch.py` now accepts
`--quant-mode`. It still compares pinned individual preparation
and Tile/Merge against grouped preparation and Tile/Merge
on the same saved CPU Q/K/V and measures paired synchronized
wall time. New candidate preflight additionally measures
synchronized preparation stage wall times (`query_gather_h2d_mu`,
`key_value_gather_h2d`, `key_int8_quantization`, `leaf_stats`,
`rowmax_topk_exclusion`). **Those stage values are diagnostic
only**, because inserting per-stage synchronization changes
the execution; timed A/B rounds do not include profiling barriers.
Monitor real PyTorch allocated peak separately for each path.

## Test on RTX 5070 Ti

Windows PowerShell:

    cd "D:\ComfyNodes-flat-split-20261111"
    git fetch origin
    git switch --detach origin/experiment/h3-flat-56head-tile-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat_triton.py h3\htrc_flat_head_batch.py tools\benchmark_h3_flat_full_pipeline_batch.py
    & $py -B tools\benchmark_h3_flat_full_pipeline_batch.py --heads 56 --repeat-available-heads --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 2 --warmups 1 --quant-mode triton_cpu

Report: `D:\ComfyOutput\H3\math_probe\flat_full_prep_quant_ab_*.json`.
First GPU gate tests exact INT8 K/scales, then all grouped final
attention parity and full pinned oracle, then paired wall and
VRAM. If triton_cpu fails compile/parity, test torch_cpu next
to isolate the H2D/casting benefit without the new quant kernel.

GPU compilation, quant exactness, precision and performance UNRUN.
Do not promote until full input-to-output wins over pinned BEST
at or below acceptable VRAM and real model integration gate.
Previous compact dq8/sdq query cache remains independently UNRUN.

