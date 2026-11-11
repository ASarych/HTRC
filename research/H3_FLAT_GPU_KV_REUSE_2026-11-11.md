# HTRC Flat — exact K/V preparation reuse across full Q

Date: 2026-11-11. Experimental branch:
experiment/flat-gpu-kv-reuse-20261111.
Parent full-Q GPU gate at 09688df7b7d174d4d7ac6de13feb4876e4f41e91.
Pinned Flat BEST e47cc7a310936547ab6c7d97bd07526eb6a135fe,
blob-identical to HTRC a8fe43e7aab8d9f951af7b4cba8fbd54f7a61e5c.

## Mathematics and lifetime

For a fixed K/V tensor and raw-head ordering, within ONE attention
call and across different Q windows, these are invariant:
- KBF, BF16 gathered keys;
- VBF, BF16 gathered values;
- INT8 K and FP32 per-token K scales KS, using fused Triton K8.

The stats kernel depends explicitly on each Q window's MU:
G, VB, LZ and SG are query-dependent and MUST be recomputed,
as must rowmax selection, topk/exclusion and Tile/Merge.
This is a K/V-preparation cache, not reuse of attention outputs
or an autoregressive KV cache. It is invalid across denoising
steps, model blocks or K/V mutation.

Optional cache input has strict identity, source tensor mutation
version, captured source head-position, shape, dtype, layout and
CUDA device validation. It fails closed if changed. External
writes bypassing PyTorch version counters require an explicit
caller lifetime guarantee.

## New paired A/B with one retained report

tools/benchmark_h3_flat_gpu_kv_reuse.py

Both arms use identical fused K8, group preparation math and
batched Tile/exact R32 Merge. The uncached control gathers and
quantizes K/V each Q window; the reuse arm does this once per
head group. Other query-dependent work is identical.

All Q windows are checked against pinned BEST and between arms.
Independent original-Flat full oracles are checked at the first,
middle and last Q window for each group; four native-head K8
quantizer gates run before timings. Per-window paired ABBA
medians are reported. Full-query candidate time is reported BOTH
steady-state and with one cache build time PER head group.
The group memory planner adds the expected resident K/V-cache
bytes to its safety estimate and halves groups if GPU OOM occurs;
incomplete group results are not accepted.

The uncached arm runs with an unused cache still allocated for
symmetrical interleaved A/B timing. Therefore its absolute memory
peak is not a stand-alone cold-control memory measurement.
Compare the new measured arms to EACH OTHER, not directly to
the earlier full-Q benchmark with a different group size.

Four distinct native heads (1/16/27/55) are captured; the
56-head case repeats these 14 times. The tested full Q is 40,759
rows/head (40 windows, last padded by 201 Q rows at 16 tiles).
Parity against pinned approximate Flat is NOT parity against
the model's dense attention or proof of video fidelity.

### Windows PowerShell, existing worktree

    cd D:\ComfyNodes-flat-split-20261111
    git fetch htrc
    git switch --detach htrc/experiment/flat-gpu-kv-reuse-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat_head_batch.py tools\benchmark_h3_flat_full_pipeline_batch.py tools\benchmark_h3_flat_gpu_kv_reuse.py
    & $py -B tools\benchmark_h3_flat_gpu_kv_reuse.py --head-cases 4,56 --repeat-available-heads --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 2 --warmups 1

ONE retained output under D:\ComfyOutput\H3\math_probe:
flat_gpu_kv_reuse_ab_*.json.

Both head cases execute in independent processes, with live
window-by-window PowerShell progress. Temporary individual case
reports are removed; one combined JSON contains both results,
summary, parity, K8 exactness, stage medians and memory.

GPU test is UNRUN in the authoring environment. The previous
full-Q benchmark confirmed fused K8 vs PyTorch (4 distinct:
179.444 -> 103.692 ms; 56 repeated logical:
1564.081 -> 848.830 ms). Those were sums of group/window medians,
not real model latency; they test a different question and must
not be treated as a same-group K/V reuse baseline.

## What should happen AFTER this gate

1. Decide by AMORTIZED speedup including cache build, not hot
   window speed alone. Require 100% Q-window parity, original
   full-Flat oracles, no errors and acceptable real VRAM.
   If gain is marginal (<5%) or memory-heavy, stop caching and
   prioritize real model validation.
2. Reuse mutable GPU workspaces across Q windows (the ALLOCATIONS,
   never the query-dependent G/VB/LZ/SG/SEL contents). Measure
   allocator overhead and reserved VRAM separately. With stable
   workspaces, test CUDA Graph capture/replay without masking
   true K/V lifetime and graph pool memory.
3. Profile leaf stats, rowmax, torch.topk, exclusions, Tile/Merge.
   The pending isolated compact INT8 query-cache experiment on
   branch b75f517d80cee94cb867f35c07700e76412be237
   (dq8/sdq) is GPU UNRUN: check actual wins before merging.
   Previous larger QG score caches were negative; avoid repeating.
   If topk/selection dominates, fuse or reduce score/selection
   memory traffic; preserve pinned Flat semantics.
4. Compare K/V storage layouts under real model VRAM: current
   cache duplicates BF16 K and V alongside K8 and scales.
   Consider direct strided native K/V reads or fused K/V packing
   only after bandwidth/allocator profiling, not by default.
5. Before any BEST promotion: capture 56 genuinely distinct H3
   heads, test full-Q at true runtime shape with the H3 checkpoint
   resident on 16 GiB RTX 5070 Ti, select group size by CURRENT
   free VRAM, then measure full denoising/video latency.
   A Flat approximation can be bit-exact to itself yet differ
   from native dense attention; assess actual quality, identity,
   temporal consistency and motion, not only internal parity.

No ComfyNodes production switch in this experiment. Any
model-resident OOM or quality regression is a hard blocker;
choose lower-residency streaming, not speculative caching.
