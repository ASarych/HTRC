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

## Local CLI smoke failure and fix (2026-11-11)

Received `flat_gpu_kv_reuse_ab_20261111-224929.json`.
Top-level `completed: false`, `cases: []`; only head case 4
was started. The child process failed *before CUDA benchmarking* with
`AttributeError: 'Namespace' object has no attribute 'case_heads'`
at `single_case()`, because argparse defined `--_case-heads` as
`args._case_heads`. This is a CLI wiring bug, not an A/B result,
K/V-cache parity failure or GPU OOM. The benchmark source now reads
`args._case_heads`; a CPU-only CLI regression test checks all used
argparse names and child flag consistency at
`tools/test_h3_flat_gpu_kv_reuse_cli.py`.

The original uploaded JSON is a failure artifact, not performance
evidence. CPU static argument-reference audit: PASS (13 used
properties, all declared). CPU unittest / Python compile on the
published fix: UNRUN; actual GPU cache test: **UNRUN** until rerun
from corrected HEAD. Only one persistent JSON is emitted.

## GPU A/B result, 2026-11-11 22:56:30: correctness PASS, first-build time contaminated

User-uploaded one-file JSON `flat_gpu_kv_reuse_ab_20261111-225630.json`,
candidate exact HEAD `2a971286a5ceb4ebd73026708512ac23aa870afb`.
Top-level `completed: true`, 4+56 embedded cases completed,
`errors: []`, OOM retries empty. RTX 5070 Ti 16 GiB,
40,759 query rows/head (40 windows; final pad 201).
Native captured raw heads 1/16/27/55 repeated for 56-head
proxy. All **360** head-group/window comparisons were bit-identical
to pinned and to each other (1080/1080 pairwise PASS checks);
original independent Flat-oracle checks 216/216 PASS.
All eight native-head K8 exact gates (4 per case):
0 INT8 row mismatches over 40,752 K rows/head; max scale error 0.

| Case | Uncached fused-K8 full-Q window-group medians | Cached reuse STEADY | Steady ratio | FIRST cache build total | Cold-start amortized ratio |
|---|---:|---:|---:|---:|---:|
| 4 distinct heads (H4 ×40 windows) | 95.65785 ms | 70.54295 ms | **1.3560x** | 622.9332 ms | **0.13794x (lose)** |
| 56 logical proxy heads (H7 × 8 groups ×40 windows) | 1365.4917 ms | 837.5103 ms | **1.6304x** | 889.1519 ms | **0.79083x (lose)** |

The first GPU cache build is anomalously slow:
622.933 ms for the first H4, 838.225 ms for first H7.
**Next seven H7 builds took 7.11–7.41 ms**, median ~7.29 ms.
An unmeasured first-use Triton specialization/compilation is a
strong hypothesis; the report cannot prove that compiler time was
the sole cause. The uncached ABBA arm was warmed; the first cache
build was not. The original `speedup_amortized_with_build`
therefore answers the *cold-build* question but is **not** a
representative recurring GPU attention-call cost.

Recorded cache residency ~100.114 MiB at H4 and ~175.200 MiB at H7
(plus the captured 119.411 MiB QKV). Both arms were timed with
cache allocated; reported peak memory comparisons are NOT
standalone no-cache memory requirements. These are CPU host
wall sums over window/group medians, **not end-to-end H3 video**.

### Corrected cost-accounting gate

The subsequent code change separates:
- `kv_cache_first_build_wall_ms`: first build, including
  any JIT/allocator cold start, retained for cold-start analysis;
- `kv_cache_build_warm_samples_ms`: 3 measured **warmed**
  cache builds per group (configurable), with median
  `kv_cache_build_wall_ms` counted ONCE per group;
- `speedup_amortized_with_build`: real repeated-call style
  full-Q estimate, including a single warmed build per group;
- `speedup_cold_start_including_first_build`: distinct cold-start
  estimate, preserving first-use cost.

Numerical attention path, cached and uncached arms, pinned
oracles, head planner, output format and 4+56 one-JSON workflow
are unchanged. This accounting fix is **GPU UNRUN**.
Do not claim an amortized positive speedup until new report
confirms it; do not promote global BEST based on this proxy.

## Confirmed warm-cache result (user-run, 2026-11-11 23:08:58)

Report: `flat_gpu_kv_reuse_ab_20261111-230858.json`.
Exact candidate SHA: `9519166116c8c2bf3027cebe73ba7ded9c3a6acb`.
Pinned reference: `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.
NVIDIA RTX 5070 Ti 16 GiB; saved H3 QKV capture; 40,759
valid queries/head, 40 windows of 1024 with 201 padded Q rows
in the final window; 4 actual native heads 1/16/27/55 or
56 **logical** heads in 8 groups of 7 by repetition.

All cases `completed: true`, `errors: []`, 0 OOM retries.
360/360 window-group combinations have bit-exact triple
comparisons (1080/1080 pass, rel-L2=0, max abs=0).
108 independent head-window native Flat oracle rows (first/middle/
last per group) pass for both cached and uncached.
8/8 independent native K8 byte+scale gates pass, 0 differences.

| Case | Uncached fused K8 full-Q sum medians | KV reuse steady | Warm cache build | Reuse incl. warm build | Real repeated-call speedup | Cold first-build penalty | Cold-start speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 distinct heads | 96.92545 ms | 72.27435 ms | 4.17830 ms | 76.45265 ms | **1.26778x** | 589.57320 ms | 0.14645x |
| 56 logical repeated heads | 1331.86450 ms | 816.42735 ms | 56.16850 ms (8 groups) | 872.59585 ms | **1.52632x** | 648.14060 ms across 8 groups | 0.90939x |

Cache residency 100.114 MiB/H4, 175.200 MiB/H7. Measured
max PyTorch allocated (with cache already resident in **both**
paired paths): uncached vs cached 385.61 vs 295.25 MiB H4;
584.16 vs 425.79 MiB H7. Do NOT interpret these as
production VRAM without the actual model weights/graph pools.
GPU input capture occupies another 119.411 MiB.

Only first build of each new head-count process is anomalous:
~589.6 ms H4 and ~598.3 ms H7; latter seven H7 first builds
are ~7.0–7.3 ms. **Compiler/JIT contribution is likely but not
independently proven**. Warm construction medians were measured
in the same run. Retain both cold and warmed costs; do not
attribute all first-call overhead to quantizing K/V.

**Decision:** positive warm repeated-call K/V prep reuse result
with correct pinned-Flat parity, no production promotion.
The next isolated experiment should reuse the *allocations* of
query-dependent stats, selection, exact output and partials
across Q windows but recompute their contents every time.
Compare KV-cached without workspaces vs same KV-cached with
workspaces, then move to real model-resident H3 gating.
