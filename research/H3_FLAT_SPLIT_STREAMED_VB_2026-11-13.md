# HTRC Flat: selectable bounded VB split streaming (2026-11-13)

Base: ASarych/HTRC experiment/flat-long-context-stream-score-20261113
@ 1c37547c699cc05a4f1b26a2bbb0263b095bec69.

## Owner contract: old modes remain

This is a research-only opt-in extension; nothing replaces the
accepted eager Flat or the successful Direct-Flow materialized Graph.
Their default branches and normal kernel specializations remain
unchanged. No reduced exact-leaf count, quality coefficient, changed
selection, or unrequested ComfyNodes integration.

| Dispatch | APL_HTRC_DIRECTFLOW_VB_MODE | Behaviour / status |
|---|---|---|
| eager (default APL_HTRC_FLAT_DISPATCH) | not consumed | Original live Flat, unchanged |
| graph + materialized (default) | materialized | Previously successful CUDA Graph with full G/VB/LZ/SG; old path |
| graph + split_stream | split_stream | Experimental bounded VB, current research code; GPU UNRUN |
| graph + other value | unrecognized | Explicit ValueError, fail closed |

fast = 256 exact leaves and balanced = 512; centre group1.
Graph tiles16/32/64 are independently selectable in this research
branch, with old tiles16 still the default. graph+split_stream is
NOT numerically approved until the full attention A/B proves it.

## Implemented producer/consumer (NOT a complete optimized score-only algorithm)

1. For each native post-RoPE real H3 head group and Q window,
   _stats_kernel(SCORE_ONLY=1) calculates exactly the old
   INT8 G, FP32 LZ, FP32 SG. It deliberately does not load V or
   perform/store the BF16 VB dot. Rowmax/topk/IDX/EXCL then executes
   the original fixed Flat selection on all leaves. This stage still
   materializes full G/LZ/SG; it does NOT become linear in tokens.
2. New _vbar_split_kernel recomputes BF16 VB for only one of the
   four fixed Tile split ranges from the exact original BF16 K/V and
   centre MU, with the same attention scale and weighted V dot as
   _stats_kernel. It writes one compact [H,centres,split_leaves,128]
   scratch. The full VB table is never allocated in this mode.
3. Existing _tile_kernel, compiled with explicit
   VB_SPLIT_STREAM=1, SPLIT_ID, SPLIT_COUNT, consumes that scratch
   with the old far-QK/PV, exact near 256/512 leaves, split-order,
   online-softmax and tail/merge math. Then the next split may
   overwrite the same scratch. Normal _tile_kernel remains its
   original default specialization VB_SPLIT_STREAM=0.
4. Captured CUDA Graph records producer -> Tile dependency on the same
   stream, repeated four times, and uses the existing cross-stream
   CUDA-event dependency for successive model blocks. Geometry cache
   keys include VB_MODE to prevent mixed buffer layouts.
5. The first real Graph attention call still recomputes eager Flat
   for every native head group and requires exact BF16 output equality
   (torch.equal). Failure is fatal and logged once. No silent fallback
   to eager or Sol.

Streamed VB reduces peak transient VB scratch to approximately
one-quarter (rounded for the actual far BN=64), not the total logical VB
write+read traffic per attention invocation. In fact it repeats
K/V reads and adds producer launches: it can be slower. Its purpose is
to validate a bounded producer/consumer primitive before moving to
a fused or register/shared-memory direct-VB Tile path.

## Model and memory scope

At 82,000 tokens, 56 heads, dim128, fast=256, four splits:
- Full BF16 VB scratch for 11 heads x 16 centres x 5,125 leaves:
  230.9 decimal MB.
- One split scratch 11 x 16 x 1,344 leaves:
  60.6 decimal MB, ~73.8% smaller.
- G is still 115.5 decimal MB for the same window/group and remains
  fully materialized. No full-H3 memory admission claim follows.

This stage does not solve 223,605-token Full HD where materialized
BF16 Q/K/V/output alone was ~11.94 GiB, before transient attention
and model weights. Source-derived graphs and VRAM checks remain gates.

## Validation prepared: one JSON, no video required

tools/test_h3_flat_streamed_vb_contract.py: CPU-only syntax, kernels,
default branches, scratch and producer/consumer ordering (10 tests).
tools/benchmark_h3_flat_streamed_vb_ab.py: synthetic two different
nonzero BF16 Q/K/V banks of native H3 shape, compares the two Graph
variants each to the eager oracle (all groups), compares outputs between
Graph variants bit-for-bit for both banks; independently compares the
second bank of each Graph variant against the native eager Flat, and
checks that changing all Q/K/V
changes the output, measures warmed repeated Graph replay with all
K8/QKV and window costs included. Output is ONE JSON.
This is not model-resident H3, Sol, or video.

A developer on a CUDA machine can run in a new isolated HTRC
worktree, without modifying the ComfyNodes checkout:

    $src = "D:\ComfyNodes-flat-split-20261111"
    $wt = "D:\HTRC-stream-vb-20261113"
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    if (Test-Path $wt) { throw "Refuse to overwrite worktree" }
    git -C $src fetch htrc
    if ($LASTEXITCODE -ne 0) { throw "Git fetch failed" }
    git -C $src worktree add --detach $wt htrc/experiment/flat-long-context-streamed-vb-20261113
    if ($LASTEXITCODE -ne 0) { throw "Worktree add failed" }
    Push-Location $wt
    try {
        & $py -m unittest discover -s tools -p test_h3_flat_streamed_vb_contract.py -v
        if ($LASTEXITCODE -ne 0) { throw "CPU contracts failed" }
        & $py -B tools\benchmark_h3_flat_streamed_vb_ab.py --tokens 15275 --heads 4 --mode fast --tiles 16 --rounds 3
        if ($LASTEXITCODE -ne 0) { throw "GPU bitwise output A/B failed" }
    } finally { Pop-Location }

One output:
D:\ComfyOutput\H3\math_probe\flat_streamed_vb_ab.json.

Do not launch another long video or promote this backend unless
GPU full-attention parity PASSes. If the new kernel is slower or
does not give exact output parity, label NO_GO and retain original
materialized Graph as the active best.

## Remaining work

- GPU compilation, CPU contract execution, two-bank full-attention
  parity, actual VRAM and warmed A/B: UNRUN.
- Whole video/56 native real-model heads with selected streamed mode:
  UNRUN.
- Fully fused G/LZ/SG/SEL score and direct VB-to-Tile register/shmem
  consumer eliminating total global VB traffic: NOT IMPLEMENTED.
- Sol crossover (40k/82k/full HD): still UNKNOWN.
- Default eager and old materialized Graph: unchanged.

## 2026-11-14 — guarded Balanced Tile32 VB candidate GPU triage

The old split_stream source contract predated the merged fused_vb
mode and had stale literal assertions on the Graph dispatcher,
SCORE_ONLY and session-key shape. These assertions now match the
existing three-mode implementation. No HTRC runtime math changed.

New tools/run_h3_vb_candidate_ab_safe.ps1 reuses the existing
full-output GPU A/B tools; a temporary exact-HEAD checkout
runs the selected CPU source contract before any GPU allocation.
It rejects active ComfyUI port 8188 and Blender. Only one
candidate is measured per invocation, and only one persistent
JSON is saved. Installed ComfyNodes and models are untouched.

The FIRST candidate is split_stream. The separate fused_vb
candidate is available later but is not implicitly tested.
Initial geometry: 15,275 tokens, 4 distinct BF16 heads,
Balanced with 512 exact leaves, Tiles32, five warmed full
attention rounds, 1024 MiB budget and 256 MiB reserve.
Both independent banks must be bit-exact to the old eager Flat
control and bit-exact between Graph variants.

Runner pins STATS_BT=16, legacy grouping, 768 MiB actual
extra-reserved Graph cap, 256 MiB minimum free memory.
Only the unrelated eight-head Tile32 statistics preflight is
disabled, since it requires eight heads while this gate uses four.
No change to default backend and no general capacity cap raise.

The single report is:
D:\ComfyOutput\H3\math_probe\flat_streamed_vb_ab.json

The existing first-call eager parity diagnostic is retained,
so its memory counters are NOT isolated Graph-pool or true
whole-model VRAM comparisons. A positive short-context result
needs 40,759-token GPU confirmation and model-resident VRAM
verification before any production adoption.

Current CPU test execution and CUDA candidate A/B remain UNRUN.


## 2026-11-14 — Completed RTX 5070 Ti Balanced Tile32 split_stream GPU A/B

Evidence: user-uploaded `flat_streamed_vb_ab.json` produced by
the single-JSON exact-HEAD research runner at
`349fdf294687cd6e7e9fdfac274d1eff4116c94d`.
The uploaded artifact reports `status=PASS_NUMERIC` and
`completed=true`. Scope: **synthetic** two distinct BF16 Q/K/V
banks, 4 distinct heads, 15,275 tokens, Balanced (512 exact
near leaves), Tile32, 5 timed complete-attention repetitions;
no H3 checkpoint, no denoise, no VAE, no video, no Sol.

| Metric | materialized | split_stream |
|---|---:|---:|
| Warm GPU event per full attention | 5.658144 ms | 8.031245 ms |
| Synchronized wall time per attention | 6.255900 ms | 8.383780 ms |
| First-call Graph/eager output parity | bit-exact | bit-exact |
| Second-bank Graph/eager output parity | bit-exact | bit-exact |
| First-call extra reserved (not isolated) | 320 MiB | 0 MiB |
| Capture + parity wall (cold effects included) | 18,474 ms | 33,433 ms |

Full Graph materialized-vs-stream outputs matched bit-for-bit
for **both** banks. Candidate is **41.94% slower on warmed GPU
events** and **34.01% slower on synced wall time** relative
to the materialized control. The report's
`wall_ratio_materialized_over_split_stream=0.746190859`
confirms no wall-time speed benefit.

Do **not** interpret 320 MiB vs 0 MiB first-call
`extra_reserved_mib` as 320 MiB memory savings. The modes
run sequentially in one PyTorch process with allocator caching,
prior Q/K/V/output retention and private Graph-pool reuse;
both numbers are per-call allocator reservation deltas, not
isolated whole-model peak memory. Similarly first-call
capture + eager oracle can include cold compilation and
different cache states; those durations are not a warmed
throughput comparison.

**Decision for speed: NO_GO.** Keep the materialized
Tile32/BT16 backend and stock fallback. Do not run longer
video or 40k/82k throughput benchmarks for this implementation
on the basis of this short result. Retain split_stream
as a research option: actual independent-process/model-resident
VRAM advantages remain **UNPROVEN**, not rejected. No
production kernel, quality policy, ComfyNodes integration,
or source arithmetic changed by this evidence commit.

**Next research gate, only if desired:** already-implemented
`fused_vb` in-Tile candidate, using the same short four-head
Balanced/Tile32 two-bank fail-closed runner with
`-Candidate fused_vb`. It may be substantially slower due
to repeated native K/V reads per centre/leaf; numerical
parity and real GPU timing remain UNRUN until local
measurement. If that also loses, stop experimenting with
the existing direct recomputation geometries and design a
cooperative multi-centre reuse strategy before another kernel
implementation.
