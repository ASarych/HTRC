# HTRC Flat: memory-to-speed Graph groups, Balanced first

Date: 2026-11-13. Research branch off
experiment/flat-dense-quality-matrix-20261113
@ 0f37448f00963d9950707273a43782167fa64b6c.

## Owner decisions reflected

1. **Balanced (512 exact near leaves) is the default Flat quality
   mode.** Fast (256 leaves) remains explicitly selectable. Existing
   real Balanced Graph video completed; it was NOT a same-condition
   Balanced-vs-Fast or Balanced-vs-Sol timing test. No assumption that
   Fast is the desired speed baseline.
2. **Direct Q/K/V may be valuable primarily for memory savings.**
   An isolated 56-head GPU A/B once found Direct QKV + Torch MU slower
   than gathered baseline; it was NOT a test of what a larger head
   group could do after recycling the saved VRAM. Direct K/V reduced
   *additional packed K/V cache* by 79.5%, while existing model-owned
   QKV remained.
3. Preserve eager, original graph/materialized, split_stream and
   fused_vb as explicit independent paths. Do not promote an untested
   candidate into production or silently replace Sol.
4. Numerical quality is **relative L2 percent against dense all-key
   attention**, not bit-exact inter-Flat agreement. Short contexts
   15,275 and 18,723 MUST be measured even when pursuing long-N gains.

## What is implemented in this research slice

- Source-only estimator h3/htrc_flat_graph_group_plan.py models native
  Direct-Flow staged **BF16 K and V once** plus K8/KS, per-geometry
  Q buffers and Flat G/VB/LZ/SG/selection/Tiles/Merge. The KBF/VBF
  native cache entries are **views of the staged K/V** and therefore
  incur ZERO additional storage; the previous packed planner counts
  them as allocations. Both distinct full-window and trailing-window
  Graph captures are counted even if their tile counts happen to be
  equal but their number of valid Q rows differs.
- New opt-in environment setting:
  APL_HTRC_DIRECTFLOW_GROUP_POLICY=reuse_aware
  (default remains legacy). It chooses a bounded head group candidate
  that minimizes replay count while including storage for *all*
  distinct group-size Graph sessions, including a ragged final group,
  and all distinct query geometries.
- Mode-aware capacity respects budget_mib, reserve_mib, the existing
  128 MiB scratch margin, post-capture maximum extra-reserved VRAM,
  actual GPU free-memory floor, and explicit max-head setting.
  The estimate uses 1.5x stage/workspace bytes plus 64 MiB fixed
  research margin, but **does not predict private CUDA Graph pools**.
  Runtime's actual max-extra and free-memory guards still have final
  authority. No OOM retry or silent fallback.
- The old/legacy group cap default remains **11**. Reuse-aware
  defaults to a max of 32 and is opt-in; APL_HTRC_DIRECTFLOW_GROUP
  can explicitly cap either policy. The session key includes policy
  and selected group cap to prevent reusing stale captured pointers.
- HTRC live backend's optional mode default changes from fast to
  balanced. The ComfyNodes HTRC adapter also needs a matching
  independent change to its APL_HTRC_FLAT_MODE fallback, because
  it always passes mode explicitly. Selecting a different H3 backend
  (Sol/dense) is entirely separate.

## Source-derived planning example (NOT measured GPU latency)

Balanced, 512 leaves, 56 heads, 16 Graph tiles, 1024 MiB total
workspace target, 256 MiB reserve, 768 MiB extra graph cap, free at
plan 5200 MiB, 1.5x head storage safety:

| Tokens | Legacy group / count | Reuse-aware estimate / count |
|---:|---|---|
| 15,275 | 11 / 6 | 14 / 4 |
| 18,723 | 11 / 6 | 14 / 4 |
| 40,759 | 5 / 12 | 5 / 12 |
| 82,000 | 3 / 19 | 4 / 14 |

The legacy counts are source-planner estimates for these specific
settings. The reuse-aware schedule is a candidate for benchmarking,
not a proven speedup. Bigger head groups may lower SM occupancy,
increase compilation/memory-pool overhead, or regress performance.
For 56 heads, 14+14+14+14 uses ONE Graph session size; the
legacy 11+11+11+11+11+1 uses TWO session sizes.

**Graph-owned sk/sv K/V copies are NOT yet removed.** Their stable
addresses are required for captured graphs to process QKV from
different model blocks and denoise steps; directly aliasing model K/V
without a producer lifetime/ownership contract is unsafe. The
research stage removes *planner double accounting*, not the actual
copy operations. Future runtime coordination would need production
QKV to be written into Graph-owned storage before quantization.

## One-file GPU comparison, no expensive video

tools/benchmark_h3_graph_group_policy.py:
- plan-only by default, no CUDA invocation without --execute;
- Balanced default; Fast available explicitly;
- short scope: 15,275 and 18,723 tokens;
- scaling scope: short + 40,759 + 82,000 tokens;
- same distinct BF16 QKV heads/seed per length for both policies;
- exact FP32 softmax over **all K/V** for 64 sampled Q positions,
  output error as relative L2 percentage and per-head/query tails;
- materialized Graph, same 512 near leaves for both policies;
- first-call eager Flat disagreement is diagnostic, never a quality
  gate; both policies should continue on numeric disagreement;
- full-attention warmed time and observed allocated/reserved VRAM
  separately; graph capture not part of warmed throughput;
- runtime failures recorded separately, not invented quality scores;
- ONE persistent report at
  D:\ComfyOutput\H3\math_probe\flat_graph_group_balanced_ab.json.
  Intermediate first-call gate reports use a temporary directory.

Example (separate HTRC checkout with current CUDA runtime):

    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m unittest discover -s tools -p test_h3_graph_group_plan.py -v
    & $py -B tools\benchmark_h3_graph_group_policy.py --scope short --execute
    # Only if useful, the same one-JSON A/B across lengths:
    & $py -B tools\benchmark_h3_graph_group_policy.py --scope scaling --execute

Do NOT run large video or infer Sol parity from this report. Short
quality+speed are mandatory, because memory optimization might help
even at short N. Two model-independent policies compare with exact
dense, not by bit equality with one another.

## Validation status

- New source model and planner: code written, GPU UNRUN.
- New mode-aware Graph path: implemented opt-in, GPU UNRUN.
- Source/CPU contracts: prepared, execution UNRUN unless run.
- Original eager and materialized graph by default: unchanged in
  execution/staging/group policy; Balanced changes only quality
  mode fallback (explicit fast retained).
- Zero-copy graph QKV and adaptive empirical fastest-group policy:
  NOT IMPLEMENTED.
- Real short-N/long-N speed gains and Sol crossover: UNKNOWN.

## RTX 5070 Ti first short A/B report and harness fix

User-run `flat_graph_group_balanced_ab.json` after the preceding CPU test
correction reached the GPU runner. Both legacy and reuse_aware arms at
15,275 and 18,723 tokens ended `RUNTIME_FAILED` with the same exact
`RuntimeError: H3 Direct-Flow is inference-only` message. No quality,
VRAM peak, full-attention GPU time or real speedup was measured.
The planner outputs 14 heads x 4 groups at both short lengths remain
source-derived *estimates only*.

Cause: the GPU A/B harness called `flow.graph_attention_bthd` while
PyTorch autograd was enabled, which the inference-only backend correctly
rejects. The GPU harness's nested `call()` now wraps the backend in
`with torch.inference_mode():` for BOTH first-call capture and warmed
replay timings, restoring the caller's prior state on return.
A new CPU regression test executes the exact nested call source with a
fake backend and asserts autograd-disabled/inference-enabled execution
and state restoration. The actual GPU benchmark on this patched HEAD
remains **UNRUN** pending a new local invocation. Do not count either
of the failed attempts as numerical-quality or throughput evidence.

## Next slice: allocator phase evidence and bounded cap sweep

The next opt-in short-N benchmark compares **five sequential arms** in
one `flat_graph_group_balanced_ab.json`: legacy11 and reuse-aware
max-head caps 7/8/11/14. A cap is NOT a forced group size: the
reuse-aware planner may select a smaller feasible group; check
`group_size` / `actual_head_group_cap` before drawing conclusions.
The 768 MiB `APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB` hard guard and
the independent 256 MiB free-memory floor remain unchanged.
The original runtime default stays legacy.

The first research capture reports `memory_trace` with allocator
`allocated_mib`, `reserved_mib`, `free_mib`, plus deltas from
the baseline for stage KV/cache, each geometry's workspace,
post-warmup, post-capture, and post-session. On fail-closed
`MemoryError`, the benchmark copies `memory_trace` and
`memory_at_failure` into its single aggregate JSON **before**
deleting the temporary per-arm gate file. These counters establish
where the extra **allocator reservation** appears, but do NOT isolate
CUDA Graph private pools as a separate directly measured category.
No extra profiling is done on warmed calls.

The benchmark no longer retains full GPU outputs between candidates.
Only 64 sampled query rows per arm are retained on CPU for the
legacy-vs-candidate disagreement percentage. Same BF16 input bank
and FP32 dense sampled-query reference are reused. The report records
actual cap, numerical quality and full attention latency for each
successful arm; failed arms retain error and memory evidence without
invented speedup. When some arms fail, successful comparisons remain
available and status is `MEASURED_WITH_FAILURES`.

Suggested single local run after updating HTRC:

    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m unittest discover -s tools -p test_h3_graph_group_plan.py -v
    & $py -B tools\benchmark_h3_graph_group_policy.py --scope short --execute

The local GPU A/B, CUDA compilation and performance measurement
remain **UNRUN** for this slice. No video test, mode promotion, cap
increase, or Sol comparison is implied.
