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

## RTX 5070 Ti second short A/B: memory gate contamination suspected

User-provided `flat_graph_group_balanced_ab(2).json` is from
`2e9b19c13985cf5a8af1ed1034cb45b08fa62dee`, synthetic BF16
56-head QKV, balanced materialized Graph, 16 tiles, 768 MiB extra cap,
64 sampled FP32 all-key dense query reference. Results:

| Tokens | 7-head full GPU time | 8-head full GPU time | Improvement for 8 | 7 / 8 extra reserved MiB |
|---:|---:|---:|---:|---:|
| 15,275 | 66.852 ms | 64.716 ms | 3.19% less time | 498 / 570 |
| 18,723 | 90.903 ms | 87.979 ms | 3.22% less time | 550 / 648 |

Legacy11 and reuse11 failed the allocator `extra_reserved <=768 MiB`
gate at 796 / 868 MiB. Reuse14 failed at 888 / 934 MiB.
Neither failure means physical free VRAM was exhausted: NVIDIA
`mem_get_info` still reported at least 12.9 GiB free.
The first captured 11-head session was only +286 / +276 MiB
reserved vs baseline. At the second head group's pre-replay
`after_session` snapshot (AFTER the prior group's Graph+eager
comparison), reservation had climbed to +712 / +766 MiB while
allocated grew much less. After the third group reservation hit
+796 / +868 MiB without equivalent live allocation growth.
Likewise, 14-head groups went from +384 / +336 MiB after capture
to +888 / +934 MiB after the next group's start.

**Interpretation:** the eager Flat diagnostic allocates large
temporary buffers and is a strong candidate for contaminating the
Graph-only reserved-memory gate. The old trace did not sample
immediately after Graph replay and before eager, so attribution
is a hypothesis, not proven causation. `reserved-allocated`
cannot be treated as private CUDA Graph pool size.

The 7- and 8-head samples produced *identical relative L2 error*
against dense on the same QKV bank: 55.329% (15,275) and 60.563%
(18,723). These are high approximation errors for random QKV.
They are not graphical-quality or video evidence; they establish
no quality regression from the change in grouping by themselves.

## Next opt-in experiment: isolate Graph from eager diagnostic

For `compare_only=True`, first-call Graph replay and allocator
trace remain active, but the eager Flat parity diagnostic is **skipped**.
That diagnostic is redundant for sampled dense quality benchmarking
and adds unrelated allocations to the Graph memory A/B. The report
explicitly uses `eager_first_call_diagnostic=SKIPPED_COMPARE_ONLY`,
`exact_flat_parity=null`, `all_groups_bit_exact=null`,
`speedup_eager_over_graph_gpu_by_group_sums=null`; it does NOT
pretend parity was checked. `after_group_graph_replay` is sampled
to distinguish Graph replay footprint from the previous eager phase.
**Real production invocation (`compare_only=False`) remains strictly
bitwise checked against eager Flat, fail-closed and unchanged.**

The same five-arm one-JSON short sweep and hard 768 MiB gate
are retained. Pairwise comparisons now use legacy when it completes,
otherwise the first successful arm, with `pairwise_reference_arm`
clearly reported. Repeat the same local 15,275/18,723 synthetic
A/B before revisiting 40k/82k or selecting a group policy.

No inference performance gain for 11/14 has yet been measured.
This slice's CPU unit execution, CUDA checks and GPU run are **UNRUN**;
nothing was promoted to the default runtime.

## RTX 5070 Ti third short A/B: contamination removed, all arms measured

User-supplied `flat_graph_group_balanced_ab(4).json` reports
`MEASURED` at both 15,275 and 18,723 tokens, balanced materialized
Graph, 56 distinct BF16 heads, 16 tiles, exactly the same input per
arm and 64 sampled all-key dense FP32 query positions.
Both the 768 MiB Graph-extra gate and the production default
`legacy` were unchanged.

| Group / policy | 15,275 full GPU ms | 18,723 full GPU ms | extra reserved MiB (15k / 18k) |
|---|---:|---:|---:|
| Legacy cap 11 | 68.404 | 96.238 | 332 / 304 |
| Reuse cap 7 | 68.407 | 100.462 | 194 / 174 |
| Reuse cap 8 | **65.435** | **89.563** | **216 / 196** |
| Reuse cap 11 | 68.254 | 102.583 | 332 / 304 |
| Reuse cap 14 | 65.797 | 89.954 | 384 / 336 |

The 8-head version was faster than legacy11 by 4.34% and 6.94%
on the respective short lengths. The 14-head version was only
0.55% / 0.44% slower than group 8, not a conclusive universal
performance difference with only three timed repeats. Group 8
uses 168 / 140 MiB less additional reserved VRAM than group 14.

All successful variants were bit-equal to legacy on **sampled 64
query positions**, not necessarily validated bitwise for the
entire output. Relative L2 vs dense FP32 was identical for
all arms at 55.329% / 60.563% on synthetic QKV; this is not
real H3 model/video quality evidence.

The previous group-11 Graph gate failure at 796/868 MiB extra
disappeared: after skipping research eager Flat diagnostic it
used 332/304 MiB. Likewise, group14 fell from 888/934 MiB
to 384/336 MiB. Memory stayed flat across reused Graph replays.
This confirms that an unrelated eager-Flat first-call allocation
was polluting the Graph-only memory gate; it does NOT establish
private Graph-pool size and is NOT a reason to alter the production
strict first-call parity path.

## Follow-up: measure group 28 without changing hard VRAM cap

The replay-count-only reuse planner favors group 14 at short N,
but real end-to-end latency did not confirm a meaningful win
over 8. A 28-head divisor produces two balanced head groups,
potentially increasing arithmetic density while reducing Graph
replays. Its real performance and reserve usage are unknown.
The existing conservative stage-budget model would exclude it
before measurement, although 14-head actual Graph reservation
suggests it *might* fit under the same 768 MiB hard gate.

This research slice adds a compare-only forced group override:
`APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP=28` alongside
`APL_HTRC_DIRECTFLOW_GROUP_POLICY=reuse_aware`,
`APL_HTRC_DIRECTFLOW_GROUP=28`, and `compare_only=True`.
It bypasses only the conservative **planning estimate**;
it does NOT bypass the actual post-capture extra-reserved and
free-VRAM guards. The override is rejected on production calls
and invalid group sizes; no changes to default `legacy`,
eager parity checks or automatic fallback.

The benchmark now accepts `--arms reuse_cap8,reuse_cap14,reuse_force28`
and saves ONE JSON for the three arms. If 28 exceeds the actual
768 MiB reservation gate, the failure is preserved with memory
trace and the remaining successful comparisons are still reported.
The pairwise reference will be the first successful selected arm,
normally cap8. No full model, real video or Sol parity is inferred.

Example single local short run with five timed rounds per arm:

    & $py -B -m unittest discover -s tools -p test_h3_graph_group_plan.py -v
    & $py -B tools\benchmark_h3_graph_group_policy.py --execute --scope short --mode balanced --heads 56 --tiles 16 --arms reuse_cap8,reuse_cap14,reuse_force28 --budget-mib 1024 --reserve-mib 256 --max-extra-mib 768 --sample-queries 64 --rounds 5

Current slice CPU unit test execution, GPU compilation, and
forced-28 end-to-end timing are **UNRUN**. No change is promoted
to production.
