# HTRC 56-head captured Graph: one-call CUDA kernel activity profiler

Date: 2026-11-16.
Branch: `experiment/flat-memory-to-speed-balanced-20261113`.
Baseline source HEAD:
`5745808a0a5ded3cb295bc82cfcb86456ddeceac`.

## Evidence and hypothesis

User-held `flat_56head_graph_stage_trace.json`, SHA-256
`39a6145cd41e79b4f0a8cd9e90937244f63ce55ea08853eef8a206ec737e2d1e`,
66,450 bytes. Two independent 56-head synthetic BF16 QKV banks,
15,275 tokens, Balanced/512, group8, all output rows Graph/eager
bit-exact; instrumented result also bit-exact.
Full warm Graph GPU Event median **66.690849304 ms**, seven
samples, 0.918676% min/max spread. One Graph boundary trace
measured **62.477376 ms** across **105** captured per-window
Graph replays, against **66.056831 ms** full instrumented GPU
Event time (~94.58%). External K/V refresh+quant, Q staging
and output copy account for roughly 3.36 ms of stream
intervals combined, not an attractive first optimization
target. Graph replay kernels include Stats, selection/TopK,
Tile and Merge, whose actual proportions remain unknown.

A separate eager observer yielded 78.03 ms for
query stats/selection and 72.77 ms for Tile in THIS run,
with a different 13-head group policy; prior eager timing
was much less stable. Do not transfer either eager ratio
onto the 66.69 ms captured Graph.

## One isolated next measurement

Reuse the already passing full-output HTRC benchmark.
New opt-in `--graph-kernel-profile` (mutually exclusive with
`--graph-stage-trace`) runs **after** two strict independent
full-output BF16 numeric banks and seven ordinary warmed
full 56-head Graph timings, on bank 1 in a still-warm
existing Graph session.

Profile **one additional full Graph call** using the
installed `torch.profiler` CPU+CUDA activity collector.
No Nsight hardware counters, kernel variant compilation
or new H3 math. Do not create an exported giant trace file.
Immediately compare the profiled full output with the
verified unprofiled output using `torch.equal`; any
mismatch aborts, preserving one fail-closed JSON.

Aggregate CUDA-device FunctionEvents by exact **kernel
name**, counts and positive per-device activity durations.
Exclude CPU launch events from device kernel duration
sums (no CPU/GPU double-count). Categorize only by
source-like names, retaining an explicit `unclassified`
bucket and full per-name data; labels are heuristic:

- `_stats_kernel`: leaf_stats;
- `_rowmax_kernel`: rowmax_selection;
- explicit `topk` or `top_k`: topk_named;
- `_tile_kernel`: tile;
- `_merge_tail_kernel` or `_merge_kernel`: merge;
- explicit memcpy/memset: device_copy;
- every other activity: unclassified, **no guessed attribution**.

The `gpu_device_activity_total_ms` is the SUM of
positive GPU activity durations observed by Kineto; it
is not automatically equal to CUDA Event whole-call time
and might include copies outside replay. The report
retains `profiled_full_gpu_event_ms`,
`profiled_full_wall_ms`, the original unprofiled
median, the profiling ratio, activity names and raw
per-name statistics, missing-duration count, category
percentages, and `profiled_execution_more_than_20pct_slowdown`.
Nothing in this report is a speed improvement or
production promotion. A label based on a kernel name
is not formal proof of stage ownership.

**Fail closed if no CUDA-device activities can be observed**
(e.g. CUPTI unsupported), or if profile output differs.
Do not silently infer that GPU spent zero time in Graph.
Do not open Nsight counter permissions or disable GPU
safety checks. If >20% full-call slowdown, preserve the
data with the warning and do not use raw percentages
to rank uninstrumented kernel latency.

## Files and compatibility

- `tools/htrc_graph_kernel_profile.py`: standalone
  Torch profiler adapter, source-name classification,
  fail-closed checks and one-call aggregation.
- `tools/test_htrc_graph_kernel_profile_cpu.py`:
  CPU-only numeric guards, classification, unknown
  bucket, no CPU double-count, invalid activity, overhead
  flag and opt-in source/runner contracts.
- `tools/benchmark_h3_56head_stage_attribution.py`:
  new lazy `--graph-kernel-profile` option; existing
  56-head baseline and `--graph-stage-trace` unchanged.
- `tools/run_h3_56head_stage_attribution_safe.ps1`:
  new `-GraphKernelProfile` switch; three explicit
  single-string Python CLI branches (no one-element
  PowerShell array splatting), one new output path,
  unchanged exact-SHA/CPU suite/VRAM preflight.
- `research/H3_56HEAD_GRAPH_BOUNDARY_TRACE_2026-11-16.md`:
  append grounded completed external boundary trace.

Single user-held output, NOT the previous JSON:
`D:\ComfyOutput\H3\math_probe\flat_56head_graph_kernel_profile.json`.
Use only one exact-head disposable checkout after
all other GPU workloads are closed. No production
`h3/` or Comfy Desktop modifications, no model/video
inference. No source CUDA/Triton kernel changes.

At authoring: 8 self-contained CPU tests PASS,
2 full-checkout source tests SKIP (not run under
Windows checkout yet), Python py_compile PASS.
Full Windows CPU/source suite, RTX GPU profile, and
model/video performance remain **UNRUN**.

## 2026-11-16 — uploaded full 56-head kernel profile: corrected evidence

Raw user-uploaded `flat_56head_graph_kernel_profile.json`:
**13,064 bytes**, SHA-256
`9e8fa930b74aea4d09a787d166af52c4788110532090db1d2e8a8dca32e28cc6`.
Exact tested source SHA:
`d2bd8d1480298ea3fc6750d9121de2c51b9ad7ab`.
Raw JSON remains user-held and is NOT overwritten.
`status=PASS_GRAPH_KERNEL_PROFILE_DIAGNOSTIC_ONLY`,
`completed=true`, both distinct BF16 QKV banks **full-output
Graph-vs-eager bit-exact** for all 56 heads, independent final
whole-output bit-exact and profiled-call exactness PASS.
This test uses synthetic, distinct BF16 QKV, 15,275 tokens,
Balanced/512, group size 8, NOT a model/video run.

Normal full 56-head Graph CUDA Event median: **66.667778015 ms**
(seven timings, max/min spread **0.850032%**); host wall
median **68.250600000 ms**; additional first-call reserved
**590 MiB**. One instrumented Graph call:
**73.319488525 ms** GPU Event, **82.208 ms** host wall,
ratio to uninstrumented median **1.099774** (~+9.98%).
No GPU hardware counter permissions/counters were needed.

### Critical aggregation bug found in this successful report

The original Python aggregator counted **1,212** CUDA-typed
profiler events totaling **133.545215 ms** and labeled
**52.3345%** of that sum `unclassified`.
But the top event is NOT a kernel:
`HTRC_56HEAD_FULL_GRAPH_KERNEL_PROFILE`, a parent
`torch.profiler.record_function` annotation lasting
**67.469985 ms**. The profiler exposed that enclosing
range among CUDA-typed FunctionEvents and the aggregator
improperly summed it **in addition to its child kernels**.
This is double-counting, not a 133.5-ms execution and
not evidence of unclassified compute taking 52.33%.

The source-accurate, post-processed device activity sums
computed from the **14 other original per-name rows**:

| Actual CUDA activity type | Calls | Device total ms | % of corrected device duration |
|---|---:|---:|---:|
| `_tile_kernel` | 105 | **41.456084** | **62.7407%** |
| `_stats_kernel` | 105 | **15.082391** | **22.8261%** |
| `_rowmax_kernel` | 105 | 2.305353 | 3.4890% |
| Named ATen TopK kernel | 105 | 1.926413 | 2.9155% |
| `_merge_tail_kernel` | 105 | 1.894362 | 2.8670% |
| Memcpy DtoD | 14 | 0.990393 | 1.4989% |
| Other non-marker activities | 672 | 2.420234 | 3.6628% |
| **Total excluding enclosing annotation** | **1,211** | **66.075230** | **100%** |

The marker itself is separately preserved as
`excluded_non_kernel_profile_annotation`, 1 entry /
67.469985 ms in the *updated profiler implementation*.
It is excluded from count, percentages and device duration.
This correction does NOT alter the original raw JSON or
GPU readings, and does NOT require another GPU run.
The corrected 66.07523-ms sum is **not** supposed to
equal the 73.31949-ms profiled full CUDA Event interval:
the difference ~7.24426 ms is not proven compute,
and activity/event times may differ due to queue gaps,
launch/profile overhead and timeline semantics.

### Conclusion and next engineering boundary

**Tile + Stats constitute 85.5668% of the corrected
recorded GPU activity duration**, across exactly 105
calls of each; per invocation they average
**394.820 us Tile** and **143.642 us Stats**.
This identifies the dominating recorded compute families,
not a full-model quality or A/B speed improvement.
Do not infer that removing either kernel saves its
entire recorded time in a different implementation.
The `torch.profiler` device activity names are only
heuristically linked to source.

The source of `_tile_kernel` at this exact HEAD
contains far-field online softmax over ~954 full leaves,
then exact-near online softmax over **512 x 16 = 8192**
selected exact key tokens, across four splits, using
BN64 for far and near in the pinned kernel. The near
path has substantially more QK/PV dot columns; that is
a **source-level workload hypothesis**, not an isolated
measured near-vs-far time ratio. It makes a distinct
near-path useful-work/dataflow hypothesis worth
considering next, rather than retesting occupancy-only
changes or altering far associative groups.

Prior results already close these specific mechanisms:
- RB2+Late-Q: 36.40% *slower* standalone Tile A/B/B/A;
- near-PV-acc: 0.465% *slower* standalone Tile;
- mask-G: numeric PASS but ~3.60% Tile-only gain,
  below pre-registered 5% promotion gate;
- cross-BN64 far compaction: FAIL strict numeric
  bit-exact PL/PACC (PACC ~75% mismatches);
- far-VB `.cg`: speed unresolved under unstable
  precondition, repeated timing closed.

**Engineering decision:** correct the profiler's
parent-marker double-counting and add a CPU regression.
Then build ONE materially different near-path
work-reduction hypothesis, with source/geometry and
strict two-bank PM/PL/PACC bit-exact gate BEFORE any
timing. If numeric parity fails, STOP; do not relax
numerical/quality contract. If it passes, test one
complete **56-head Graph A/B/B/A** under matched inputs,
stability and VRAM gates, not a new standalone
Tile-only sweep. Never enable on production defaults
without complete H3/video evidence. No new GPU
execution or candidate kernel was made by recording
this result.
