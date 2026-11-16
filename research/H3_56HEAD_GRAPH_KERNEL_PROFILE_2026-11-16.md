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
