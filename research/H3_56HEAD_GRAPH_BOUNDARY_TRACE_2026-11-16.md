# HTRC 56-head Graph boundary trace (research-only)

Date: 2026-11-16.
Branch: `experiment/flat-memory-to-speed-balanced-20261113`.
Base SHA:
`e87d135573a18b40cd6a27ae48b475ea6a1786fb`.

## Why a distinct stage attribution is necessary

Two independent 56-distinct-head BF16 synthetic QKV banks passed
full Graph/Eager output parity, including all query rows, on
RTX 5070 Ti. The uninstrumented 56-head Graph median was
**66.636765 ms** GPU Event, **68.30 ms** synchronized wall,
seven warmed samples, and **590 MiB** first-call added
reserved memory. The user-held source report
`flat_56head_stage_attribution.json` is 4,871 bytes,
SHA-256
`405908d7b751e18926c0aaf69ec1942bc714d0b700bc1e24b99221e4f3660309`.

The existing eager stage observer reported
`query_stats_selection=3313.251 ms` and
`tile=1166.718 ms` GPU Event intervals — but those
are *eager* with head cap 13 and significant CPU submission
gaps, not the **66.6 ms Graph**. Treating those values
as an internal Graph bottleneck would be a false inference.

## One opt-in post-parity Graph boundary experiment

No new Triton math, no Graph recapture or installed ComfyUI edits.
Only `tools/` research code changes. Keep the existing
`tools/benchmark_h3_56head_stage_attribution.py`
strict 2-bank all-output exactness, seven uninstrumented GPU
timings and independent final Graph/eager parity.

With `--graph-stage-trace`, immediately after normal
uninstrumented timings and before separate eager stage observer:
record ONE additional full Graph call on bank 1, under a
temporary `unittest.mock.patch.object(flow,'_replay',...)`
which is restored on every exit. This patch changes only
the Python dispatcher for that single measurement, not
captured kernels or Graph state; the trace follows
the existing original `_replay` statement order and
refuses if its source fingerprint changes.

Each of seven 8-head Graph groups has exactly
15 query windows and four coarse stage labels:

- `kv_refresh_quant`: copy current graph-owned K/V,
  native K8 quant and exact K/V tail refresh (combined).
- `q_stage`: copy/pad fresh Q into graph-owned window.
- `graph_replay`: execute the already captured
  stats/TopK/Tile/Merge Graph for one window.
- `output_write`: copy the current Graph output into
  the final per-head result.

Exactly **322** raw stream event intervals:
7 × (1 K/V + 15 Q + 15 Graph + 15 write).
Keep every record, group and window, the full instrumented
GPU Event/wall time, and stage CPU submission intervals.
A fresh traced output must be bit-exact to the
uninstrumented bank-1 full output; missing events or
shape drift fail closed. All elapsed times must be
finite positive.

**Caveats:** events between stages measure GPU-stream
intervals including host queue gaps and CUDA Event overhead;
they are NOT isolated kernel wall-time. The captured
Graph replay remains an **aggregate** of internal
stats, top-k, Tile and Merge kernels. No attribution
inside that Graph is claimed here. The trace collects
hundreds of events and may materially perturb dispatch;
report instrumented/uninstrumented full-time ratio,
flag `measurement_distorted_over_20pct` rather than
silently deleting samples, lowering precision thresholds
or turning instrumented intervals into accepted speedups.
The normal seven-call 66.6 ms baseline remains the
primary uninstrumented performance measurement.

## Code and target execution

- `tools/htrc_graph_stage_trace.py`: fail-closed
  source/order assertion, patch-scoped single-call
  tracing, full-output parity, summary and raw intervals.
- `tools/test_htrc_graph_stage_trace_cpu.py`:
  fake event completeness, distorted diagnostic label,
  numeric mismatch, invalid samples, source fingerprint,
  and patch restoration on exceptions.
- `tools/benchmark_h3_56head_stage_attribution.py`:
  new optional `--graph-stage-trace`; legacy default
  and its one-JSON report are retained.
- `tools/run_h3_56head_stage_attribution_safe.ps1`:
  new `-GraphStageTrace` switch; same strict CPU/GPU
  environment and exact HEAD preflight, with one
  *new filename* for the trace.
- No `h3/` files touched. No weights downloaded,
  no production dispatch, model or video inference.

PowerShell runner after fetching/pinning the new HEAD:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_h3_56head_stage_attribution_safe.ps1 -ExactHead <EXACT_HEAD> -IClosedOtherGpuWorkloads -GraphStageTrace
```

ONE JSON:
`D:\ComfyOutput\H3\math_probe\flat_56head_graph_stage_trace.json`.
It includes original-source baseline, strict two-bank
parity and separate eager stages for context plus the
new `graph_stage_trace` boundary result. Original
`flat_56head_stage_attribution.json` remains untouched.

Status is
`PASS_GRAPH_BOUNDARY_TRACE_DIAGNOSTIC_ONLY`, not
speedup admission. The trace is one candidate to
focus *subsequent* engineering; no new kernel
optimization can be declared based on its event percentages.

Authoring tests: the helper's seven CPU tests and
Python compilation passed locally. End-to-end Windows
source suite, GPU trace on 5070 Ti and model/video
remain **UNRUN** until user executes.

## 2026-11-16 — completed external Graph boundary trace

Source: user-uploaded `flat_56head_graph_stage_trace.json`,
**66,450 bytes**, SHA-256
`39a6145cd41e79b4f0a8cd9e90937244f63ce55ea08853eef8a206ec737e2d1e`,
tested exact HEAD `5745808a0a5ded3cb295bc82cfcb86456ddeceac`.
The raw source is user-held, **not** added to the repo.
`status=PASS_GRAPH_BOUNDARY_TRACE_DIAGNOSTIC_ONLY`,
`completed=true`,
`numeric_gate=PASS_TWO_BANK_FULL_OUTPUT_GRAPH_EAGER_BIT_EXACT`,
and instrumented output exactness also PASS.
Both independent BF16 QKV banks checked **all 56 heads and all
query rows**. Synthetic input, not a real model.

Seven normal whole-call Graph timings (group8, 15,275 tokens):
GPU Event median **66.690849304 ms**, wall median
**68.425200000 ms**, seven-sample event max/min spread
**0.91867567%**. First-call extra GPU reservation **590 MiB**
in both banks. Instrumented one-call GPU Event
**66.056831360 ms**, wall **72.1932 ms**,
`instrumented_over_unprofiled_gpu_ratio=0.990493`
and `measurement_distorted_over_20pct=false`.
The ratio is descriptive; a ~0.95% difference in separate runs
is NOT a speedup and is well within timing variability.

Exactly **322** raw boundary intervals, seven groups with
15 windows each, all window replays count **105**.
Summed CUDA Event intervals:

| Boundary | GPU Event ms | % of full instrumented GPU Event |
|---|---:|---:|
| `kv_refresh_quant` (7 groups) | 1.512512 | 2.290% |
| `q_stage` (105 windows) | 1.037216 | 1.570% |
| `graph_replay` (105 windows) | **62.477376** | **94.581%** |
| `output_write` (105 windows) | 0.806592 | 1.221% |
| Sum | 65.833696 | 99.662% |

Event intervals include CPU enqueue gaps; both CPU and GPU
timing sets overlap. Do not add CPU submit timings to GPU
intervals, or interpret these intervals as isolated GPU kernel
durations. No kernel counters were collected.

Average recorded Graph replay interval is **0.595023 ms**
(105 windows; median **0.594464 ms**, range
**0.568128–0.616960 ms**). It is comparable across all
seven head groups (per-group Graph totals roughly
8.887–8.990 ms), with a shorter final partial window
(index 14, mean **0.578834 ms**). This is the first
complete-path measurement identifying the correct
**Graph-internal** optimization target.

The separately observed eager 56-head stage intervals
in THIS run changed dramatically versus the previous
full-stage diagnostic:
`query_stats_selection=78.0344 ms`,
`tile=72.7735 ms`,
`kv_workspace_setup=16.1207 ms`,
`merge_tail=3.00675 ms`,
`output_copy=3.20458 ms`. Graph uses group8, eager group13;
neither eager sample is a valid internal Graph breakdown.
The earlier eager Q-selection interval of 3313.251 ms must
NOT be promoted as the Graph bottleneck.

**Decision:** close the external-boundary diagnostic; do
not redo staging, Q packing or output copies in search of
large 56-head speedups at this geometry. The next useful
research slice profiles **kernel activity inside** an already
captured, warmed, original Graph replay:
leaf stats, rowmax, topk if kernel names permit, Tile,
Merge, explicitly preserving unclassified activities.
No change to numerical math, group policy or production
defaults. Kernel name matching is a hypothesis, not
verified source-to-kernel linkage; no speed/admission claim
from a profiled run.
