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
