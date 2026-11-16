# Full 56-head HTRC stage attribution — research-only v1

Date: 2026-11-16. Branch: `experiment/flat-memory-to-speed-balanced-20261113`.
Base source SHA: `4d03adf27d057e1d394f681b10add3c8b993036c`.

## Motivation and what is already measured

The fixed 24-block original Tile control experiment completed numerically,
but the last 12 block medians still varied by **6.552325%** (threshold 1.0%)
and the two late-cohort half-medians drifted **4.190213%** (threshold 0.5%).
That is `INCONCLUSIVE_PRECONDITION`, with **zero** far-VB `.cg` candidate
captures. Stop changing the threshold or retrying that microbenchmark.

The earlier 56-distinct-head full-attention synthetic Graph comparison
reported group-8 over legacy-11 improvements of **4.34% at 15,275** tokens
and **6.94% at 18,723** tokens, but validated only 64 sampled query rows,
not complete-output bitwise parity. The 28-head grouping experiment produced
little additional speed with far more reserved VRAM. A separate real H3 Graph
ON/OFF test recorded 64 vs 48 seconds for warmed six-step denoise, but did
not provide a complete per-stage breakdown. These are distinct comparisons;
do not add speed percentages or infer full video benefits from Tile tests.

## Research question

Where is time spent in a complete, **native 56 distinct-head** Flat invocation?
Reuse two *existing* measurement capabilities without introducing new math:

- `h3/htrc_flat_directflow_live.py::graph_attention_bthd`: capture, strict
  complete-output per-head eager/Graph equality on each of two independently
  seeded BF16 QKV input banks, and warmed full-call CUDA Event timing. The
  Graph path includes fresh K/V refresh + native K8 quant, Q staging, all
  windows (stats, selection, top-k, Tile, Merge) and output writes.
- `h3/htrc_flat_live_backend.py::flat_attention_bthd(observer=...)`: the
  **existing** eager stage GPU Event and CPU enqueue intervals for
  `kv_workspace_setup`, `query_stats_selection`, `tile`, `merge_tail`, and
  `output_copy`. This is **not** the distribution of time *inside* a CUDA
  Graph. The eager path can use a different head-group size, and the report
  always retains that actual group capacity. `query_stats_selection` also
  contains Q preparation and K/V-dependent state/TopK work; this first slice
  does **not** separately measure quantization within the captured Graph.

CUDA Event intervals include replay-queue gaps and observer overhead. CPU and
GPU intervals overlap and must **not** be summed to claim pure kernel time.
No kernel profiler/counters or RTX Nsight permissions are assumed. Do not
read stage timing as proof that a specific GPU kernel is compute/bandwidth
bound, and do not compare eager stage sums with warmed Graph totals as an
A/B speedup. This is stage *attribution* for finding the next hypothesis.

## Fixed single-shape experiment

- GPU NVIDIA RTX 5070 Ti SM120, 16 GiB, CUDA BF16.
- Distinct synthetic QKV (`torch.randn`), 56 heads, 15,275 tokens, D=128;
  bank seeds 42026 and 42027, **not** a four-head repeated proxy or a real
  H3 checkpoint. Inputs are identical across the Graph and eager comparison
  for each bank and never stale/reused across denoising steps.
- Balanced mode, 512 exact leaves, 16 tiles, materialized VB,
  Q native_torch_mu, K/V native_direct. Research `reuse_aware` group-cap 8;
  actual eight-head Graph group required, with 768 MiB post-capture
  extra-reserved guard and the existing allocator/free-VRAM limits.
- Two separate first-call Graph sessions (clear cache between banks), each
  invoking the **existing mandatory strict** `compare_only=False` eager/Graph
  parity for every contiguous head group, **all output rows**, not 64 sampled
  rows. Mismatches or missing report fields -> `FAIL_CLOSED` before timing.
- After second strict gate: five Graph warmups, then seven CUDA Event and
  synchronized wall samples of the *entire* 56-head invocation. Retain all
  raw samples and medians, report elapsed variability; no outlier exclusion
  or inferred kernel speedup. Reuse the same second QKV bank for an independent
  complete eager output check (`torch.equal`) and exactly one existing eager
  stage-observer recording. The observer measurement is **not** included in
  Graph timing; allocator peaks are noted as shared-process non-isolated.
- Explicit user confirmation of no other GPU workloads. Windows wrapper
  checks for listening ComfyUI on port 8188, Blender/Ollama processes and
  NVIDIA Comfy Python clients **before** launching the benchmark; no client
  processes, services, clocks, drivers or installed ComfyUI files are changed.

## Implementation and source protection

Files:

- `tools/benchmark_h3_56head_stage_attribution.py`: one report, exact-HEAD
  check before GPU imports, two-bank full-output parity, warmed Graph timing,
  existing eager observer, fail-closed errors. No new Triton code.
- `tools/test_h3_56head_stage_attribution_cpu.py`: admission rules, malformed
  observer rejection, wrong exact HEAD before CUDA import, source/runner
  contracts, CLI, metric distinction.
- `tools/run_h3_56head_stage_attribution_safe.ps1`: disposable detached
  exact source checkout, pre-GPU `py_compile` and source/CPU tests,
  single-JSON verification, clean temporary checkout.

Run **only once** with an exact source commit after its push:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_h3_56head_stage_attribution_safe.ps1 -ExactHead <EXACT_NEW_HEAD> -IClosedOtherGpuWorkloads
```

File: `D:\ComfyOutput\H3\math_probe\flat_56head_stage_attribution.json`.
If the first full-output parity fails or memory gate blocks the Graph, retain
`FAIL_CLOSED` and do not infer timing. If the full run completes, the
status is **`PASS_ATTRIBUTION_ONLY`**, not a speedup claim or production
admission. The next decision should depend on observed stage proportions and
unprofiled full-Graph duration, not another ungrounded Tile microvariant.

At authoring: new Python `py_compile` **PASS**; 10 self-contained CPU tests
**PASS**, 1 full-repository-only source test **SKIP** on the local
partial test tree. The target Windows full CPU suite, RTX GPU measurement,
full H3 denoise/video, and production integration are **UNRUN**.
`h3/` kernel/dispatcher/defaults and `main` remain unchanged.


## 2026-11-16 — completed full 56-head source-attribution experiment

User-uploaded `flat_56head_stage_attribution.json`,
**4,871 bytes**, SHA-256
`405908d7b751e18926c0aaf69ec1942bc714d0b700bc1e24b99221e4f3660309`,
tested exact source commit
`e87d135573a18b40cd6a27ae48b475ea6a1786fb`.
Raw report remains user-held. Status
`PASS_ATTRIBUTION_ONLY`, `completed=true`,
`PASS_TWO_BANK_FULL_OUTPUT_GRAPH_EAGER_BIT_EXACT`.
The independent source banks both checked ALL output rows from
56 distinct synthetic BF16 heads in seven 8-head groups.
Bank 0/1 both passed strict Graph/eager bit-exact output parity;
independent final whole-output parity also passed.

Warmed complete Graph call (QKV refresh, native K8 quant, all 15
query windows per head group, captured Graph kernels, Merge and
output writes), seven unfiltered measurements:

| Metric | Observed |
|---|---:|
| GPU Event median | **66.636764526 ms** |
| Synchronized host-wall median | **68.300000000 ms** |
| GPU Event max/min spread | **2.886776%** |
| First-call extra reserved (both banks) | **590 MiB** |
| Peak allocated (shared process) | 1045.299 MiB |
| Peak reserved (shared process) | 1430 MiB |

This synthetic complete-attention timing cannot be directly
substituted for H3 video denoising or compared as a matched
A/B with other report sessions. No real model/video was run.

**Separate existing EAGER stage observer (NOT Graph):**
the eager head planner selected capacity 13 and 15 windows.
Event interval sums (ms) were:
`kv_workspace_setup=95.848`,
`query_stats_selection=3313.251`,
`tile=1166.718`,
`merge_tail=4.152`,
`output_copy=2.879`.
The observed eager GPU event interval sum is 4582.849 ms;
query/selection contributes **72.3%** and Tile **25.5%** to
THIS event-interval sum, not to the Graph call time.
Eager phase GPU intervals include queue gaps, CPU enqueue and
instrumentation delays, so these are not pure GPU kernel costs.
Different group capacities (Graph 8 vs eager 13) also preclude
direct stage-time comparison. Do not target Graph query
selection merely because 3313 ms appears in the eager report.

**Conclusion:** the numerical and 56-head full-call baseline
are now measured. The missing evidence is the split between
K/V staging+quant, Q staging, captured per-window Graph work
(stats/TopK/Tile/Merge together), and output write-back ON
the Graph path itself. A new opt-in diagnostic records those
external Graph boundaries, without modifying `h3/` code,
and preserves the uninstrumented reference and full-output
bitwise result. See `H3_56HEAD_GRAPH_BOUNDARY_TRACE_2026-11-16.md`.
