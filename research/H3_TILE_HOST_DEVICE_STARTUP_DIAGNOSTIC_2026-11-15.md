# HTRC: original Tile startup wall/host submission/GPU duration diagnostic

Date: 2026-11-15. Branch:
`experiment/flat-memory-to-speed-balanced-20261113`.
Build on `4f4cafd2d904c6705f4942b9b0ecd0d2ddd0a086`.

## Reproduced evidence and correction
Two complete user-provided reports measured the same original Tile32 Graph on
RTX 5070 Ti. Both passed exact original two-bank BF16 QKV, mask fingerprint,
compiled resources (255 registers, 49,664 B CTA, no spills), and original
PM/PL/PACC/Merge raw-bit gates. Both failed only in the first quiet phase;
later unprofiled block medians were nearly constant.

| Report | Exact source HEAD | Quiet-before spread | Monitored spread | Quiet-after spread | Status |
|---|---|---:|---:|---:|---|
| `flat_tile_quiet_environment.json`, SHA-256 `8e925c3dfbf66e548ea227f09b6f266623cf62abc2a732e944b62d245a481cfc` | `0d548f6c0cb4c912efc5f31853726c20021cab05` | 3.41062049% | 1.33489099% | **0.18070350%** | Unstable |
| `flat_tile_quiet_environment(1).json`, SHA-256 `197a1e449628afd389a58f67955b3e1fae07cac136b010a5826863c0f8102b2f` | `4f4cafd2d904c6705f4942b9b0ecd0d2ddd0a086` | **7.95696279%** | 2.56692562% | **0.12091714%** | Unstable |

Second raw JSON: 45,322 bytes, 90 samples, one 256-replay original
Graph per sample. 6 startup quiet block medians (microseconds):
396.822, 425.967, 428.397, 428.024, 408.521, 412.182.
Second phase 6 block medians: 399.143, 408.779, 408.886,
409.100, 409.224, 409.389. Final six:
409.687, 409.630, 409.730, 409.904, 410.125, 410.048.
Quiet-before vs quiet-after phase medians differ 2.25881860%.
No raw samples or outliers removed; pre-registered <=3% gate still FAIL.

**Correction of previous external-GPU-client interpretation:**
`nvidia-smi --query-compute-apps` is called **inside** the running
GPU benchmark, after the process itself has imported CUDA and launched
the source Graph. The listed Comfy Desktop `standalone-env/python.exe`
could be the benchmark's own Python interpreter, since the runner uses
the Python environment installed by Comfy Desktop. Its presence in the
before/after report snapshots does **not** establish a competing
ComfyUI service or interference. The first run listed PID 19608,
second listed PID 19020; neither report recorded `os.getpid()` to
match them definitively. The new test does.

## New experiment: source attribution, no candidate
Only the existing original source Tile32 and its persistent CUDA Graph.
The original two-bank, resource and full PM/PL/PACC/Merge gates run
first. No research kernel, alternative math, model or video execution.

Pre-registered measurement schedule (all same persistent Graph):
1. `startup_wall`: 12 blocks x 5 samples x 256 Graph replays.
   Host `perf_counter_ns` wall time; separately record the Python
   replay-enqueue span and completion wait. **No CUDA Events.**
2. `paired_host_gpu`: 12 blocks x 5 samples x 256 Graph replays.
   Reuse a fixed start/stop CUDA Event pair across samples;
   record GPU elapsed time and host wall/enqueue/completion spans.
3. `settled_wall`: 6 blocks x 5 samples x 256 Graph replays.
   Again NO CUDA Events; check whether late uninstrumented wall
   measurements remain stable.

The first/last six startup blocks are pre-defined cohorts; no
retrospective outlier removal or threshold relaxation.
Original output partials and Merge are verified **after every phase**.
Capture the process's own PID and `sys.executable`, alongside
read-only GPU-client snapshots, so the self-process can be
distinguished from another client in later interpretation.
Preserve the exact source commit, all raw samples, 3% descriptive
spreads and every check in ONE report.

**Measurement caveats:** host enqueue/completion durations are
overlapped with device execution; neither is a pure independent
contribution to kernel latency. CUDA Events introduce their own
measurement overhead; compare trends within the instrumented phase,
not absolute GPU Event timing against prior unprofiled runs.
Even synchronized wall/event differences cannot by themselves prove
thermal, power or scheduler causality. If both host and GPU event
metrics drift, further controlled tests are needed. If GPU event
duration is stable while enqueue varies, host launch jitter is
plausible, not proven.

A completed diagnostic has
`status=RECORDED_ORIGINAL_TILE_HOST_DEVICE_DIAGNOSTIC_ONLY`
regardless of whether startup/paired/final spreads are above 3%.
This is not a speed PASS. No promotion of far-VB `.cg`, mask-G,
GPU Event baseline admission or H3 model/video inference.

## One Windows command and one JSON
Run from the existing source checkout after closing other GPU workloads:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_h3_tile_host_device_attribution_safe.ps1 -ExactHead <NEW_EXACT_HEAD> -IClosedOtherGpuWorkloads
```

The wrapper fetches the research branch into a disposable clone,
checks the exact HEAD and CPU contracts, and launches one GPU
diagnostic only on the RTX 5070 Ti. Output:
`D:\ComfyOutput\H3\math_probe\flat_tile_host_device_attribution.json`.

Authoring status: CPU **UNRUN**, GPU **UNRUN**, full attention/model/video
**UNRUN**. No `h3/` or production code modified.
