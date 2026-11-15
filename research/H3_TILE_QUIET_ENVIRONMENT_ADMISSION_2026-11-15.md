# Original Tile quiet/unprofiled wall-time environment diagnostic

Date: 2026-11-15. Continuation of exact
`experiment/flat-memory-to-speed-balanced-20261113@6f0433d759b8b5e7a91c7a8f9fb48e76e1b660a2`.
This file specifies a **new diagnostic**, not a new Tile kernel or an
admission to production.

## Why
The preceding two-bank original-only fresh/sentinel GPU Event report
`flat_tile_control_stability.json` passed bit-exact original Tile/Merge
but failed the pinned <=3% stability gate: fresh capture median spread
15.4758%; same-Graph sentinel medians spread 25.008%; 47/84 raw GPU
Event samples deviated >3% from their own seven-sample median.
Neither far-VB .cg nor mask-G has an admitted >=5% speedup.
This diagnostic must not turn those negative findings into a PASS.

## Deliberate isolation
- **Only** the original full Tile is used (Balanced/512, 15,275 BF16
  tokens, four heads, one 2,048-query window, original 11-token tail
  merged only for correctness), not a research candidate.
- Before timing, repeat existing two-bank original Graph/eager,
  EXCL/IDX source-mask, exact detached Tile+Merge, and compiler-resource
  gates (255 regs/thread, 49,664 B/CTA, zero spills).
- Capture one persistent original CUDA Graph **once** and run that same
  Graph for all phases. No fresh captures, CUDA Events, Nsight,
  clocks/power changes, or competing research kernels.
- Require explicit operator confirmation that other GPU workloads
  were closed; Windows runner separately checks ComfyUI port 8188,
  Blender and Ollama. Snapshot NVIDIA compute clients before/after.
  **Not proof of GPU exclusivity**: WDDM graphics contexts may be absent.
- Three fixed phases, in order: QUIET BEFORE, MONITORED, QUIET AFTER.
  Each phase = six blocks x five samples x 256 Graph replays;
  each sample synchronizes the GPU and records host monotonic
  start/end timestamps, wall ms/replay, every raw sample and spread.
  Wall time includes CPU Python replay submission and CUDA sync:
  **it is not pure GPU kernel execution time** and is not comparable
  to the preceding CUDA Event timings.
- During **MONITORED only**, spawn external `nvidia-smi` read-only
  sampling every 200 ms; retain timestamp, SM/memory clock, power,
  temperature, GPU/memory utilization, P-state and host receipt
  monotonic timestamp. Start the reader before timing and stop it
  after; external polling is itself a possible source of disturbance.
  Other GPU-client data are pre/post snapshots; neither can establish
  continuous absence of other workloads.
- Recheck exact original PM/PL/PACC and Merge raw bits after each phase;
  any mismatch, missing telemetry (<5 samples), compiler change or
  wrong exact HEAD is **FAIL_CLOSED** with the partial single JSON.

## Interpretation
A separate CPU decision reports within-phase wall-time block-median
spread, quiet-before/after median drift, and whether monitored-only
or both monitored/quiet phases vary. A <=3% label is **only** about
this host-clock diagnostic under this particular state; it is not a
GPU Event stability PASS, not an H3 full attention benchmark, and not
a candidate speed/quality PASS. Correlation of telemetry with wall
times does not by itself establish root cause. Raw samples are never
discarded, and a slow sample is never removed post hoc.

If the unprofiled quiet segments still drift >3%, investigate
competing GPU/CPU activity, power/clock behavior, launch or sync
overhead; do not start new kernel A/B trials. If quiet segments
are stable but monitored drifts, investigate observation overhead
before trusting fine-grained telemetry. A clean wall result would
justify a **new** CUDA Event baseline stability test with
pre-registered admission criteria, not production promotion.

## Exact Windows command (single JSON)
From the existing source checkout; **close other GPU jobs first**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_h3_tile_quiet_environment_safe.ps1 -ExactHead <EXACT_NEW_HEAD> -IClosedOtherGpuWorkloads
```

Output: `D:\ComfyOutput\H3\math_probe\flat_tile_quiet_environment.json`
(the prior `flat_tile_control_stability.json` remains intact).
The safe wrapper makes a disposable exact-SHA clone, runs the
existing original numeric CPU suites and this probe's CPU tests,
runs the GPU diagnostic only after they pass, validates the
single report, and deletes the temporary clone.

Source: `tools/benchmark_h3_tile_quiet_environment.py`,
`tools/test_h3_tile_quiet_environment_cpu.py`,
`tools/run_h3_tile_quiet_environment_safe.ps1`.

At authoring: static code review only; CPU tests = **UNRUN** on the
target Windows runtime; GPU wall timing and telemetry = **UNRUN**.
H3 model/video = **UNRUN**, full attention candidate speed = **UNRUN**,
promotion = **PROHIBITED**. No `h3/` or production source modified.


## 2026-11-15 — uploaded quiet/monitor/quiet result: first quiet phase unstable

Evidence: user-uploaded original `flat_tile_quiet_environment.json`,
**45,312 bytes**, SHA-256
`8e925c3dfbf66e548ea227f09b6f266623cf62abc2a732e944b62d245a481cfc`,
measured on exact source HEAD
`0d548f6c0cb4c912efc5f31853726c20021cab05`.
The raw JSON is not committed here; its source fingerprint and results are
recorded without changing the acceptance threshold.

Report: `schema=htrc-tile-quiet-environment-v1`,
`completed=true`, `numeric_gate=PASS_TWO_BANK_ORIGINAL_TILE_MERGE_BIT_EXACT`,
`status=WALL_CONTROL_UNSTABLE_DIAGNOSTIC_ONLY`,
`classification=QUIET_PHASE_VARIATION_OR_INTER_PHASE_DRIFT`.
Both distinct original BF16 QKV banks passed bit-exact native Graph/Tile+Merge
and independent eager equivalence. Sentinel initial, all phase-end PM/PL/PACC,
and original Merge raw-bit parity PASS. Original resources: 255 regs/thread,
49,664 B/CTA, no spills. No new candidate and no CUDA Events.

| Phase | 6 block-median values (ms / replay) | Phase median (us) | Max/min spread |
|---|---|---:|---:|
| Quiet before | .409501, .419146, .410844, .409188, .409313, .423144 | 410.173 | **3.41062049% FAIL** |
| Monitored | .404586, .409383, .409520, .409709, .409880, .409987 | 409.614 | 1.33489099% |
| Quiet after | .410110, .410073, .410219, .410397, .410548, .410814 | 410.308 | 0.18070350% |

Quiet pre/post median drift **0.03299868%**. Within each phase there were
30 raw wall-clock samples (six blocks of five, 256 graph replays/sample),
for **90 original-only samples**. Counts of raw samples >3% from their
own block medians: quiet-before **3**, monitored **0**, quiet-after **1**.
Quiet-before block 1 spanned **385.374–419.850 us** per replay; block 6
reached **431.068 us**. These outliers are retained; no retrospective
sample exclusion or altered threshold.

The monitored phase yielded **19** nvidia-smi records; SM clocks
**2662–2947 MHz**, temperature **59.33–75.09 C**, power
**107.27–257.77 W**, and GPU utilization 0–100%.
Sampling is coarse and receipt timestamps are not exact driver-event
timestamps. Do **NOT** conclude nvidia-smi improves stability or that
thermal/power changes, scheduling, GPU contention or Graph capture caused
the first-phase failures. These are hypotheses only.

**Important environmental admission gap:** pre/post nvidia-smi
compute-client snapshots both include a distinct long-running
`C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\standalone-env\python.exe`
(PID **19608** in this report), although ComfyUI port 8188 was not
listening. Epic Games Launcher, Epic web helper and
EOSOverlayRenderer were also listed along with browser, desktop and
other contexts; WDDM reports GPU-client presence, not their actual
kernel activity during measured samples. This is a plausible
contamination risk, **not proven interference**.

**Next and only admitted test:** close remaining Comfy Desktop
standalone Python and other optional GPU-using apps (Epic launcher
and overlay especially); re-run the **same three-phase control-only
protocol once** on the new exact source commit, without changing
warmup counts, 3% limit, or graph geometry. The runner now fails
closed before clone/CPU/GPU if nvidia-smi still lists the specific
Comfy Python context; it does not terminate processes. The runner
also warns about Epic clients but does not make their presence an
automatic cause claim. Compare the new raw block series to the
preserved original report, not just a PASS/FAIL label.

If quiet-before remains unstable, do not promote candidate A/B;
investigate host submission/synchronization jitter and workload/clock
state with a separate pre-registered controlled test. Even a stable
wall-only rerun would **not** establish CUDA Event baseline stability,
full attention speed, model/video quality or production admission.

Documentation and Windows preflight update only: no `h3/` math,
Tile CUDA kernel, Graph runtime or production code changed.
New GPU rerun **UNRUN**.


## 2026-11-15 — second, stricter-preflight quiet/monitor/quiet rerun

User report `flat_tile_quiet_environment(1).json`, 45,322 bytes,
SHA-256 `197a1e449628afd389a58f67955b3e1fae07cac136b010a5826863c0f8102b2f`,
exact source HEAD `4f4cafd2d904c6705f4942b9b0ecd0d2ddd0a086`.
Original two-bank numeric, source masks, compiler resource envelope,
persistent sentinel and all phase-end PM/PL/PACC/Merge raw-bit gates PASS.
Candidate speed, model/video and production promotion remain UNRUN/PROHIBITED.

| Phase | Median (us) | Block-median spread | Result |
|---|---:|---:|---|
| Quiet before | 419.074 | **7.95696279%** | FAIL |
| Monitored | 408.993 | 2.56692562% | <=3% |
| Quiet after | 409.817 | **0.12091714%** | <=3% |

Quiet pre/post median drift 2.25881860%. All 90 raw samples are
retained without exclusion. The previous run's quiet-after spread
was 0.18070350%; the new run again shows late stability despite
the unstable first quiet phase.

**Correction:** this document's earlier inference that the
`standalone-env/python.exe` entry was a likely *separate* ComfyUI
GPU client was unwarranted. The compute-client snapshots are taken
INSIDE the benchmark after its own Python process has initialized
CUDA. The observed Python PID may well belong to the benchmark itself,
which uses Comfy Desktop's Python installation. Neither run records
its own os.getpid(), so neither proves competing CUDA work.
The preflight check performed before launching the Python benchmark
passed. Do not reinterpret this client line as evidence of
external interference. The follow-up host/GPU attribution diagnostic
explicitly records its own PID, preserving independent attribution.

**Decision:** same original control numerics PASS, stability FAIL.
Do not rerun identical quiet/monitor/quiet measurements again to
seek a passing score. Next: pre-registered startup wall and paired
host enqueue vs GPU Event timing using the original sentinel Graph
under `H3_TILE_HOST_DEVICE_STARTUP_DIAGNOSTIC_2026-11-15.md`.
That is distinct from a candidate optimization or speed admission.
