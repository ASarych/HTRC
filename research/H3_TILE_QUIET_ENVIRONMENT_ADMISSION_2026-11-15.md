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
