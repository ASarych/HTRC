# HTRC: resource/time triage for the existing Materialized Flat Tile32

Date: 2026-11-14. Base branch:
`experiment/flat-memory-to-speed-balanced-20261113`, checked against
`39f2daaf5bb308d4b92bc8179e666bf351a31289` before this slice.

## Evidence already accepted

The uploaded `flat_materialized_ncu.json` passed all gates:
`PROFILE_COUNTERS_COLLECTED`, `ncu_exit_code=0`,
`strict_child_parity_marker=true`, **12 GPU Graph nodes**, four each of
Stats, Rowmax, Tile. On the pinned BF16 Balanced/512, 15,275-token,
four-head, Tile32 shape:

| Kernel | DRAM read for 4 nodes | DRAM write for 4 nodes | L2 TEX read sectors for 4 nodes |
|---|---:|---:|---:|
| Stats | 126,117,376 B | 123,636,992 B | 9,711,410 |
| Rowmax | 75,555,328 B | 0 B | 3,171,608 |
| Tile | 302,623,488 B | 32,213,760 B | 57,970,466 |

Tile was responsible for ~60% of the sampled DRAM reads and ~81.8% of
sampled L2 read sectors. **These figures do not measure elapsed time or
prove a DRAM bandwidth bottleneck.** Nsight Compute Graph-node replay
can change cache residency and launch behavior; the four kernels in each
stage are a bounded sample, not the entire attention call.

## Separate read-only next-step profiler

The new `tools/profile_h3_flat_resource_ncu.py` uses the original,
unchanged `tools/profile_h3_flat_materialized_ncu.py --child` for all
GPU work. This is **not a new Triton kernel**, not a production path,
and not a model or video test. It reuses the original pinned
`materialized`, exact eager/Graph first-call parity, independent
bank-B parity, five warmups, and NVTX-marked Graph replay.

* `gpu__time_duration.sum`: **mandatory**, kernel duration *under Nsight
  replay only*, not equivalent to unprofiled complete attention time.
* `sm__warps_active.avg.pct_of_peak_sustained_active`: **mandatory**,
  runtime achieved occupancy; report the fraction as a percentage.
* `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` and
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`: optional throughput
  context; no speedup may be inferred from their ratio.
* `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` and corresponding `st`:
  optional local-memory transaction evidence. These may come from spills
  **or** thread-private arrays, so positive sectors are not a proof of
  spilling and zero sectors do not by themselves prove perfect occupancy.
* Parse `launch__registers_per_thread`,
  `launch__shared_mem_per_block_allocated`, and occupancy limit attributes
  from **available** raw CSV launch metadata only. Never invent absent
  launch attributes or claim register pressure from a theoretical CTA
  footprint alone.

The tool queries installed Nsight Compute via `--query-metrics-mode all`
**before starting any GPU child**, selects only advertised metrics and
fails closed if duration or achieved occupancy is absent. It never
silently fills missing metric values with zeros. The German/Russian/US
localized numeric parser handles NBSP, the misdecoded Cyrillic grouping
character observed on this machine, and comma decimals. Metric units
come from the actual CSV units row; unrecognized time units invalidate
the timing gate. Every stage retains its independent samples, median,
spread and available launch metadata.

**Single output:**
`D:\ComfyOutput\H3\math_probe\flat_materialized_resource_ncu.json`

The old successful `flat_materialized_ncu.json` is **not overwritten**.
Only up to 12 Graph nodes are profiled by default. Nsight raw stdout is
consumed in memory; there is no separate CSV, NCU report, model file or
video output. A truncated diagnostic tail is included only on error or
for source inspection.

Use the Windows exact-HEAD runner
`tools/run_h3_materialized_resource_ncu_safe.ps1`. It checks ComfyUI and
Blender are stopped, fetches the pinned branch into a **disposable detached
checkout**, runs all CPU-only tests first, pins the existing
materialized/Tile32/Stats BT16 environment, and restores it afterwards.

## Interpretation and next decision

Compare the **individual** Stats/Rowmax/Tile Nsight replay durations,
achieved occupancy, launch registers/shared memory and local accesses.
If Tile time and local sectors are high with high registers and low
occupancy, investigate register pressure and spills in the existing Tile.
If occupancy is healthy but DRAM saturation is high, investigate data
movement/coalescing and near/far traffic. If both compute and memory
throughput are low, test for latency or instruction-dependency pressure
before making a memory optimization.

No optimization may be selected solely on this 12-node profiler. The
accepted old ABBA full-call median (Materialized ~4.285 ms at this
synthetic shape) stays the independent unprofiled control, not a
comparable duration to any sum of profiled nodes. A new candidate would
still need bit-exact two-bank output, isolated ABBA timing and independent
model-resident VRAM gate before real H3/video.

**At commit:** CPU-only unit tests must be executed and reported with exact
status; this document does NOT assert any GPU performance measurement.

References:
- NVIDIA Nsight Compute Profiling Guide:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- NVIDIA Nsight Compute Compute Triage Guide:
  https://docs.nvidia.com/nsight-compute/ComputeTriage/
- NVIDIA Nsight Compute CLI:
  https://docs.nvidia.com/nsight-compute/NsightComputeCli/
