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


## 2026-11-14 — accepted resource/time counter result

Source: user-uploaded flat_materialized_resource_ncu.json.
Exact SHA-256:
233476a568b0a61f7c3d35428179b060ed909fba696047aaaff1bf25a5875dfc
Source git HEAD: 01996061cec72ba252a3634892c9c642ab90c78c.
The JSON bytes are NOT checked into HTRC; these derived facts
can be rechecked against the user-provided file.

All gates PASS:
status=PROFILE_RESOURCE_COUNTERS_COLLECTED, completed=true,
hardware_profile=true, metric_preflight=PASS_ADVERTISED_MANDATORY_METRICS,
ncu_exit_code=0, strict_child_parity_marker=true,
has_stats_and_tile=true, parsed_launches=12, timing unit ns,
no missing metric columns, rejected_required_metric_rows=0.
Four separate CUDA Graph kernel nodes each of Stats, Rowmax and Tile.
Synthetic input 15,275 tokens, 4 BF16 heads, balanced/512, Tile32,
STATS_BT=16, materialized. Real H3/video UNRUN.

| Measured value (median of 4 kernel nodes) | Stats | Rowmax | Tile |
|---|---:|---:|---:|
| Nsight-replayed kernel duration, microseconds | 91.920 | 31.568 | 422.960 |
| Achieved occupancy, % | 46.160 | 52.345 | 15.570 |
| Launch registers / thread | 80 | 70 | 255 |
| Launch allocated shared memory / CTA, bytes | 15,360 | 11,264 | 50,688 |
| Register-limited resident CTAs / SM | 6 | 7 | 2 |
| SMEM-limited resident CTAs / SM | 6 | 9 | 2 |
| DRAM throughput, % sustained elapsed peak | 69.97 | 68.41 | 21.73 |
| SM throughput, % sustained elapsed peak | 55.53 | 31.32 | 64.54 |
| Local-memory load sectors median | 0 | 0 | 0 |
| Local-memory store sectors median | 0 | 0 | 0 |

Individual Nsight kernel duration values (microseconds):
- Stats: 91.712, 93.376, 90.752, 92.128.
- Rowmax: 33.664, 31.200, 31.936, 31.136.
- Tile: 425.312, 422.592, 423.328, 420.928.

Interpretation from NVIDIA launch metadata plus the same-user
SM120 hardware inventory:
- 128 threads (4 warps) per Tile CTA, 255 registers/thread,
  50,688 bytes of shared memory/CTA.
- 65,536 registers/SM and 102,400 shared bytes/SM.
  Two resident Tile CTAs require 65,280 registers and
  101,376 shared bytes: both resources independently allow
  only 2 CTAs, and the measured launch limits confirm 2.
- 48 possible resident warps/SM (from device limit 1,536
  threads/SM); 2 * 4 = 8 warps, 16.67% theoretical warp
  occupancy. Measured achieved 15.57% is consistent.
  Resource pressure, not exclusively DRAM throughput,
  is a credible next optimization target.
- With 3 Tile CTAs at the SAME 128 threads/CTA the
  necessary resource ceilings (before allocator granularity)
  are at most 170 registers/thread AND at most
  floor(102400/3)=34,133 shared bytes/CTA.
  Merely reducing one of those will leave the other
  at the 2-CTA limit. Even achieving 3 CTAs gives
  no guaranteed speedup and may increase work or launches.

IMPORTANT no false bottleneck labels:
- Tile was ~4.60x Stats in the separately profiled
  kernel replay medians. Do NOT extrapolate this to a
  whole-attention latency ratio or compare it with the
  4.285-ms warmed unprofiled whole-call control.
- GPU DRAM saturation for Tile median 21.73%, versus
  SM throughput 64.54% and low warp residency. This
  weakens the hypothesis of a solely HBM-bandwidth-bound
  Tile, but does not establish that specific FP/INT/Tensor
  cores are saturated or classify warp stall reasons.
- Local-memory sector counters were exactly zero in all
  twelve samples. There is no positive measured local
  traffic in this profile, but absence of these sectors
  cannot prove no compiler spills in every specialization.
- The 2026-11-11 flattened 56-head Tile proxy had
  ALREADY reported 255 registers and about 49.7 KiB
  shared. Current data independently confirms the
  resource bottleneck for the NEW four-distinct-head
  balanced/512 materialized graph shape; don't claim
  first-ever discovery of 255 registers.

## Next engineering admission (no GPU candidate yet)

The current h3/htrc_flat_head_batch.launch_prepared_tile_batch
hardcodes Tile grid (heads*tiles,splits,1), RB=1 and
LATE_NEAR_Q=0. The underlying _tile_kernel already has
compile-time RB (query-row ownership) and LATE_NEAR_Q
(later production of near Q operands); kernel math is
not being rewritten here. Existing default near/far
block BN is 64 and PROD tile_stages=2.

A reasonable MINIMAL opt-in experiment, not yet implemented:
1. Preserve exact-head materialized control, unchanged
   Stats, Rowmax, topk, key quantization and Merge.
2. First separately examine compiler resources for
   LATE_NEAR_Q=1 (reduce far-loop Q live ranges)
   and tile_stages=1 (possible SMEM change).
   Test each separately before considering their
   combination, so the reason for any resource change
   is identifiable. These settings still require
   independent numerical admission: altered compiler
   scheduling can affect BF16/FP32 reduction.
3. Do NOT blindly force maxnreg=128 or treat it as
   increased occupancy; local spilling could be worse.
4. Only if individual candidates plausibly reach
   BOTH resource ceilings, or demonstrate another
   substantive register/stall improvement, admit an
   opt-in RB=2 row-ownership variant. RB changes
   CTA grid shape and can change reduction geometry.
5. For any candidate that reaches GPU testing: full
   outputs BIT-EXACT on two distinct QKV banks,
   first-call vs eager parity, independent-capture
   warmed four-block A/B/B/A timing with at least
   seven per-block rounds, no unrun-as-pass, no
   model/video before a reproducible micro win.
   Track real compiled register/SMEM/local sectors
   and allocator budget, not just theoretical counts.

No new GPU, register-pressure optimization kernel,
production switch, model, or video test was performed
in this evidence-closing documentation slice.
