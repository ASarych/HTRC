# HTRC materialized Graph: first Nsight Compute node-counter gate

Date: 2026-11-14. Base:
experiment/flat-memory-to-speed-balanced-20261113
@ 30c785509bb003bb0940df799a72a61f2fea0ca1.
This is an opt-in synthetic profiler, NOT an H3/ComfyNodes
production change and NOT a kernel optimization.

## Evidence recovered

The user's sm120_transport_inventory.json returned
status=CAPABILITY_ONLY_NO_GPU_KERNEL (not a profiling PASS):

- NVIDIA GeForce RTX 5070 Ti, compute 12.0, 70 SM.
- 48 MiB reported L2 (50,331,648 bytes).
- SMEM: 49,152 B default/block, 101,376 B opt-in/block,
  102,400 B/SM. 65,536 regs/SM and 1,536 threads/SM.
- PyTorch 2.12.1+cu130, torch CUDA runtime 13.0,
  Triton 3.6.0.
- Nsight Compute 2026.1.1 ncu.BAT present at:
  C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.BAT
- GPU counter permissions and actual profile are UNKNOWN.
  Hardware TMA multicast remains NO_GO for GeForce RTX 50,
  and Triton SM120 multi-CTA is not verified.

Logical native BF16 K+V footprint, assuming four heads:
15,275 tokens ~29.834 MiB, 40,759 ~79.607 MiB,
82,000 ~160.156 MiB. At 82k this exceeds the device's
total L2 by >3.3x before G/VB/Q/output/model. However
the working set per kernel/split and residence/reuse
intervals matter more than full allocation volume;
this does NOT show the L2 hit rate or bandwidth bottleneck.

## NEW scope: one bounded profiler JSON, no other artifact

The new read-only tools/profile_h3_flat_materialized_ncu.py
launches Nsight Compute CLI on the SAME script in a
short-lived child process. It executes EXISTING
h3/htrc_flat_directflow_live.graph_attention_bthd
materialized mode, not a new CUDA kernel.

Pinned shape: 15,275 tokens, four distinct BF16
heads, Balanced/512 exact leaves, Tile32,
1024/256 MiB budget/reserve, STATS_BT=16,
legacy head grouping and unchanged 768-MiB extra
Graph allocation guard. It uses two Q/K/V banks.
The existing first-call full eager/Graph bit-exact
gate, independent bank-B eager parity and five
untimed full-attention replays run BEFORE any
profiled region. The profiler sees one warmed
full Graph attention invocation wrapped in the
HTRC_MATERIALIZED_REPLAY NVTX push/pop range.

Nsight CLI options:
--nvtx --nvtx-include HTRC_MATERIALIZED_REPLAY/
--graph-profiling node
--kernel-name regex:(_stats_kernel|_rowmax_kernel|_tile_kernel)
--launch-count 12
--csv --page raw --print-units base

Only a bounded subset of selected Graph kernel
nodes is collected, not every kernel in a
full H3 attention call. The explicit counters are:
dram__bytes_read.sum, dram__bytes_write.sum,
lts__t_sectors_srcunit_tex_op_read.sum and
launch__registers_per_thread.

The script parses stdout in memory and emits
ONE JSON at
D:\ComfyOutput\H3\math_probe\flat_materialized_ncu.json.
No .ncu-rep, .csv, model output or video is stored.
A profile status PROFILE_COUNTERS_COLLECTED requires
valid DRAM+L2 counters for at least one Stats AND one
Tile node and strict child parity marker.
PROFILE_PARTIAL, no counters, profiler errors,
permissions errors and CUDA mismatches fail closed.
The JSON contains the diagnostic output tail
for interpreting NCU failures; no fake zero-valued
measurements are inserted.

IMPORTANT: Nsight kernel replay may flush/change
caches and serialize launches. Counter results
from individual kernels are NOT full-call GPU
durations and do not match ordinary ABBA timing.
DRAM bytes and L2 sector counts concern distinct
memory paths: their ratio is NOT a measured
L2 hit rate. Even a successful 12-node sample
cannot establish representative long-context
physical traffic, L2 occupancy, CTA cluster
support or an optimization speedup.

The Windows exact-HEAD safety runner is
tools/run_h3_materialized_ncu_safe.ps1. It
uses a disposable detached Git checkout, checks
ComfyUI/Blender are closed, executes
tools/test_h3_materialized_ncu_cpu.py before GPU,
and restores the process environment after use.

If Nsight reports ERR_NVGPUCTRPERM, the JSON
records BLOCKED_PERFORMANCE_COUNTER_PERMISSIONS;
do not silently retry as administrator or alter
developer security settings. The user may
choose to enable counters in NVIDIA App,
System > Advanced > Developer > Manage GPU
Performance Counters, or use an elevated console.
Reference: https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute

Official CLI semantics:
https://docs.nvidia.com/nsight-compute/NsightComputeCli/
https://docs.nvidia.com/nsight-compute/ProfilingGuide/
Metric semantics and cache caveats:
https://docs.nvidia.com/nsight-compute/ComputeTriage/

## Status and next condition

New CPU contracts: UNRUN until executed on
Windows exact-HEAD or elsewhere.
New Nsight GPU profile: UNRUN.
Real H3 model, 40k/82k profiler data, video
and model-resident VRAM: UNRUN.
No CUDA/Triton, eager, Graph, fallback, quality,
precision or main branch modifications.

Only AFTER inspecting actual counters and
resource data should we decide whether L2
traffic optimization is worthwhile. If the
counters fail or sampled nodes are incomplete,
diagnose the profiler gate, not the math.

## 2026-11-14 — Windows ncu.BAT pipe-token FAIL_CLOSED diagnosis

User's local Windows run on exact HEAD
34ca7f0225c49a1e9e213c38fa4e775d40536a09:
CPU source contracts PASS; profiler JSON FAIL_CLOSED,
ncu_exit_code=255, strict_child_parity_marker=false,
completed=false. The final captured message starts
"_rowmax_kernel" followed by CP866-misdecoded Russian
Windows "not recognized as internal or external command".
This is strong evidence for cmd.exe interpreting literal
pipe characters from the old NCU kernel regex
regex:(_stats_kernel|_rowmax_kernel|_tile_kernel)
as Windows command pipelines. It is NOT a CUDA numerical
failure and no DRAM or L2 counters were collected.

Windows .BAT scripts may be executed through cmd.exe
even with subprocess.run(..., shell=False): Python docs
https://docs.python.org/3/library/subprocess.html#security-considerations
NVIDIA documents regex: substring matching for --kernel-name:
https://docs.nvidia.com/nsight-compute/NsightComputeCli/

Fix is limited to the research profiler:
- Replace the alternation regex with an equivalent safe
  character-class expression regex:_[srt][a-z]*_kernel
  covering _stats_kernel, _rowmax_kernel and _tile_kernel.
  The JSON parser still admits ONLY those exact three
  kernel families. The regex also avoids parentheses,
  pipes, redirection and other cmd.exe metacharacters.
- command() explicitly rejects accidental reintroduction
  of Windows cmd metacharacters in the NCU filter.
- Capture subprocess stdout/stderr as bytes. Decode
  UTF-8 first, Windows OEM fallback (typically cp866)
  for correctly legible Russian cmd.exe diagnostics.
- Add two CPU regression tests; retain all prior gates,
  same Graph node profile, same 12 max profiled launches,
  no new CUDA kernel, and unchanged production/math.

This is a Windows launch repair, NOT proof that Nsight
will collect valid hardware counters. CPU regression
tests and the new GPU profiling run are UNRUN until
executed; if the next run fails, inspect the one JSON
for the next distinct error rather than guessing.


## 2026-11-14 — Nsight real CSV obtained but 0 parsed launches

Second user-run at ec410f3be7bcd0342a00e6762734bcc122032e50
after fixing the Windows cmd pipe reported:

    status=FAIL_CLOSED_NO_VALID_COUNTERS_OR_PARITY
    ncu_exit_code=0
    strict_child_parity_marker=true
    completed=false
    profiled_launches=0
    has_stats_and_tile=false

The diagnostic tail contains genuine NVIDIA Nsight 2026.1.1
raw-page CSV lines for _rowmax_kernel, _tile_kernel and
_merge_tail_kernel. The CUDA Graph ran successfully and the
existing eager output parity marker passed. No performance
measurements have yet been VALIDATED because the old parser
rejected every counter row. Its mandatory DRAM metric names
are not necessarily valid on this SM120 card: newer SM120
documentation notes dram__bytes_op_read.sum and
dram__bytes_op_write.sum, whereas the old tool requested
the legacy dram__bytes_read.sum and dram__bytes_write.sum
only. Some local CSV values contain non-breaking-space
thousands separators misdecoded as Cyrillic a, and some
localized decimals use commas, e.g. 2a572a000 and 4,55.
The old float(value.replace(",", "")) is invalid for both.
The precise per-column cause cannot be established from the
last 6000 characters of the prior JSON because the CSV HEADER
was not retained there; do not claim one cause as proven.

Repair to the research-only tool:
1. Query exact installed NCU with --query-metrics-mode all
   BEFORE launching CUDA. Select verified, supported DRAM
   read/write metric names; prefer SM120 op_read/op_write,
   fall back to legacy only if present. Also query a supported
   L2 read-sector metric (TEX source preferred; broader
   all-read count explicitly labeled if used).
   If any mandatory family is unavailable, FAIL_CLOSED
   before launching the child process, and record the reason
   in the same JSON. No GPU counter value is synthesized.
2. Parse raw CSV with explicit metric-name columns and handle
   Russian grouping separators (NBSP or their CP866-decoded
   Cyrillic small a) and decimal comma. All three required
   counter values must parse as finite nonnegative INTEGERS.
   n/a, invalid or fractional values are rejected, never
   zeroed. Only real stats and Tile rows can satisfy the
   accepted profile; partial samples remain FAIL_CLOSED.
3. Narrow the safe regex to regex:_[srt][iot][a-z]*_kernel
   which covers stats/rowmax/tile and avoids accidentally
   matching _merge_tail_kernel through its _tail_kernel
   suffix; still no Windows cmd metacharacters.
4. Add compact JSON csv_diagnostics: total header columns,
   names of discovered DRAM/L2 metric columns, missing
   required columns and a count of kernel rows. This lets
   future parse failures be diagnosed without copying
   enormous Unicode-corrupted numeric rows.
5. Add CPU-only tests for SM120 and legacy metric discovery,
   raw wide CSV, localized integers, fail-closed preflight
   and the kernel filter.

External context:
- NVIDIA CLI exposes --query-metrics-mode all and says that
  an unsupported or mistyped metric may return n/a.
  https://docs.nvidia.com/nsight-compute/NsightComputeCli/
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- RTX 5070 SM120 practitioner evidence lists the op_read
  and op_write DRAM metric names for Nsight 2026.2.1;
  this is NOT confirmation on the user's installed 2026.1.1.
  https://github.com/Olajide-Badejo/GPU-Roofline-Profiler/blob/main/docs/profiling_guide.md

No change to Stats/Tile/graph math, exact selection, H3
model, fallback, CUDA kernels or production. Existing
sm120_transport_inventory.json remains CAPABILITY_ONLY,
not a GPU bandwidth result. Previous failed report is
retained as evidence (may be overwritten by next opt-in
test). New CPU and GPU acceptance tests UNRUN until
explicitly executed; no DRAM/L2 results claimed.


## 2026-11-14 — CPU test-only stale assertion repair

User's exact-HEAD Windows test run at
ef3e63f246ee50f979dc4fabfc4e72e75e53decd:
FAILED (failures=1); fail-closed PowerShell wrapper reports
"NCU CPU parser/source contracts failed; no GPU profile started".
The full traceback was not supplied; source inspection found
one guaranteed obsolete source-string assertion in
test_one_output_and_no_fake_success. The test was still searching
for the older completed=bool(parsed) and strict marker expression,
which the previous commit intentionally replaced with a stricter
condition:

    result["completed"] = bool(
        result["status"] == "PROFILE_COUNTERS_COLLECTED" and
        result["strict_child_parity_marker"]
    )

The source test now asserts this actual strict condition and
hardware_profile = completed, without changing the production,
profiling or CSV parser code. No evidence of a new CUDA failure.
The updated test suite and new Nsight run remain UNRUN until
explicitly executed; no GPU counter result is claimed.
