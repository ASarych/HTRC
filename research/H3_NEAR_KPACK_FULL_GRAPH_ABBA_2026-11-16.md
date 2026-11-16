# Near K8+FP32 scale pack — full 56-head Graph A/B/B/A gate

Date: 2026-11-16.
Repository: `ASarych/HTRC`.
Branch: `experiment/flat-memory-to-speed-balanced-20261113`.
Base: `a01929baefd165a03207d88f5661f731917b5356`.

## Motivation from actual GPU measurements

On RTX 5070 Ti, 15,275-token Balanced/512, 56
independent BF16 synthetic heads:
- Warm full Graph GPU-event median **66.667778015 ms**.
- Corrected Torch profiler CUDA activity total **66.075230 ms**
  (the enclosing `record_function` was not a kernel).
- Tile **41.456084 ms** / 62.74% and Stats **15.082391 ms**
  / 22.83% of recorded activity.
- External Graph replay intervals 62.477376 ms / 105 windows.
- First-call extra Graph reservation roughly 590 MiB,
  non-isolated from the shared allocator and unrelated to
  real checkpoint-resident VRAM.

The preceding 4-head, 32-centre
`flat_near_kpack_numeric.json`
(5,348 B, SHA-256
`331200a6ffd863608e62d51dd5bb44408d637d4794065e254c498d8d856d1fab`,
source `a01929baefd165a03207d88f5661f731917b5356`)
passed both independent BF16 QKV banks with ZERO raw
PM/PL/PACC differences and ZERO Merge differences.
Actual allocation for original-ordered near K8 and
FP32 scales was 132 MiB; source vs candidate
compiler metadata 255 regs/49,664 B shared/zero spills.
No speed result was produced.

This gate tests whether contiguous near K8 addressing
can outweigh the cost of repacking for each 56-head
Graph call. **Hypothesis ONLY; no inferred speedup.**

## Geometry and shared-scratch constraint

Full 56-head Graph uses 7 head groups x 8 distinct
heads and 15 ordered windows (14 full and one final).
Its window geometries are Tile16/1024 rows and
Tile15/939 valid rows, while the original two-bank
4-head numerical gate was Tile32/2048 rows.

Allocate **one** external contiguous scratch for
each captured 8-head *session* before graph capture:
8 heads x 16 maximum centres x 8192 selected near
tokens x (128 K8 bytes + 4 scale bytes) = 132 MiB.

The same scratch is reused sequentially, and in
strict stream order, by the 16-centre and 15-centre
captured Graphs. It is retained via the live session,
not allocated again inside capture. A separate K8
packing kernel reads each unsorted original topk IDX,
copies the selected exact original K8 bytes and FP32
KS bits, and runs **inside** Graph after stats/topk.
BF16 V remains in the original unmodified K/V cache
and is addressed with the original IDX leaf/token
mapping. Far G/LZ/VB/EXCL, 4 splits, online softmax
order, Merge and tail are unchanged.

For the research pseudo-head layout, each query
centre is one pseudo-head for contiguous K8.
The native BF16 V pointer must still index its
real 8-head source: `t // REAL_TILES_PER_HEAD`,
with constexpr 16 or 15 depending on geometry.
The existing 4-head numerical gate defaults to
`REAL_TILES_PER_HEAD=32`.
The exact source-clone CPU gate verifies five
specific changes against canonical production
`h3/htrc_flat_triton.py`; it is a closed
allowlist, not permission to change Tile math.

All 105 windows per full Graph call execute the
K8+scale packing again. This means logical
K8+scale scratch *writes* of up to
**105 x 132 MiB = 13.54 GiB per complete 56-head call**
(a conservative upper bound because the final
window uses only 15 centres). The corresponding
source reads, extra launch overhead and memory
transactions may overwhelm any benefit. Do not
compare isolated Tile timings to an end-to-end
Graph speedup. There is a credible NO_GO outcome.

The separate first-call 590+132=722 MiB estimate
has just 46 MiB nominal margin under the
**unchanged 768 MiB research hard-extra-reserved
cap**. Allocator fragmentation, private graph pools
and other live tensors are not measured by adding
these two numbers; the REAL post-capture reserved
and minimum-free memory gates in the existing HTRC
Graph runtime remain authoritative. If rejected,
return FAIL_CLOSED and do not time. No cap increase,
VRAM offload workaround or promotion is allowed.

## Single exact-source GPU experiment

Tool: `tools/benchmark_h3_near_kpack_graph_abba.py`.
Runner: `tools/run_h3_near_kpack_graph_abba_safe.ps1`.
One persistent JSON:
`D:\ComfyOutput\H3\math_probe\flat_56head_near_kpack_graph_abba.json`.

1. Pin exact branch HEAD before Torch import.
   Close ComfyUI, Ollama, Blender and other GPU loads;
   disposable detached checkout and CPU source tests.
2. Generate two different, reproducible 56-distinct-head
   BF16 QKV banks (15,275 tokens), and obtain a separate
   **original eager** full-output bank-B oracle.
3. Untimed original control capture: first-call Graph-vs-eager
   exactness across all seven 8-head groups; complete bank B
   Graph must bit-match independent eager reference.
   Keep CPU copies of complete bank-A/B controls.
4. Untimed candidate capture: only wrap the existing
   `flow._one_window` function in a temporary research
   launcher injection. The default eager reference
   **must not be patched**. Preserve original stats/topk
   and Merge code. Require first-call 56-head
   Graph-vs-original-eager bit parity, complete bank-A/B
   output raw BF16 parity, and on BOTH input banks
   original-vs-candidate raw FP32 PM/PL/PACC and
   original Merge parity for the last eight heads'
   complete 1024-row window. If this fails, STOP
   BEFORE any timed control or candidate runs.
   Record actual compiler spill, register, shared
   metadata for original vs candidate and fail on
   an increased resource envelope.
5. After numerical admission, four independently
   captured complete 56-head Graph sessions in
   CONTROL / PACKED / PACKED / CONTROL order.
   Each captures once, checks mandatory original
   eager parity for all 56 heads, checks bank-B
   complete output and candidate two-bank partials
   (when candidate), then five fresh warmup replays
   and **seven unfiltered CUDA Event + synced-wall
   samples** per capture. Timing contains K/V refresh,
   K8 quantization, near K8 pack inside captured
   windows, Stats/TopK, Tile, Merge and output copy.
   Every capture has a post-timing bank-B
   full-output bitwise check.
6. Require controls AND packed capture medians each
   spread <=3%. A speed signal requires BOTH >=5%
   median-of-block-median reduction AND >=5%
   worst-pair reduction, plus both candidates faster
   than both independent controls. If unstable, label
   `INCONCLUSIVE_CAPTURE_DRIFT` and do not
   promote/retry to chase a positive result. If
   below 5%, `NO_GO_FULL_GRAPH` despite numeric PASS.
7. Preserve source HEAD, every raw GPU/wall
   timing sample, both bank numerical checks,
   compiler resource metadata, shared scratch
   size, allocator snapshots with limitations,
   and an explicit final A/B decision in ONE JSON.
   Do not generate large trace/tensor files or
   run model/video at this stage.

A positive result is only
`PASS_NUMERIC_PROMISING_FULL_GRAPH`,
**research-only** and does not justify promotion to
production: model-resident VRAM, video quality,
long-context stability, and 16 GiB checkpoint
fit are independently UNRUN. A negative result
closes this expensive per-window K8 reindexing
strategy without re-running a weaker standalone
Tile benchmark.

## Implementation files

- `tools/htrc_near_kpack_triton.py`: original
  research Tile with dynamic real-head V indexing
  `REAL_TILES_PER_HEAD`; no change to production.
- `tools/htrc_near_kpack_graph_adapter.py`:
  a context-scoped patch only around original
  `flow._one_window`; one stable scratch for both
  geometries and original graph-capture ordering.
- `tools/benchmark_h3_near_kpack_graph_abba.py`:
  strict preflight plus complete-Graph four-capture
  ABBA and one atomic JSON.
- `tools/test_h3_near_kpack_graph_abba_cpu.py`:
  ABBA math, worst-pair, unstable controls, temporary
  patch restoration, source and runner contracts.
- `tools/run_h3_near_kpack_graph_abba_safe.ps1`:
  exact SHA checkout, CPU tests, live GPU preflight,
  fixed round count and fail-closed report schema.
- `tools/test_h3_near_kpack_numeric_cpu.py`:
  updated canonical source-clone allowlist for
  the explicit real-tiles constexpr.
- `research/H3_NEAR_KPACK_NUMERIC_GATE_2026-11-16.md`:
  record verified original numerical experiment.

At this authoring step, full new Windows CPU suite,
all CUDA captures/A/B and actual model/video performance
are **UNRUN**. Pure source/contract reviews do NOT count
as executed Windows tests. No runtime `h3/` source or
production dispatcher is modified.

## 2026-11-16 — completed RTX 5070 Ti independent-capture ABBA: **NO_GO / CLOSED**

User-provided final `flat_56head_near_kpack_graph_abba.json` is **19,691 bytes**, SHA-256
`af9e99fee7cdf53554ccc19b645246e780ad3da651377c3652c7d1594f90e42d`.
Exact executed source **`d47c846b3f396cdefe2f686eb52b4950bdf4a93c`**.
Raw JSON is user-held and is not committed or overwritten.
The experiment reports `PASS_NUMERIC_NO_GO_FULL_GRAPH`, `completed=true`,
`PASS_TWO_BANK_FULL_56HEAD_AND_RAW_PM_PL_PACC_BIT_EXACT`, and full
candidate timing `MEASURED_FULL_56HEAD_GRAPH_INCLUDES_PACK`.
Both independent synthetic 56-distinct-head BF16 QKV banks pass raw
full-output Graph/eager parity. PM/PL/PACC and Merge are raw-bit-identical
in the two 8-head full-window checks, and full bank-B output remains
bit-exact after each timed capture. This is not model/video evidence.

Four **independent captures**, each with five post-capture warmups and
**all seven** unfiltered full-attention GPU Event samples, in
CONTROL / PACKED / PACKED / CONTROL order:

| Arm | Capture | GPU Event median (ms) | Within-capture max/min spread | Wall median (ms) |
|---|---:|---:|---:|---:|
| Original control | 1 | **67.246658** | 2.705% | 68.9745 |
| Near K8+scale packed | 2 | **92.351936** | 2.762% | 94.2890 |
| Near K8+scale packed | 3 | **92.393631** | 2.417% | 94.0138 |
| Original control | 4 | **67.051102** | 0.904% | 68.6300 |

Control capture-median median **67.148880005 ms**; packed capture-median
median **92.372783661 ms**. Candidate is **25.223904 ms slower** per
complete 56-head Graph call, or **37.564146% slower** than original.
The worst-pair reduction is **-37.795843%** (negative = slower).
Between-capture control medians vary only **0.291653%**, packed medians
**0.045148%**, both below the 3% stability gate. Every timed block's
within-block spread is also below 3%. This is a stable, repeatable
negative result, not a timing-precondition failure. The pre-registered
>=5% median and worst-pair speed improvement is **NOT met**.

The K8 and scale scratch is one stable **138,412,032-byte (132 MiB)**
buffer per 8-head Graph session, shared by both captured window geometries
(Tile15 and Tile16). It is repopulated in each captured window, so its
packing cost is included in full-call timing. Source says
`captured_tile_launches=12` per session during construction/warmup,
not the number of replays measured in the full call. The difference
between packed and original full-call times is a **net** cost of packing
plus changed Tile access, not a separately measured packing kernel time.
No isolated kernel attribution is inferred from this +25.22-ms difference.

The *first-call* reported added Graph reserved memory increases from
**590 MiB original** to **738 MiB packed** (+148 MiB), within the
unmodified 768-MiB research gate but leaving only **30 MiB** margin.
The shared-process `reserved_after_timing_mib` snapshots are
2,270 MiB (original) vs 2,578 MiB (packed); these snapshots are NOT
isolated model-resident/private Graph-pool VRAM measurements and
must not be treated as a +308-MiB intrinsic cost.
CUDA H3 checkpoint, actual denoising/video, 16-GiB model residency,
and end-to-end quality are **UNRUN**.

**FINAL DECISION: NO_GO_FULL_GRAPH / CLOSED.** Do not port K8+scale
near packing to `h3/` production, do not increase reserved-memory
limits, do not seek a positive result by retuning sample count or
repeating isolated Tile measurements. Keep research source and raw
numeric gate for historical reproducibility. The next distinct
approach should reduce *actual* work or materialized state without
adding 132 MiB of per-window repacking. Compare existing split-stream
VB / fused-VB candidates first; they already exist in HTRC and their
GPU numeric/performance admission must be recovered, **not assumed PASS**.

All earlier sections describing the ABBA as `UNRUN` represent the
state at authoring time. They are superseded for this exact source and
run by the completed outcome above.
