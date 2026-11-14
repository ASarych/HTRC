# HTRC / SM120: CTA-cluster, TMA and L2 transport admission

Date: 2026-11-14. Starting HEAD:
experiment/flat-memory-to-speed-balanced-20261113
@ e4234b73a8d53d74c9bcda673faa2fc037e6d567.
Research only; no production kernel, ComfyNodes, Graph math,
window selection, BF16 precision, or H3 model changes.

## Verified external capabilities (NOT a local GPU experiment)

1. NVIDIA's official GPU compute-capability table lists the
   GeForce RTX 5070 Ti at **SM120 / CC 12.0**:
   https://developer.nvidia.com/cuda/gpus
2. The NVIDIA CUTLASS SM120 GeForce example explicitly states
   "Multicast feature of TMA load" is unsupported on
   GeForce RTX 50 and fixes its ClusterShape to **1x1x1**:
   https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
   The example mentions cluster launch control for scheduling;
   that does not establish TMA multicast or a multi-CTA DSM
   consumer on SM120.
3. The CUDA Programming Guide describes thread block clusters,
   distributed shared memory and TMA as supported for newer
   compute capabilities. These are DISTINCT from hardware
   TMA multicast and from an installed Triton compiler
   supporting a particular lowering path:
   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
4. Triton issue #10963 (closed 2026-07-20) reported that
   num_ctas>1 was accepted for SM120 while its compiler's
   cluster feature check rejected SM12. Triton issue #10973
   (OPEN as of this review) corrects the premise: SM120
   PTX can represent cluster ops, but Triton lowering
   rejects specific two-CTA async shared-store modules.
   The latter supersedes the earlier rejection proposal.
   https://github.com/triton-lang/triton/issues/10963
   https://github.com/triton-lang/triton/issues/10973

**Admission decision:** Hardware TMA *multicast* as the
HTRC V distribution mechanism is NO_GO on this specific
GeForce RTX 5070 Ti. Do not assume all CTA clustering is
absent: PTX-level support and this practical compiler/kernel
capability are separate statements. Do not use the fact
that Triton accepts num_ctas=2 as a capability PASS.
There is no measured CUDA inter-CTA transport result.

## Minimal read-only Windows inventory

tools/probe_h3_sm120_transport.py imports Torch/Triton from
the installed Python environment and records GPU name, CC,
available device properties (SM count and supported
shared-memory limits if exposed), Torch/CUDA/Triton versions,
and whether ncu is on PATH. It does not allocate an
attention buffer, compile or execute any Triton/CUDA kernel,
measure L2 traffic, capture a Graph, or load H3.
The source classification fail-closes unknown devices.
It emits ONE JSON for an exact-HEAD run:

D:\ComfyOutput\H3\math_probe\sm120_transport_inventory.json

If the local Triton version differs from the upstream issue
environment, the issue is still a research warning, NOT a
locally observed compile result. The probe's
status=CAPABILITY_ONLY_NO_GPU_KERNEL is not PASS_CUDA,
PASS_CLUSTER, or PASS_MULTICAST. The four CPU source tests
are in tools/test_h3_sm120_transport.py.

## Cache-local alternative and optimistic break-even

CUDA devices CC>=8.0 support L2 cache access-policy
windows and cache residency *hints*, not guaranteed
persistence. Triton's tl.load can specify .ca/.cg and
evict_first/evict_last; actual lowering and effect depend on
the installed version and load form. They must be tested as
opt-in control knobs, not as mathematical optimization:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
https://triton-lang.org/main/python-api/generated/triton.language.load.html

The existing source-cost model
tools/estimate_h3_cooperative_kv.py now reports one
**optimistic necessary** cache hit fraction for the
*extra repeated native V or K/V sources* if CTA clusters
cannot multicast. It counts only modeled K/V reads and
VB write/read. It assumes a completely uncached 1024 B
reference per centre-leaf, uncached candidate fixed
traffic, and that only additional loads might hit L2.
The baseline itself can hit L2; full attention also pays
for G/LZ/SG/SEL, top-k, Graph replay and tensor-core work.

For B=8 centres and R=4 query-row groups on long
context, the worst leaf-union bound U=1 gives:

  reference materialized: 1024 bytes/pair
  staged FP32 weights: 576 fixed + 2048 variable bytes/pair
  direct K/V replay:    256 fixed + 4096 variable bytes/pair

To get source-only HBM traffic *below* the optimistically
uncached reference:

  staged W: (1024 - 576)/2048 = 21.875% maximum misses
  => >78.125% additional V reuse needed for strict gain.
  direct:   (1024 - 256)/4096 = 18.750% maximum misses
  => >81.250% additional K/V reuse needed for strict gain.

The mathematical equal-break-even thresholds are 78.125%
and 81.25%; strict less-than requires slightly higher hit.
These are NOT observed cache hit rates, performance
predictions, or GPU occupancy results. The B=8/R=1 staged-W
case has a better modeled byte balance but keeps 256 KiB
logical FP32 accumulators in one CTA, so still lacks a
credible resource-admitted design.

## Next viable experiment gate

Do not implement cluster TMA multicast, nor another
B=2/4/8 naive in-Tile recompute variant.

First establish physical memory bottleneck in the
EXISTING warmed materialized Tile32 code using available
profiling support (Nsight Compute if installed), without
model/video and without altering numerical logic:
* Capture actual DRAM vs L2 read traffic at the stats,
  Tile far, and Q/selection phases; report both.
* Record registers per thread, shared memory, spills,
  active warps/SM and launch count where profiler supports.
* Keep exact same 15,275/40,759 attention geometry and
  original 512 selected near leaves; never infer 82k
  results from 15k without a separate budget gate.
* Use one final JSON if a new standalone profiler wrapper
  is ever implemented; external profiler output must
  never be reported as PASS without actual data.
* Only if enough reference/candidate traffic reaches HBM
  and CTA resource estimates admit B/R, design an opt-in
  L2-local consumer and compare full-output numeric parity
  and strict ABBA, plus model-resident memory afterward.

L2 hint syntax is supported by documented Triton but
whether a particular cache policy improves HTRC is UNRUN.
A capability-inventory JSON is not a profiler trace.
There is no fresh H3/video run, cluster compilation,
multicast runtime test, L2 hit-rate measurement or
TMA unicast kernel here.

**Final status:** TMA multicast NO_GO; multi-CTA Triton
lowering UNVERIFIED/KNOWN UPSTREAM ISSUE; L2 source
optimization REQUIRES REAL PROFILER EVIDENCE; working
materialized Tile32/STATS_BT16/eager fallback unchanged.
