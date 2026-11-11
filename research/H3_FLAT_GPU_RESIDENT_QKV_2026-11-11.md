# H3 Flat: GPU-resident QKV full-path A/B (experimental, UNRUN)

Date: 2026-11-11. Repository: ASarych/HTRC.
Parent: `8d8ae6c79ab9c7ed1050cf7ede92dd8efe300ff3` (batched experiment; inherited from original BEST at
`a8fe43e7aab8d9f951af7b4cba8fbd54f7a61e5c`).
Original numerical BEST is ComfyNodes `e47cc7a310936547ab6c7d97bd07526eb6a135fe`,
byte-identical to HTRC a8fe for pinned source files.

## Why this is needed

The previously tested fused Triton key quantizer is exact but the CPU-QKV
full-path candidate loses: 545.401 vs 509.192 ms (sum of five group
medians), +7.11% time and +38.38 MiB peak per H12. K/V CPU gather+H2D
rather than K8 quant dominates. But H3 runtime normally generates QKV on
GPU: benchmark CPU offload and GPU-native workloads **separately**.
Do not subtract PCIe costs from one path alone: both pinned per-head
control and batched candidate must consume the **same resident QKV**.

## Change

- Opt-in `--input-residency gpu` preloads the captured Q/K/V tensor
  arrays to CUDA once **outside** the synchronized A/B measurement.
  Default `cpu` keeps the old CPU-mmap + H2D benchmark unchanged.
- Grouped index-select indices are created on the QKV source device;
  this allows both CPU and CUDA gather while retaining exact
  selection order and repeated-proxy semantics.
- All Q/K/V must share a device; GPU QKV with `*_cpu` input-cast
  modes is rejected to avoid silently changing the measurement.
- Report includes `input_residency`, `capture_gpu_resident_mib`,
  accurate CPU-vs-GPU gather diagnostic names, and separate filename
  `flat_full_prep_gpu_qkv_ab_*.json`.
- The same first-group native full-Flat oracle, final parity on every
  group, synchronized wall ABBA, per-group incremental peak, current
  free-VRAM head planner, and OOM retries apply.
- Preloaded capture VRAM is excluded from *incremental group peak* but
  **persists** and is counted in free VRAM at planning time. There is
  no checkpoint/model, CUDA Graph, full-Q or full denoise in this gate.

## Run locally on RTX 5070 Ti (Windows PowerShell)

```powershell
cd D:\HTRC
git fetch origin
git switch --detach origin/experiment/flat-gpu-resident-qkv-20261111
$py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
& $py -m py_compile h3\htrc_flat_head_batch.py tools\benchmark_h3_flat_full_pipeline_batch.py
& $py -m unittest discover -s tools -p test_h3_flat_head_batch_plan.py
& $py -B tools\benchmark_h3_flat_full_pipeline_batch.py --heads 4 --input-residency gpu --quant-mode triton_gpu --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 4 --warmups 1
& $py -B tools\benchmark_h3_flat_full_pipeline_batch.py --heads 56 --repeat-available-heads --input-residency gpu --quant-mode triton_gpu --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 4 --warmups 1
```

The 4-head run uses distinct native captures; the 56-head run is
explicitly **only a repeated-head proxy**, not actual model validation.
Return both JSON reports. Also compare `--quant-mode torch_gpu` if
quantization gain needs to be isolated; keep source residency identical.
Do not combine a GPU-input result with the previous CPU-input result to
claim a speedup. No production change or promotion until accuracy,
full-operator performance, VRAM and eventually model-resident tests PASS.

Tests: CPU py_compile/unittest **UNRUN** in this commit; GPU validation
**UNRUN** (CUDA runner only available on the user's system).

## Next gate: fused K8 contribution without batching confound

A **new paired backend-only** script was added at
`tools/benchmark_h3_flat_gpu_quant_isolation.py` in
`f2f90fead46890d6a66523bd85f4bf746c60dded`.

Unlike the first GPU-resident experiment (pinned individual heads
versus grouped+fused K8), this directly measures **two grouped paths**
differing **only** in the quantizer: `torch_gpu` versus
`triton_gpu`. GPU QKV is loaded once outside both timed paths.
The pinned BEST and independent original Flat are used for **output
parity checks only**, not included in this timing. ABBA within each
group; exact BF16->INT8/scale gate on **all distinct captured native
heads** before timing; arbitrary positive head count and VRAM planner.

Run in the existing HTRC-branch worktree on RTX 5070 Ti:

```powershell
cd D:\ComfyNodes-flat-split-20261111
git fetch htrc
git switch --detach htrc/experiment/flat-gpu-resident-qkv-20261111
$py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
& $py -m py_compile tools\benchmark_h3_flat_gpu_quant_isolation.py
& $py -B tools\benchmark_h3_flat_gpu_quant_isolation.py --head-cases 4,56 --repeat-available-heads --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 4 --warmups 1
```

Since `3ef7f1c6e887c508a5921b0bdced42dfb5c14dd0`,
`--head-cases 4,56` runs both cases with one user command, bundles
the full results and a summary in **one persistent**
`flat_gpu_quant_isolation_combined_*.json` file. Each child case
executes separately to avoid GPU process contamination; its intermediate
JSON lives only in a temporary folder and is removed. Original
`--heads N` single-case mode remains supported. The 56-head case
still repeats only four native heads.

## Measured results: isolated GPU fused K8 vs PyTorch K8

Reports received 2026-11-11, both running candidate exact SHA
`4bb9d6bcb807235334f4c6a7bf459117dc820cdc`:
`flat_gpu_quant_isolation_ab_20261111-215819.json` (4 distinct
heads) and `flat_gpu_quant_isolation_ab_20261111-215827.json`
(56 logical repeated heads). These prior two user-run reports
were generated before the combined JSON output option existed;
their numerical kernel path is unchanged by the aggregator.

| GPU resident batched QKV | Torch K8 full median wall | Fused Triton K8 full median wall | Speedup | Peak incremental VRAM Torch -> Triton |
|---|---:|---:|---:|---:|
| 4 distinct heads (one group) | 3.22845 ms | 2.04115 ms | **1.58168x** | 262.552 -> 163.695 MiB |
| 56 logical heads (sum five groups' medians) | 38.95105 ms | 21.39640 ms | **1.82045x** | 785.218 -> 491.485 MiB (max group H12) |

Four captured raw heads 1/16/27/55; exact gate for each
40,752 K rows: zero INT8 mismatches, zero scale abs difference.
All grouped results exact to pinned reference and torch/triton
pairwise; all four independent full-Flat oracles PASS.
Both runs complete, with no errors or OOM.

The isolated backend gain is primarily in preparation; Tile/Merge
math and batching are identical between these two measured paths.
No inference latency claim: 16 tiles/head = 1024 Q rows,
not full Q; 56 logical head jobs reuse four captured native heads.
All tests were on a GPU-resident capture without an H3 checkpoint,
CUDA Graph, or denoise/video. Thus **research winner**, not
production or global BEST switch.

Next research priorities: full-query coverage, 56 genuinely
unique head capture, and model-resident VRAM. The separate
query-cache branch remains unmerged.

## Combined 4+56 single-JSON rerun: PASS (2026-11-11)

User-supplied source `flat_gpu_quant_isolation_combined_20261111-221657.json`.
Its exact candidate is `9bfe07a5fe514967214eb04897401ec6377958b2`;
pinned BEST `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.
The new `--head-cases 4,56` output protocol and subprocess isolation
were successfully exercised; top-level `completed: true`,
`errors: []`. Both embedded cases completed. No intermediate
case JSON needs to be shared.

| GPU-resident batched test | PyTorch K8 (ms) | Fused Triton K8 (ms) | Speedup | Peak incremental VRAM Torch -> Triton |
|---|---:|---:|---:|---:|
| 4 distinct native heads, one group | 3.26400 | 2.06170 | **1.58316x** | 262.552 -> 163.695 MiB |
| 56 logical repeated heads, 5 groups | 39.74870 | 21.48310 | **1.85023x** | 785.218 -> 491.485 MiB (max H12 group) |

4 K8 exact gates on 40,752 rows each: 0 INT8 byte mismatches,
FP32 scale maximum abs difference 0 for heads 1/16/27/55
in **each** case. All GPU output comparisons
(`torch_vs_pinned`, `triton_vs_pinned`, `torch_vs_triton`)
PASS with rel-L2=0 and max-abs=0 for all 6 groups combined;
independent full Flat oracles on 4 native heads PASS in each case.
No OOM retries. Compared to earlier separate 4-head/56-head
reports, results are directionally consistent; modest timing
variation is expected across executions.

The single-JSON **reporting contract is verified**, as is the
fused K8 performance gain at 16 tiles/head. Restrictions remain:
56 jobs only four distinct heads; 1024 Q rows/head, not full query;
checkpoint-free synthetic repeated-head workload; no CUDA Graph
or actual denoise/video. Retain fused K8 as research winner,
**do not promote to global or production BEST** without
model-resident and full-Q gates.
