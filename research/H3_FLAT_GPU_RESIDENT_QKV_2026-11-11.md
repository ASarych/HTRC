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
