# H3 Flat: full-Q GPU-resident K8 isolation, windowed and parity-gated

Date: 2026-11-11. **Research only.**
Source positive GPU-resident isolated-K8 branch:
`experiment/flat-gpu-resident-qkv-20261111@d5c312b01525edf716ef01acd23e720b06e44f9f`.
Current full-query experiment branch:
`experiment/flat-gpu-full-query-20261111`.

## Why

The repeated isolated GPU K8 test on RTX 5070 Ti has been verified at
16 query tiles per native head (16 x 64 = **1024 query rows**):
4 distinct heads: PyTorch 3.264ms vs Triton 2.062ms (**1.583x**).
56 **logical** heads (4 native captures repeated, groups 12+12+12+12+8):
sum of group medians PyTorch 39.749ms vs Triton 21.483ms (**1.850x**).
Both exact. But this is a **prefix** benchmark, not all Q tokens.

New opt-in `--query-coverage full` processes the saved full native Q
axis on GPU in fixed `--tiles` windows, with a final zero-padded
partial query window. The K/V tensor and exact tail remain unchanged.
Final Q padding is part of both candidates' bitwise oracle checks,
but the report labels valid Q rows separately.

- The same GPU-resident captured Q/K/V, per-head pinned numerical
  reference, torch_gpu vs triton_gpu grouping and paired ABBA apply.
- Every window/head group checks final-attention parity against
  pinned BEST for both candidates, and cross-candidate parity.
- The first, middle and final query windows also run four separate
  native-head independent full Flat oracles (first head group).
- Budgeted head groups and OOM halving are re-planned per query window.
- `--head-cases 4,56` still emits **exactly one persistent JSON**,
  `flat_gpu_quant_isolation_combined_*.json`, with two fully embedded
  case reports; temporary child JSON files are deleted.
- The default `--query-coverage prefix` code path is preserved.

### PowerShell: existing worktree, one GPU command and one final JSON

```powershell
cd D:\ComfyNodes-flat-split-20261111
git fetch htrc
git switch --detach htrc/experiment/flat-gpu-full-query-20261111
$py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
& $py -m py_compile tools\benchmark_h3_flat_gpu_quant_isolation.py
& $py -B tools\benchmark_h3_flat_gpu_quant_isolation.py --head-cases 4,56 --repeat-available-heads --query-coverage full --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 2 --warmups 1
```

The 56-head case is intentionally marked
`repeated_capture_proxy`: it is NOT a unique 56-model-head capture.
Captured H3 Q length may have a partial last window; report field
`query_total_valid_rows_per_head` is taken from actual capture,
not assumed. For a captured Q length of 40,759, 16-tile windows
produce 40 windows/head and 201 zero-padded query rows in the last
window.

### Measurement limitations and decision gate

This is a **full-query coverage** test, not a contiguous full-query
single Triton call and not a model denoise. All query-window group
median wall times are summed for the comparison. The implementation
**re-materializes K/V staging, K8 quantization, leaf statistics
and selection separately per query window** for both competitors.
An optimized real H3 runner may instead build/reuse key-only state
once per head group and stream queries; this A/B should be followed
by a K/V-state reuse experiment if full-Q timing proves expensive.
No model checkpoint, 56 distinct model heads, CUDA Graph, or video
quality/latency is exercised.

Technical validation: standalone full-Q window/padding/group-loop
logic received a mocked CPU smoke PASS, including a short final window.
Integrated Python imports, GPU execution, full Q parity and full-Q
speed **UNRUN** until user runs the command. Fail closed on any parity,
K8 exactness, capture shape or OOM error. No BEST promotion and no
production switch in this branch.
