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

## Full-Q GPU result: completed and bit-exact on RTX 5070 Ti

User-run `flat_gpu_quant_isolation_combined_20261111-222800.json`
from exact candidate `bbc5d1ecf632dfd1c24c94975169ebaa06fc9e78`
and pinned BEST `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.
Combined `--head-cases 4,56 --query-coverage full` produced ONE
persistent JSON with `completed: true`, `errors: []` and no
OOM retries in either child case.

Captured Q length is **40,759** rows per head, processed in
**40 windows** of 16x64 rows (last valid Q range
`[39936,40759)`, padded 201 Q rows). Coverage is complete.

| Case | Windows × head groups | Torch K8 sum median wall | Triton K8 sum median wall | Gain | H12/4-group max incremental peak |
|---|---:|---:|---:|---:|---:|
| 4 distinct native heads | 40 × 1 = 40 | 179.444 ms | 103.692 ms | **1.73054x** | 262.552 → 163.695 MiB |
| 56 logical heads, 4 repeated | 40 × 5 = 200 | 1564.081 ms | 848.830 ms | **1.84263x** | 785.218 → 491.485 MiB |

Across all 240 window-group combinations, the three pairwise
parity checks `torch_vs_pinned`, `triton_vs_pinned`,
`torch_vs_triton` have **rel-L2=0, max-abs=0**, all PASS.
Independent full Flat oracles for raw heads 1/16/27/55 PASS on
windows 0, 20, 39 in both cases. All four K8 exact gates PASS
in each case (40,752 rows/head, 0 INT8 byte mismatches and
zero maximum absolute scale difference).

**Research decision:** full-query window coverage and fused K8
gain PASS. These are *sums of group-window medians* over repeated
K/V preparation, not whole-model latency. There is no model
checkpoint, CUDA Graph, 56 distinct native heads or video
quality validation. No global BEST/production promotion.

## Streaming console fix (after this measurement)

The previous combined runner used `subprocess.run(capture_output=True)`
and only printed the child process output *after* a head case
finished. This explains the apparent hang despite a successful
JSON. Now the parent uses `subprocess.Popen` and forwards each
`WINDOW`/case-progress line immediately (with the same one
persistent JSON result, temporary child files, parity and kernel
math unchanged). **GPU re-run of the console-only change UNRUN**;
the uploaded 2026-11-11 full-Q result applies to the previous
mathematically identical candidate.

Next most meaningful gate: prepare/reuse K/V-side state across
query windows, then compare isolated cache reuse against this
full-Q baseline.
