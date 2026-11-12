# Direct QKV native-strided kernel slice (2026-11-12)

Base: `b810f2e1de1a0f0d4efac600ed7c6990cc13c408` plus previous native-Q kernel slice `a45d54c104fb5d622ee699257f8cdc4554e39a12`.
Branch: `experiment/flat-direct-qkv-kernels-20261112`. Research only, opt-in, no production promotion.

## Implemented

- `h3/htrc_flat_triton.py`: opt-in `_direct_q_mu_kernel` for BF16 Q source with physical strides and correct final partial-row mean, `_quant_key_int8_native` for K8/KS without full KBF staging (preserves round-nearest-even PTX), plus opt-in strided-Q addressing in `_rowmax_kernel`, `_tile_kernel` and tail Merge; opt-in BF16 K/V in `_stats_kernel` and V in near Tile.
- `h3/htrc_flat_head_batch.py`: `query_staging="native_direct"` alongside `gather` and `view_copy`; `prepare_gpu_head_kv_cache(..., native_direct=True)`. Native K/V is currently **BF16 only**; other input dtypes fail explicitly. Requires unique consecutive head positions. Source QKV views remain owned by the attention call.
- `h3/htrc_flat_live_backend.py`: orthogonal `query_staging` and `kv_staging` flags, both default to gather. Other attention math/leaf geometry/topk/EXCL unchanged.
- Direct K/V retains packed INT8 K8 and scale KS; the full BF16 KBF/VBF allocations are replaced by strided source views, and alias bytes are excluded from persistent-cache accounting.
- No graph capture or persistent Direct-Flow dispatch is implemented in this slice; they are future gates building on the same strided kernel interfaces.

## Four independent benchmark arms

`baseline` staged Q and K/V; `direct_q` native Q only; `direct_kv` native K/V only; `direct_qkv` both.

`tools/benchmark_h3_flat_direct_qkv_ab.py` uses nonzero synthetic native BF16 inputs, 4 and 56 genuinely unique synthetic heads, full Q windows, a final partial query window and exact partial K tail. For every group: require direct vs packed K8 byte/scale parity; require Q values exactly equal, same IDX/EXCL, bounded numerical differences for MU/G/VB/LZ/SG, and final Tile+Merge outputs in first/middle/last windows. An ERROR or divergence saves one JSON and fails closed. Timings are synchronized GPU events for full prep+Tile+Merge; cache creation cost is reported separately and may contain cold JIT effects.

The synthetic test does not check original H3 model or visual/temporal quality. GPU benchmark **UNRUN**, CPU `py_compile` / unittest **UNRUN** on the authoring side. Default live production behavior unchanged.

## One Windows PowerShell command

```powershell
cd D:\ComfyNodes-flat-split-20261111
git fetch htrc
git switch --detach htrc/experiment/flat-direct-qkv-kernels-20261112
$py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
& $py -m py_compile h3\htrc_flat_triton.py h3\htrc_flat_head_batch.py h3\htrc_flat_live_backend.py tools\benchmark_h3_flat_direct_qkv_ab.py tools\test_h3_flat_direct_contract.py
if ($LASTEXITCODE -ne 0) { throw "Compile FAILED" }
& $py -m unittest discover -s tools -p test_h3_flat_direct_contract.py
if ($LASTEXITCODE -ne 0) { throw "Contract test FAILED" }
& $py -B tools\benchmark_h3_flat_direct_qkv_ab.py --head-cases 4,56 --rounds 2 --warmups 1
if ($LASTEXITCODE -ne 0) { throw "Direct QKV parity or performance gate FAILED" }
```

Report: `D:\ComfyOutput\H3\math_probe\flat_direct_qkv_ab.json`.

## Follow-up status

- `view_copy`: numeric PASS / perf NO-GO (previous uploaded A/B).
- `direct_q`, `direct_kv`, `direct_qkv`: implementation prepared, **GPU UNRUN**.
- `direct_flow` (CUDA Graph/persistent orchestration): **NOT_IMPLEMENTED / UNRUN**.
- Real 56 distinct model heads/denoise/video/VRAM quality: **UNRUN**.

One experiment at a time, no quality coefficient, no leaf-count/centre-group changes.
