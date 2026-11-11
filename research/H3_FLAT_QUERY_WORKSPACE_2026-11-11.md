# HTRC Flat: query-window GPU workspaces (research gate, UNRUN)

Parent exact HEAD: `ed69065c4a736396c8fe7234e8b34af61c7f45df`
in `experiment/flat-gpu-kv-reuse-20261111`.
Experiment branch: `experiment/flat-gpu-query-workspace-20261111`.
Only model-independent user-captured QKV is exercised.

## Motivation and isolated variable

The previous full-Q 4+56 single-JSON result
`flat_gpu_kv_reuse_ab_20261111-230858.json`
proved cached-KV attention parity and **1.26778x** (4 native)
and **1.52632x** (56 logical proxy) *warmed-build-inclusive*
speedups. The remaining per-query prep path still allocates
G, VB, LZ, SG, rowmax SEL, topk IDX, EXCL, Tile partials and
Merge output for every 1024 Q rows. This stage tests whether
keeping **only their storage** alive across Q windows helps.

Both arms use the **same** per-group GPU KBF/VBF/K8/KS cache,
fused K8 and exact Tile/R32 tail Merge:
- `fresh`: allocates all query-dependent outputs afresh.
- `workspace`: preallocates G/VB/LZ/SG/SEL/IDX/EXCL,
  Tile PM/PL/ACC and Merge output once per group. The stats
  kernel **recomputes all G/VB/LZ/SG**, rowmax redoes SEL,
  torch.topk recomputes IDX and EXCL is explicitly zeroed
  before scatter on **every** Q window. Tile/Merge overwrite
  their output buffers every window. Q gather/MU, topk
  intermediate indices and tail gather are still fresh.

Same model-free captured GPU source, same logical heads, same
40 windows, same original pinned numerical reference and
independent full Flat oracle on first/middle/last windows,
same ABBA synchronized host wall, same group VRAM budget.
This is NOT a test of caching query-dependent results,
and not a direct comparison with the earlier uncached K/V run.

## Timing and memory accounting

Report separately:
- `sum_window_group_medians_fresh_ms` and
  `sum_window_group_medians_workspace_steady_ms`.
- **Identical** median warmed KV cache build cost added to
  both full-Q paths.
- One measured `query_workspace_setup_wall_ms` per head
  group added **only** to workspace candidate, for fair
  within-one-attention-call amortization.
- `speedup_steady`,
  `speedup_amortized_with_build` and cold-start sensitivity.
- Cache resident bytes, workspace live bytes, PyTorch total
  allocated peak, group planner cap and any OOM reduction.

The baseline executes with the unused workspace still resident
to preserve interleaved timing; thus absolute VRAM of `fresh`
is NOT standalone lower-bound memory usage. The planner adds
workspace memory to the existing KV+transient reservation.
When the group is too large, OOM halves it and retries.

## One PowerShell command, one JSON, live progress

```powershell
cd D:\ComfyNodes-flat-split-20261111
git fetch htrc
git switch --detach htrc/experiment/flat-gpu-query-workspace-20261111
$py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
& $py -m py_compile h3\htrc_flat_head_batch.py tools\benchmark_h3_flat_full_pipeline_batch.py tools\benchmark_h3_flat_gpu_query_workspace.py tools\test_h3_flat_gpu_query_workspace_contract.py
& $py -m unittest discover -s tools -p test_h3_flat_gpu_query_workspace_contract.py
& $py -B tools\benchmark_h3_flat_gpu_query_workspace.py --head-cases 4,56 --repeat-available-heads --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --rounds 2 --warmups 1 --cache-build-rounds 3
```

One persistent report:
`D:\ComfyOutput\H3\math_probe\flat_gpu_query_workspace_ab_*.json`.
Child reports are temporary and deleted. Both 4-head native
and 56-logical proxy cases are embedded in that single file.

**All new code GPU UNRUN.** CPU static unittest and Python
compile are also UNRUN in the connector environment; the
commands above form the validation gate. Do not call PASS
until uploaded report confirms all parity and memory.

## After this gate (do not stack speculative optimizations)

If query-workspace amortized latency and VRAM are positive,
retain the research variant and test CUDA Graph capture/replay
with stable allocations, plus warmed vs cold graph pool memory.

Next priority is **real model-resident H3**: acquire QKV capture
from 56 genuinely distinct model heads and run a full attention
call while H3 weights remain allocated on the RTX 5070 Ti 16GiB.
Compare original native attention, pinned Flat and candidate
separately (math error, throughput, GPU memory). Only then
measure full denoising steps, frames and video quality.
If most time is now in stats/rowmax/topk kernels, profile and
test the independent query-cache experiment, never merge it
without positive isolated evidence.

Global BEST and ComfyNodes production remain unchanged.
