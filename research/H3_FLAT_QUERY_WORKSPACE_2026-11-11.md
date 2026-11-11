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

## GPU result — single combined JSON, 2026-11-11 23:27:32

User-provided `flat_gpu_query_workspace_ab_20261111-232732.json`.
Candidate exact `a0234bca6fbf99f6af0058cea9c9cefa3903528d`
and pinned original `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.
RTX 5070 Ti (16 GiB), Q length 40,759/head, K aligned 40,752
plus 7 exact tail rows, 40 windows, 201 Q padding rows.

Top-level and both cases `completed=true`; `errors=[]`, no OOM.
All 520 group×window triples **bit-exact** against pinned and between
fresh/workspace (1560/1560 pairwise gates, max abs=0, rel-L2=0).
All 294 independent full-Flat-oracle comparisons PASS; all native
4+4 head K8 exactness gates PASS with zero byte and scale differences.

| Case | Fresh medians + common warmed KV build | Workspace medians + KV build + workspace setup | Amortized ratio | Steady ratio | Workspace live MiB |
|---|---:|---:|---:|---:|---:|
| 4 unique native heads (40 windows × H4) | 74.62075 ms | 72.75010 ms | **1.02571x** | 1.03601x | 71.904 MiB |
| 56 logical heads (40 windows × 12 groups, H5+H1) | 1104.45485 ms | 1068.09950 ms | **1.03404x** | 1.04777x | 89.880 MiB max |

H4 cache resident 100.114 MiB. H5 cache resident
125.143 MiB. H5 persistent workspace=89.880 MiB,
without including the existing 119.411 MiB GPU capture.
H4 peak allocated in ABBA with both paths' workspace resident:
fresh ~367.66 MiB vs reuse ~297.03 MiB. H5:
fresh ~429.15 MiB vs reuse ~340.98 MiB.
These are in-process peaks while the unused workspace remains
resident for *both* arms; they do not equal model-resident
end-to-end memory needs.

**Decision:** retain workspace only as an opt-in research feature,
useful for stable addresses/CUDA Graph and reducing allocations.
A 2.6–3.4% full-Q amortized speedup does not justify more
standalone allocator microbenchmarks. Next priority is the
**real model-resident, 56 distinct-native-head attention gate**
with true post-QKV/post-RoPE tensors and original attention
restored afterward. It must not be described as full denoising
or a video quality test.
