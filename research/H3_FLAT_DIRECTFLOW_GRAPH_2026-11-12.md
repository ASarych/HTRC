# HTRC Direct-Flow launch architecture — static-window CUDA Graph feasibility

Date: 2026-11-12. Branch \`experiment/flat-directflow-graph-20261112\`.
Base: \`c7cf60befe464dec5ec183ccd993f05ace425b1e\`.

## What this slice DOES

- Research-only probe: identical **Direct Q + Direct BF16 K/V** query preparation, leaf Stats, Rowmax/TopK/EXCL, Tile and exact Merge/Tail, executed once as ordinary eager CUDA launches and once with \`torch.cuda.graph(graph)\` capture + replay.
- Query mode \`native_torch_mu\` or \`native_direct\` (no staged FP32 Q). K/V native BF16 and exact K8/KS held within one invocation. Source Q/K/V and all workspace addresses remain **STATIC** for replays.
- CUDA Graph warmup uses >=3 rounds on a separate side stream, no GPU->CPU synchronization during capture. Cache built before graph capture; no H2D tensor-pos allocation on native-Q path.
- Run one 11-head group for the first full and last short query windows (or override with CLI). Short window geometry is computed from actual remaining query rows, tile and center group. No fixed five-tile rule.
- Capture starts from fixed source pointers; eager and graph use **separate** workspaces and compare IDX, EXCL, MU and final output bit-for-bit **before** accepting timings.
- One JSON reports GPU event medians, CPU submit and sync wall medians, capture wall time, extra graph reserved VRAM and amortization break-even estimate. Graph memory reserve limit is fail-closed (default 512 MiB).
- No changes to live/prod dispatch, math, leaves, numerical tolerances, model weights or Comfy loader.

## What this slice DOES NOT prove

- **No real model/video loaded**. Graph captured from fixed synthetic BF16 QKV; new post-RoPE tensors every block/step will have different addresses. This capture cannot simply be reused on different addresses.
- Graph capture itself takes time and owns private GPU allocations. One graph per model block or attention invocation may be slower and exceed 16GiB budget. Replay speed must not be interpreted as end-to-end H3 improvement.
- A future Direct-Flow design needs **stable allocation/descriptor contract** across layer and step inputs, or must explicitly measure input copies into persistent buffers and cache lifetime. No host-to-device transfers or changing QKV pointers allowed by this proof.
- It does NOT implement persistent producer/consumer scheduling, TensorRT, or any new mathematical attention kernel.

Official CUDA Graph constraints and private pool guidance:
- https://docs.pytorch.org/docs/stable/notes/cuda
- https://docs.nvidia.com/dl-cuda-graph/latest/torch-cuda-graph/torch-integration.html

## Windows PowerShell run (one JSON)

Use the same proven ComfyUI interpreter \`$py\` as prior successful RTX Direct QKV tests:

\`\`\`powershell
cd D:\ComfyNodes-flat-split-20261111
git fetch htrc
git switch --detach htrc/experiment/flat-directflow-graph-20261112
& $py -m unittest discover -s tools -p test_h3_flat_directflow_graph_contract.py
if ($LASTEXITCODE -ne 0) { throw "Static Direct-Flow contract FAILED" }
& $py -B tools\benchmark_h3_flat_directflow_graph_ab.py --group-heads 11 --window both --rounds 25 --warmups 4
\`\`\`

Output: \`D:\ComfyOutput\H3\math_probe\flat_directflow_graph_ab.json\`.
If capture fails or graph storage exceeds memory cap, save this same JSON: \`completed=false\` and explicit \`errors\`. Do NOT silently fall back to eager or claim GPU PASS.

## Status

Source prepared, CUDA Graph **UNRUN**; local Python compile/unit tests **UNRUN** at GitHub authoring time. Production/main unchanged. Exact eager vs replay numerical gate mandatory.


## RTX failure before capture: missing implicit `group` (2026-11-12)

The local uploaded [`flat_directflow_graph_missing_group_20261112.json`](evidence/flat_directflow_graph_missing_group_20261112.json) reported `completed=false`, `cases=[]`, `KeyError: 'group'` in preflight at `cfg["kern"]["group"]`. No kernels were captured or timed; Graph numeric and performance gates stay **UNRUN**, not FAIL.

Root cause: `flat_config("fast")` uses an implicit center group 1 when the `kern` dict omits the key, matching existing Flat callers. This benchmark incorrectly required the key twice. Added `center_group_from_config` and used its validated result both for preflight and for variable-window planning/reporting; the exact 256-leaf/group1 experiment contract is unchanged. Added an executable CPU-only unit test for absent/explicit/invalid group values. The original report is archived; rerun the same one-JSON command after updating the branch. A source-level/static review is not a substitute for a CUDA Graph capture PASS.
