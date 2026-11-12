# H3 live Direct-Flow CUDA Graph: opt-in real-model gate (2026-11-12)

**Research branch**: experiment/flat-directflow-real-h3-20261112, based on
HTRC experiment/flat-directflow-mutable-qkv-20261112
@ d201029cbbb0daf5009e92f0e149c1c83318bb52.

## Purpose and evidence boundary

The earlier synthetic mutable-QKV proof passed full-output bitwise parity
for two distinct BF16 banks (11 heads, 18,723 tokens, 19 windows), with
CPU+GPU sync wall improvement 15.6142 -> 14.3770 ms (1.086x). The
fixed-geometry CUDA Graph alone did not prove all 56 real H3 heads or
16 GiB model-resident memory. This slice ports that exact Graph staging,
K8/KS refresh, exact tail refresh, Q-window refresh and two-geometry
replay to the actual HTRC Flat H3 output function.

The **default** htrc_flat_live_backend.flat_attention_bthd is still eager.
The ComfyNodes canonical external adapter already calls this function and
needs NO edits to select the experimental path when
APL_HTRC_FLAT_DISPATCH=graph is set before ComfyUI starts.

## Implementation

- No new Flat mathematical kernel, approximated centroids, quality
  coefficient or exact-leaf-count change. Only mode=fast, tiles=16,
  BF16 native QKV, group=1, exact_leaves=256, workspace=on.
- The live entrypoint validates this setting; unsupported configurations
  raise immediately. Off-by-default eager uses the same old code.
- The model's real post-RoPE Q/K/V are native BHSD GPU sources; each graph
  uses its own **stable staging** K/V and Q-window tensors. No graph owns
  model weights, raw model QKV pointers, or a cross-step computed K8.
- Graph sessions keyed by sequence geometry and head-group size persist
  for reuse across sequential blocks/timesteps; one bounded cache is
  cleared on geometry change or failure. Device-wide RLock avoids concurrent
  stage reuse. Head groups are bounded by both the conservative live Flat
  memory planner and a maximum of 11 (configurable downward).
- At every call/group: copy fresh K/V, requantize native K to K8+KS,
  refresh exact K/V tail, copy each current Q window and replay the graph.
  The same geometry graph is reused for ordinary windows and a separate
  one for the dynamic final short window. All output rows are written
  before the next head group reuses the graph stage.
- First real attention call runs Graph and the existing **native-eager
  HTRC Flat** for every genuine head group on the same live QKV.
  Requires **bitwise identical BF16** outputs; aborts without silent
  fallback on any mismatch. Eager baseline is
  query_staging=native_torch_mu + kv_staging=native_direct, the same
  mathematical path as the captured graph. It is not the old gather
  implementation and does not prove dense/Sol visual quality.
- First-call per-group GPU event & sync wall include all QKV stage copies,
  K8+KS refresh, 19 windows, Tile, Merge and BF16 output; graph capture
  is measured separately and excluded from these GPU times.
- No model-resident full-video performance or quality claim until an
  **actual user GPU run** returns the JSON plus video telemetry.
- Fail closed if additional cached graph allocated memory exceeds
  APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB or free VRAM falls below
  APL_HTRC_DIRECTFLOW_MIN_FREE_MIB. This probe must not silently
  substitute eager/Sol/dense in selected graph mode.
- The opt-in run is purposely more expensive on the **first** model
  attention call because it compiles graphs and verifies eager parity.
  Full denoising speed must be measured separately in steady state.
- No additional tensor/weight capture files are written. ONE JSON is
  written on first-call PASS or FAIL_CLOSED; pass is limited to one real
  full attention call, not whole video.

## Windows: prepare isolated HTRC worktree

Use only an existing HTRC repository and remote; do not assume D:\ComfyNodes
exists (it does NOT on the owner's machine). Do NOT switch the working
Direct-Flow tree.

    $src = 'D:\ComfyNodes-flat-split-20261111'
    $wt = 'D:\HTRC-directflow-real-20261112'
    $py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
    if (Test-Path $wt) { throw "Target exists: $wt" }
    git -C $src fetch htrc
    if ($LASTEXITCODE -ne 0) { throw 'HTRC fetch failed' }
    git -C $src worktree add --detach $wt htrc/experiment/flat-directflow-real-h3-20261112
    if ($LASTEXITCODE -ne 0) { throw 'HTRC worktree failed' }
    Push-Location $wt
    try {
        & $py -m unittest discover -s tools -p test_h3_flat_directflow_live_contract.py -v
        if ($LASTEXITCODE -ne 0) { throw 'CPU contracts failed' }
    } finally { Pop-Location }

## Actual H3 run (only after verifying live ComfyNodes is installed)

Stop the previous ComfyUI backend before starting a new one; an already
running backend does NOT pick up new environment variables. Do not switch
a dirty ComfyNodes checkout. Run from the installed production-capable
ComfyNodes script, explicitly opting into the graph. Set before launch:

    $env:APL_HTRC_ROOT = $wt
    $env:APL_HTRC_FLAT_DISPATCH = 'graph'
    $env:APL_HTRC_FLAT_MODE = 'fast'
    $env:APL_HTRC_FLAT_TILES = '16'
    $env:APL_HTRC_FLAT_WORKSPACE = '1'
    $env:APL_HTRC_DIRECTFLOW_GROUP = '11'
    $env:APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB = '768'
    $env:APL_HTRC_DIRECTFLOW_MIN_FREE_MIB = '256'
    $env:APL_HTRC_DIRECTFLOW_REPORT = 'D:\ComfyOutput\H3\math_probe\flat_directflow_real_h3_ab.json'

    $nodes = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\custom_nodes\ComfyNodes'
    if (-not (Test-Path (Join-Path $nodes 'run_h3_baseline.ps1'))) {
        throw 'Installed H3 runner not found: check ComfyNodes deployment'
    }
    & (Join-Path $nodes 'run_h3_baseline.ps1') -Length 39 -Steps 2 -AttentionMode htrc_flat_gpu

The intended output is a single JSON at the path above with:
- actual H3 56-head first-call group parity and group latencies;
- aggregate eager/graph event times (NOT complete video gain);
- capture and extra-pool memory cost, current free VRAM;
- explicit FAILED status if the Graph/parity/VRAM gate aborts.

After the first successful gate, do a separate realistic same-prompt
full video run with graph flag ON vs OFF to prove whole-sampler benefit
and visual/temporal parity, while respecting the owner's 16GiB limit.
Do NOT promote to main / default or assert video speed before those
real runs. Do not rerun superseded standalone Model-Resident Shadow.

## Status

- Source created in HTRC research-only branch; ComfyNodes production
  code and defaults untouched.
- GPU real full-call and complete H3 video: UNRUN.
- CPU AST and staging contracts: UNRUN until executed in a Python
  environment with source checkout. Do not count them as PASS by
  merely reading this document.
