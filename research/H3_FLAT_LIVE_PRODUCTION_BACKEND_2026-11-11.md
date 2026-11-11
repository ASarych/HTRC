# Flat HTRC as an actual selectable H3 runtime backend

Research implementation only. Flat outputs (not a shadow) are sent to H3 out_proj.
Canonical Flat source: ASarych/HTRC branch experiment/flat-model-resident-shadow-20261111.
Real APL H3 wiring: ASarych/ComfyNodes branch experiment/h3-flat-production-backend-20261111.

## Runtime ownership and semantics

- ComfyNodes h3/production_runtime.py selects htrc_flat_gpu via APL_H3_ATTENTION_MODE
  or the optional attention_profile dropdown on APLH3ProductionRuntime.
- ComfyNodes h3/attention_runtime.py owns preflight, driver headroom, model block
  streamer, fallback policy, and memory telemetry.
- ComfyNodes h3/attention_htrc_flat.py imports the canonical external HTRC
  checkout via APL_HTRC_ROOT, not the stale ComfyNodes research copy.
- HTRC h3/htrc_flat_live_backend.py consumes native post-QKV/norm/RoPE BHSD
  GPU tensors without a whole-capture dump or copying all QKV to CPU.
- It returns actual [1,S,H,128] BF16 output via GPU K8 group caches,
  newly computed query statistics/selection and Tile+R32 exact tail merge.
- Cache is private to ONE attention invocation and ONE head group.
  Never reused across blocks, CFG steps, denoise cycles, or prompts.
- K/V head indices are unique real native model heads, never logical repeats.
- No persistent CUDA graph, no implicit production fallback on OOM.
- Unsupported input mask/batch/layout may explicitly go to dense fallback
  through ComfyNodes; supported B1 BF16 self-attention must use Flat.
- During a Flat call, the original native dense/Sol output is NOT computed.

## VRAM responsibility

Two cooperating but separate controllers:

1. Model-level APL attention_runtime.memory_plan() reserves full BF16 output
   plus a bounded Flat per-group arena (default 1024 MiB) and a device-scaled
   memory floor (~1024 MiB on 16 GiB GPU). It publishes reserve and asks
   _apl_prepare_attention_transient BEFORE dispatch.
2. HTRC then allocates the output, reads CURRENT GPU free memory and limits
   head group count via plan_tile_head_batches + K/V and query workspace cost.
   If even one head cannot fit, fail closed; do not silently switch backend.

Additional structural planning is owned by ComfyNodes tools/h3_memory_math.py.
The launcher disables DynamicVRAM/Comfy compiler for the Flat mode so APL owns
model residency as with existing Sol. These are estimates, not yet GPU verified.
Use APL_HTRC_FLAT_BUDGET_MIB, APL_HTRC_FLAT_RESERVE_MIB, APL_HTRC_FLAT_TILES
and APL_HTRC_FLAT_MODE to tune the arena/query window/mode.

## Native first-forward/39-frame test (Windows PowerShell)

Check out BOTH experiment branches to their own working copies.
Do not change the HTRC directory's existing origin remote; use its htrc remote.

    cd D:\ComfyNodes-flat-split-20261111
    git fetch htrc
    git switch --detach htrc/experiment/flat-model-resident-shadow-20261111

Install/check out ASarych/ComfyNodes experiment/h3-flat-production-backend-20261111
in the actual custom_nodes checkout used by the ComfyUI server, or have that
branch loaded by the ComfyNodes development launcher. Do not overwrite a
dirty ComfyNodes checkout or a currently running server's imports.

    $env:APL_HTRC_ROOT = 'D:\ComfyNodes-flat-split-20261111'
    $env:APL_H3_ATTENTION_MODE = 'htrc_flat_gpu'
    $env:APL_HTRC_FLAT_BUDGET_MIB = '1024'
    $env:APL_HTRC_FLAT_RESERVE_MIB = '256'
    $env:APL_HTRC_FLAT_MODE = 'fast'
    $env:APL_HTRC_FLAT_WORKSPACE = '1'

Run in actual ComfyNodes repository directory:

    .\run_h3_baseline.ps1 -Length 39 -Steps 8 -AttentionMode htrc_flat_gpu -FirstForwardOnly

Then, only on real first-forward success:

    .\run_h3_baseline.ps1 -Length 39 -Steps 8 -AttentionMode htrc_flat_gpu

The baseline launcher performs tools/check_h3_runtime_wiring.py --mode htrc_flat_gpu
and uses APL-controlled block residency. All existing output, telemetry,
recovery and model checkpoint files follow normal H3 baseline semantics.
The launcher refuses unsupported H3 lengths not equal to 17*k+5.

## Safety and evidence needed

The preceding K/V and query-workspace A/B had all 520 group×window parity
checks PASS, but the 56 jobs repeated FOUR captured native heads.
The new selected Flat path has NOT been GPU tested with 56 true model heads.
The Fast mode is approximate: parity with dense/Sol and video quality are
not established. Compare same seed/settings against Dense/Sol for:
first-forward finite + VRAM + Flat dispatch, real denoising step/second,
all 56 heads, 39-frame quality, temporal consistency, and total memory.
Do not call production-ready or promote to main until actual local gates.
GPU model/visual tests currently UNRUN. No CI jobs launched.
