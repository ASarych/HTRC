# H3 live Direct-Flow CUDA Graph: RTX 5070 Ti run record (2026-11-12)

**Branch:** `experiment/flat-directflow-real-h3-20261112`  
**Code under test:** HTRC `c8b2257085975d158dae02c086ecec26ec0001f1`  
**Machine:** NVIDIA GeForce RTX 5070 Ti, 16 GiB

## Run conditions

The installed ComfyNodes runner was invoked as:

```powershell
run_h3_baseline.ps1 -Length 39 -Steps 2 -AttentionMode htrc_flat_gpu
```

- Checkpoint: `minimax_h3_ref2va_hyperflow_flash_14b-nvfp4_from_int8.safetensors`.
- Scene: bear fight from the same reference-conditioned prompt used in the previous baseline; seed `424242`; 39 frames; 1344×768; 2 denoising steps.
- The run log reports HTRC Flat active at `tau=0.00`, 56 heads, 15,275 attention tokens, `exact_kv=false`; attention memory plan 2.32 GiB, driver free memory 5.58 GiB before and 5.10 GiB after the call.
- Direct-Flow first-call gate: fast mode, 16 tiles, workspace enabled, native BF16 QKV, 256 exact leaves, center group 1, group cap 11, no quality coefficient. Graph geometry cache used group sizes 1 and 11.
- Prompt filename is recorded as `bear_fight_reference.txt`; the full prompt body and reference-image hash were not captured in the JSON. No source image or generated video is included here.

## Result A — first real attention call

Source was the actual H3 post-RoPE BHSD QKV before `out_proj`. The gate compared graph replay against the native-eager HTRC Flat path for all 56 heads, grouped `11+11+11+11+11+1`, at 15,275 tokens.

- Status: **PASS_FIRST_REAL_CALL**; every group was bitwise identical.
- Capture plus parity wall time: 15.196 s.
- Additional allocator-reserved memory reported after the gate: 224 MiB; free VRAM after: 5,227 MiB.
- The JSON's summed per-group eager/graph timing ratio is 69.47×. **Do not treat it as an acceleration result**: per-group measurements contain multi-second outliers and independent synchronizations; capture is excluded. It does not represent steady-state whole-attention or video speed.

Raw measured output: [`research/evidence/flat_directflow_real_h3_first_call_20261112.json`](evidence/flat_directflow_real_h3_first_call_20261112.json).

## Result B — complete sampler attempt

The same 39-frame / 2-step attempt did not complete. At 0/2 progress, approximately 24 seconds into sampling, ComfyNodes raised `RuntimeError: Reusable H3 slot entry layout mismatch` while background-prefetching an H3 block through `h3/hybrid_loader.py`. The runtime log showed about 5.10 GiB free after the attention call, so this is recorded as a slot-layout error, **not an OOM**. No completed video or full-sampler timing was produced.

The reusable-slot fallback was subsequently added to the installed ComfyNodes source, but this run predates that change and does not validate the fix.

## Disposition

- Real-model first-call exact-parity gate: **PASS**.
- Full 39-frame denoise/video: **ABORTED before completion**.
- End-to-end speed, visual quality, and temporal quality: **UNPROVEN**.
- The existing earlier fixed-input CUDA Graph report remains a separate synthetic-input result. This report does not replace it or promote the experimental path to a default.
