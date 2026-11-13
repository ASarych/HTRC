# H3 Flat Balanced CUDA Graph video run — 2026-11-13

## Result

A full 39-frame HyperFlow run completed with HTRC `balanced` attention and CUDA Graph dispatch enabled. The actual first H3 post-RoPE QKV call passed an eager-vs-graph **bit-exact parity gate for all 56 heads** at 15,275 tokens. The run completed without OOM or memory-admission denial.

The generated video is `D:\ComfyOutput\H3\backend\H3_htrc_flat_gpu_balanced_graph_on_p-h3_00001_.mp4` (1344×768, 24 fps, 1.625 s). Total Comfy prompt time was **206.11 s**. Sampler progress intervals were **76.02, 6.93, 6.38, 10.26, 6.71, 6.75 s** (sum 113.05 s). The first interval includes **33.044 s** for the first Graph capture and eager parity gate; it is not a steady-state step time.

## Conditions

- Model: `minimax_h3_ref2va_hyperflow_flash_14b-nvfp4_from_int8.safetensors`; encoder: `p-h3-image_text_encoder_w4a8.safetensors`; video VAE: `minimax_h3_video_vae_fp16.safetensors`.
- Prompt: `C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\custom_nodes\ComfyNodes\h3\prompts\bear_fight_reference.txt`; SHA-256 `96dd9149192c9595fd18fca384bb7b5c8f3135bb1b2573a68dd48422463d2ca8`.
- Reference: `D:\ComfyOutput\H3\backend\APL_klein_quality_anime_safe20_seed20260911_00001_.png`; SHA-256 `c9dc0978c5543078496b078c976cc7fb6d60bc359f1af41de6229170b98a6ccd`.
- Seed `424242`, 6 Euler steps, `HyperFlowSigmas`, BasicGuider CFG 1.
- HTRC mode `balanced`, 512 exact leaves, group 1, 16 tiles, workspace enabled; CUDA Graph dispatch; query staging `native_torch_mu`; K/V staging `native_direct`; no quality coefficient; reusable slots disabled.
- TRT video decoder.

## Measurements

- Graph gate capture + parity wall: **33.044 s**.
- Graph cache extra reserved memory: **288 MiB**; free model-resident VRAM after the first call: **10891 MiB**.
- Host memory telemetry during the approximate request interval: minimum available **3.30 GiB**, maximum commit used **29.96 GiB** (70 samples). No OOM or manager denial.
- The first eager parity calculation included cold compilation, so its per-group duration is not a warmed speed comparison.

## Limits

The initial attempt with `qwen3vl_4b_fp8_scaled.safetensors` was rejected by the current ComfyNodes encoder loader before denoising. This successful run used the installed P-H3 4B encoder; visual quality is therefore not a strict encoder-matched comparison with the earlier Graph A/B videos. A single frame at 1.0 s was inspected; temporal quality was not quantitatively assessed. Per-step values above are prompt progress-event deltas because this reused backend did not write a separate per-step CUDA profile file for this submit.

Full machine-readable evidence: [`evidence/h3_flat_balanced_graph_run_20261113.json`](evidence/h3_flat_balanced_graph_run_20261113.json).
