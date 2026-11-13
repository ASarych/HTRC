# H3 run with reusable slots disabled (2026-11-13)

## Outcome

Completed a 39-frame reference-conditioned H3 run with reusable slots explicitly disabled. The earlier attempt with slots enabled stopped before completion with `Reusable H3 slot entry layout mismatch`; this retry used the same requested video conditions and the legacy allocation path. The successful run is not evidence that the slot-layout issue is fixed.

- Checkpoint: `minimax_h3_ref2va_hyperflow_flash_14b-nvfp4_from_int8.safetensors`
- Text encoder: `qwen3vl_4b_fp8_scaled.safetensors`
- Video VAE: `minimax_h3_video_vae_fp16.safetensors`; TensorRT decode
- Prompt: `bear_fight_reference.txt`; reference-conditioned image input
- Reference: `APL_klein_quality_anime_safe20_seed20260911_00001_.png` (SHA-256 `c9dc0978c5543078496b078c976cc7fb6d60bc359f1af41de6229170b98a6ccd`)
- Seed 424242; 39 frames; 1344×768; 24 fps; 2 steps
- Attention: `htrc_flat_gpu`; HTRC DirectFlow graph dispatch; mode fast; 16 tiles; workspace enabled; group cap 11
- Reusable slots: disabled (`-DisableReusableSlots`); legacy allocation path; final counters allocations=0, reuses=0, trims=0
- HTRC source root: `D:\HTRC-directflow-real-20261112`, HEAD `c8b2257085975d158dae02c086ecec26ec0001f1`. The working tree had local modifications in `h3/htrc_flat_head_batch.py`, `h3/htrc_flat_live_backend.py`, `h3/htrc_flat_model_shadow.py`, and `h3/htrc_flat_windows.py`; therefore HEAD alone does not identify the exact source bytes used.
- Run report elapsed: 210.06 s. TensorRT engine preparation: 30.89 s (deserialize 26.83 s; context creation 1.75 s). VAE decode: 37.41 s, 16 calls / 56 tiles.
- Dynamic memory manager: 96 grants, 5 denials, 6 contractions; no ledger errors or failed blocks. Five speculative prefetches were denied because the granted VRAM was below the physical transfer cost; the sampler still completed.
- DirectFlow first real-call gate passed exact parity across all 56 heads / 15,275 tokens. Capture+parity took 3.859 s and reserved 192 MiB extra. This one-shot gate is not a whole-video speedup measurement; no end-to-end baseline comparison was run.

## Artifacts

- Local JSONL log: `D:\ComfyOutput\H3\logs\H3_run_20261113-122822.jsonl` (38,089 bytes)
- Local run summary: `D:\ComfyOutput\H3\htrc_flat_gpu\h3_htrc_flat_gpu_turbo8_trtvae_baseline.json` (1,392 bytes)
- Local output video: `D:\ComfyOutput\H3\htrc_flat_gpu\H3_htrc_flat_gpu_turbo8_trtvae_baseline_00001_.mp4` (4,216,839 bytes; intentionally not copied to Git)
- DirectFlow first-call JSON is committed alongside this report.

Visual/temporal quality and end-to-end acceleration are not evaluated by this run record.
