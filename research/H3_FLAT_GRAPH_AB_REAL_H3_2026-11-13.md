# Real H3 Graph ON/OFF A/B (2026-11-13)

## Conditions

Four complete runs through the installed ComfyNodes runtime: two Graph OFF (HTRC eager dispatch), then two Graph ON (DirectFlow CUDA Graph). The backend was restarted only between dispatch modes; the second repeat in each mode reused that mode's backend. No model, runtime, or dependency was downloaded.

- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GiB.
- Checkpoint: `minimax_h3_ref2va_hyperflow_flash_14b-nvfp4_from_int8.safetensors`.
- HyperFlow checkpoint configuration reports 6 steps; all four runs used 6 (the generic launcher default of 8 was not used).
- Prompt: `bear_fight_reference.txt`; reference image `APL_klein_quality_anime_safe20_seed20260911_00001_.png`, SHA-256 `c9dc0978c5543078496b078c976cc7fb6d60bc359f1af41de6229170b98a6ccd`.
- Same seed 424242, reference-conditioned mode, 39 frames, 1344×768, 24 fps, Euler + Simple, denoise 1.0, BasicGuider; same model/text encoder/ClipProj/VAE.
- Attention: `htrc_flat_gpu`, mode fast, 16 tiles, workspace enabled, group cap 11, tau 0, quality coefficient 0.
- Only A/B change: `APL_HTRC_FLAT_DISPATCH=eager` versus `graph`.
- Reusable GPU slots disabled in all runs. Each final log says `enabled=False allocations=0 reuses=0 trims=0`.
- Attention phase profiling was enabled in every run. It synchronizes at profile boundaries and retains per-step latent checkpoints; therefore timing figures include the same profiling overhead in both conditions.
- HTRC source root: `D:\HTRC-directflow-real-20261112`, HEAD `c8b2257085975d158dae02c086ecec26ec0001f1`; its working tree already had local edits in `h3/htrc_flat_head_batch.py`, `h3/htrc_flat_live_backend.py`, `h3/htrc_flat_model_shadow.py`, and `h3/htrc_flat_windows.py`. ComfyNodes HEAD was `7e2559212c59f79cffe5740e0fa35d82afacf6c3`; its working tree was also already modified. No source files were changed for this measurement.

## Results

| Dispatch | Repeat | Six-step denoise progress | Full prompt elapsed |
|---|---:|---:|---:|
| Eager / Graph OFF | 1 | 75 s (12.5 s/step avg) | 227.6 s |
| Eager / Graph OFF | 2, warm backend | 64 s (10.7 s/step avg) | 165.8 s |
| DirectFlow / Graph ON | 1 | 53 s (8.8 s/step avg) | 188.8 s |
| DirectFlow / Graph ON | 2, warm backend | 48 s (8.0 s/step avg) | 142.1 s |

Across the two repeats, denoising averaged 69.5 s OFF and 50.5 s ON (27.3% less with Graph ON). The warm repeat was 64 s OFF versus 48 s ON (25% less). Full prompt elapsed averaged 196.7 s OFF and 165.5 s ON (15.9% less). The first run in each mode includes a fresh backend; report startup/conditioning and decode separately when interpreting end-to-end totals.

## Full attention call and accuracy

The Graph ON real-model gate covered one complete H3 attention invocation: all 56 heads, 15,275 tokens, six head groups. It passed bitwise parity against eager for every group. Summed per-group GPU events were 88.455 ms for Graph and 397.686 ms for eager; capture plus parity wall time was 1,214.35 ms. These independent per-group event sums are diagnostic and must not be read as whole-video speedup. The steady-state whole-denoise timings above are the end-to-end performance comparison.

The two saved final AV latents (OFF repeat 1 and ON repeat 1) matched bitwise. All six retained step checkpoints also matched bitwise across OFF/ON; repeat 1 and repeat 2 matched bitwise within each dispatch as well. Maximum absolute difference and RMSE were 0.0 at every compared step.

## VRAM and memory manager

- HTRC attention admission target: 2.32 GiB.
- Graph capture added 192 MiB allocator-reserved memory; free VRAM after the real-call gate was 5,227 MiB.
- First-call driver peak-drop / memory retained at return: OFF repeat 1 = 0.44/0.19 GiB; OFF repeat 2 = 0.28/0.06 GiB; ON repeat 1 = 0.41/0.41 GiB; ON repeat 2 = 0.19/0.06 GiB.
- Spot checks during denoising saw total GPU free memory as low as about 1.0 GiB. These were point samples, not a continuous peak trace.
- All four dynamic-budget reports had zero denials, failed blocks, ledger errors, or OOMs.

## Artifacts

The four MP4s and two final latent files remain under `D:\ComfyOutput\H3`; the logs and individual run JSON reports remain locally there. The MP4s were not uploaded. The HTRC source modifications listed above were not changed by this test.

The runtime's current profiler records the first two attention calls per run, not a summed steady-state attention duration over every transformer block. This report therefore gives the full real-call graph/eager gate plus full denoise wall time, and does not claim a full-model attention-time sum.
