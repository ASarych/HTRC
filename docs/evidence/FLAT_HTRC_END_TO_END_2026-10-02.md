# Flat HTRC — real full-video runtime evidence (2026-10-02)

**Prominent result:** The fastest *documented completed full-video run in the HTRC family* in the source repository was the **flat, tree-free HTRC `flat_live` fast/INT8 configuration**. This is a measured ComfyUI run, not a linear per-head extrapolation.

## Reproducibility and provenance

- Original Git branch: [`ASarych/ComfyNodes:research/h3-flat-live`](https://github.com/ASarych/ComfyNodes/tree/research/h3-flat-live), tip `571a7a3e6cbc7c76112ca21ea5d4f2a21a3d649f`.
- Canonical written empirical record: [`h3/STATE.md`](https://github.com/ASarych/ComfyNodes/blob/research/h3-flat-live/h3/STATE.md), heading **"Full run 2026-10-02 01:19 (flat_live, fast, PROD int8 bn64 stages2 splits4)"**; source commit title `STATE: full run with int8 PROD (cycle 24.4 s)`.
- Configuration/conditions: RTX 5070 Ti, 1344 × 768, **124 video frames**, eight denoise steps, 50 H3 transformer blocks, 40,759 token query shape; `APL_H3_RRC3_EVALUATOR=flat_live`, `APL_H3_FLAT_MODE=fast`; INT8 Q/K path, tile_bn=64, tile_stages=2, splits=4. CUDA graphs were disabled for the measured ComfyUI pipeline.
- The source metrics are summaries recorded in git; the original local telemetry JSON, checkpoints, and video bytes are **not bundled** with this release. They have not been reproduced in the standalone export.

## Observed time progression

| Experiment (flat HTRC only) | Real denoise cycle time | Interpretation |
| --- | ---: | --- |
| Before main-thread `_dll` search fix | **45–48 s/step** (cycles 2–8) | Large host launch gap; GPU busy about 62–68% |
| After resolving `_dll` once | **33–34.5 s/step**, first cycle 32.8 s | GPU busy 93.9%, attention around 0.54 s/block |
| 2026-10-02 01:19 full run, tuned real INT8 Q/K | **24.4–25.0 s/step for cycles 3–7**, **26.7 s** for cycle 8 | Fastest recorded successful full-video flat HTRC run; the first and second cycle timings were not individually given in the saved summary |

**Entire Comfy prompt, final run:** **386 s = 6 min 26 s**, including work outside the denoise cycles. Earlier balanced flat run recorded prompt 533 s, 425 s total denoise, about 0.73 s attention per block; these numbers are for a different configuration and are not to be averaged with the final fast/INT8 run.

**Micro-level evidence from final profiled block:** attention accounted for 476 ms of a 755 ms sampled block window; GPU busy 93.9%; tile kernel 2.64 ms/head, statistics 1.07 ms/head, rowmax 0.43 ms/head, top-k 0.12 ms/head, merge 0.12 ms/head. Rough cycle accounting in the state record: 10–12 seconds spent in flat attention and the rest in H3 work and orchestration. This is a *measured contextual breakdown*, not an independent new run.

## Accuracy, validity, and limitations

- Flat HTRC is a tree-free approximation, **not the accepted hierarchical HTRC second-order mass/first-order value-dipole algorithm**. No valid 24.4-second result exists for that distinct hierarchical implementation in the evidence examined.
- Runner docs report `fast` mode at 256 exact source leaves and a saved-capture output-vs-dense relative L2 error distribution of about **9.9 / 26.5 / 36.7 percent** (median/p90/max) over the original GPU test corpus. These figures are **per attention head/capture**, not a video-quality score.
- The real INT8 Q/K isolated sweep gave median rel-L2 differences against dense of **5.51% (INT8) versus 5.50% (BF16)** for a separately selected sweep configuration, not a proof of unchanged full-video quality for the final fast run.
- Original full-video comparisons visually observed similar scene/identity/motion across distinct configurations, but no blind/large-scale quality study or reproducible published rendered outputs is supplied. The final fast/INT8 run does not include a measured perceptual-quality guarantee.
- The first hierarchical 243-frame block-0 canary (~170 s for **one block**) and the later per-head saved-capture streaming gates are on different workloads/algorithms and **cannot** be compared directly as denoise-step results.
- Do **not** claim that flat HTRC is the fastest attention algorithm generally, or that it beats a baseline at matched quality. The supported claim is narrower: it is the fastest **documented complete HTRC-family video run** found in this code history.

## Original preserved sources

- `h3/htrc_flat.py` — mathematical PyTorch implementation and parameter configurations.
- `h3/htrc_flat_triton.py` — Triton kernels, including CPU-launch overhead fix and real INT8 Q/K path.
- `h3/htrc_flat_runner.py` — multihead operator wrapper, memory/chunk planning and optional CUDA graph support.
- `h3/profile_trace.py` — profiling instrumentation.
- `tools/sweep_h3_flat_kernels.py`, `tools/profile_flat_kernel.py`, `tools/compare_h3_error_structure.py` — saved-capture benchmarking and error analyses (require appropriate runtime inputs).

This export preserves original file blobs byte-for-byte. No source file or full-run metrics were modified or manufactured.
