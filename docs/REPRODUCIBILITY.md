# Reproducibility and research scope

This repository contains **two separate HTRC-family attention paths** and a third, minimal CPU teaching/reference implementation. Their mathematical formulas, measured speed and readiness are not interchangeable.

## 1. Runnable CPU / NumPy mathematical reference

Requirements: Python 3.10+ and NumPy (declared in `pyproject.toml`).

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/demo.py
```

The unit tests assert exact log-partition/gradient/Hessian/value/value-Jacobian identities, parent aggregation, and exhaustive-leaf recovery on small synthetic inputs. These tests do **not** validate the original B16 RRC3 leaf law or H3 Triton CUDA runtime.

## 2. Experimental hierarchical HTRC

Files: `h3/htrc_live.py`, `h3/htrc_executor*.py`, `h3/rrc3_kernels.py`. The original research used 16-token leaves, query centroids, a mass-routing tree, and CUDA kernel state build/traversal.

The available early 243-frame / 56-head live canary (~170 s) measured **one attention block**; it must not be interpreted as an entire denoise step. Later per-head replay work also lacks a recorded end-to-end video denoise-step result for the same hierarchical implementation.

CUDA code is preserved for source inspection and adaptation, **not** presented as a working standalone pip package. Parts of the runner may import original H3 runtime modules, require capture files and assume experimental GPU shape contracts.

## 3. Flat HTRC (the fastest complete-video HTRC run on record)

Files: `h3/htrc_flat.py`, `h3/htrc_flat_triton.py`, `h3/htrc_flat_runner.py`. Profiling tools: `h3/profile_trace.py`, `tools/sweep_h3_flat_kernels.py`, `tools/profile_flat_kernel.py`, and `tools/compare_h3_error_structure.py`.

Measured on 2026-10-02 in the original full ComfyUI runtime: RTX 5070 Ti, 1344x768, **124 frames**, 8 steps, 50 transformer blocks, flat fast INT8 Q/K. Denoise steps 3-7 took **24.4-25.0 s per step**; step 8 took **26.7 s**, whole Comfy prompt **386 s**. Earlier stages were 45-48 s/step, then 33-34.5 s/step after removing a Python/Triton host bottleneck.

See [the evidence report](evidence/FLAT_HTRC_END_TO_END_2026-10-02.md) for exact conditions, errors and limitations.

To reproduce those timings rather than just test the NumPy reference, an independent investigator must reconstruct the original model weights, graph, NVIDIA GPU execution environment, workload, source/sink key layouts, reference inputs and launch configuration, then capture comparable end-to-end timings and output quality. The original telemetry/video files and checkpoints are **not** redistributed here.

## What is verified in this repository?

- Original research files and empirical reports are preserved as source-text records.
- The CPU test suite was exercised during preparation of the mathematical reference; rerun it after cloning.
- No new CUDA compilation, ComfyUI full-video run, or baseline comparison has been executed by this publication process.
- The 24.4-25.0 s figure is **not a claim of speedup over any attention baseline at equal quality**, nor a measurement of hierarchical HTRC.

## Origin and privacy

The public repository was formed as a new commit on top of the existing public `ASarych/HTRC` initial commit, by copying selected source *files* from private `ASarych/ComfyNodes`. The private repository's commit ancestry is **not** included. Origin tips:

- Hierarchical mathematics: `542a0d3e68aff62cfd910a967b2e7cf3ac760d46`.
- Earlier runtime experiments: `79c5d494adc26a2410422f79a49355996e1948af`.
- Flat HTRC 2026-10-02: `571a7a3e6cbc7c76112ca21ea5d4f2a21a3d649f`.
