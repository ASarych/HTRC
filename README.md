# HTRC — Hierarchical Tilted Routed Collapse

**Public research release of the HTRC attention family: hierarchical HTRC, measured flat HTRC, mathematical reference, derivations, and empirical results. Experimental; not a drop-in production attention backend.**

This repository preserves HTRC theory, derivations, original experimental PyTorch/Triton code, an independently testable NumPy mathematical reference, and historical evidence. It excludes the full ComfyUI application, model checkpoint weights, raw private QKV captures, prompt files and unrelated research. **Two HTRC-family variants are included: hierarchical HTRC and the faster measured, tree-free flat HTRC operator.**

## Getting started

The CPU mathematical reference can be installed without ComfyUI or a GPU:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/demo.py
```

The `h3/` PyTorch/Triton implementations are **archival research code**, not packaged CUDA extensions. They require an NVIDIA GPU, matching PyTorch/Triton, the original H3 integration and the model-specific Q/K/V capture conventions. See [reproducibility notes](docs/REPRODUCIBILITY.md) before treating them as runnable in isolation.

## HTRC variants and measured video-generation performance

**Fastest documented completed HTRC-family full run in the available source history: `flat_live`** (RTX 5070 Ti; 1344x768; **124 frames**; eight denoise steps; 50 H3 transformer blocks). The 2026-10-02 run with INT8 Q/K, `fast` mode and tuned Triton kernels recorded **24.4–25.0 s per denoise step for steps 3–7**, 26.7 s for step 8, and **386 s for the entire prompt**. These are actual ComfyUI runtime observations, **not a per-head-to-step extrapolation**. Earlier flat runs ranged 45–48 s/step before a CPU stall fix, then 33–34.5 s/step, then ~24–25 s/step after INT8 tuning.

See **[full flat HTRC results](docs/evidence/FLAT_HTRC_END_TO_END_2026-10-02.md)** and **[flat-operator semantics](docs/theory/FLAT_HTRC_OPERATOR.md)**. Original, unchanged source lives in `h3/htrc_flat.py`, `h3/htrc_flat_runner.py` and `h3/htrc_flat_triton.py`; the original profiling/sweep scripts are in `tools/`.

**Do not conflate these with hierarchical HTRC.** Its recorded early full-head, 243-frame block-0 prototype was ~170 s **for a single block**, not a finished denoise step. Later research optimized per-head streaming on saved captures but did not establish a matching full-video run for the same hierarchical implementation. The tree-free flat variant uses a different selection and value approximation; its speed and quality measurements must stay separate. The flat result is the fastest *documented full-video HTRC-family* run here, not a claim that it outperforms all other attention methods.

## What is mathematically established

For finite attention, at scaled query s define Z(s)=Σ exp(s·k), L=log Z, conditional value u=Σp·v. Then

- g=∇L=E[K];
- H=∇²L=Cov(K,K);
- J=∂u/∂s=Cov(V,K).

For node N made of children i, at the SAME query centroid, with mass weights w_i = exp(L_i)/Σ exp(L_j),

- L_N = logsumexp(L_i);
- g_N = Σ w_i g_i;
- H_N = Σ w_i [H_i+(g_i-g_N)(g_i-g_N)^T];
- u_N = Σ w_i u_i;
- J_N = Σ w_i [J_i+(u_i-u_N)(g_i-g_N)^T].

These are exact derivative identities. Approximating L_N(s0+δ) by L_N+g_N·δ+½δ^TH_Nδ, and u_N by u_N+J_Nδ, is NOT exact except at the expansion point. The mass-weighted opening certificate is a practical *heuristic*, NOT a rigorous global error bound.

See `docs/theory/` for original detailed derivations and tests.

## How to read the source

- `docs/theory/` — frozen derivations / theoretical variants. Some are historical proposed methods and were **subsequently rejected**.
- `docs/evidence/` — result files, including negative results and numeric experiment conditions.
- `h3/htrc_live.py` — the original hierarchical research state builder, using PyTorch.
- `h3/htrc_flat*.py` — the later flat (tree-free) HTRC-family operator, CUDA/Triton kernel and runner with real completed-video runtime evidence.
- `h3/rrc3_kernels.py` — B16 RRC3 and GPU/Triton kernels needed by the experimental executor.
- `h3/htrc_executor*.py` and `h3/htrc_telemetry_kernels.py` — opt-in experimental GPU executor and helpers, **not an end-to-end Sol replacement**.
- `tools/` — preserved experiment-specific checks/replay scripts, including flat HTRC runtime kernel sweeps and profiling. They may require saved H3 tensors or modules absent from this clean export.

**Important limitation:** The archived GPU source requires NVIDIA CUDA, PyTorch, Triton and the original H3 runtime shapes/protocol. Flat HTRC was measured inside ComfyUI, but **this standalone repository has not independently reproduced those GPU numbers**. The NumPy tests exercise the mathematical reference only; they do not validate either CUDA backend or end-to-end video generation.

## Findings and important negative results

The 2026-09-25 held-out even/odd-query offline tree experiment (eight representative heads) recorded at open=0.02: mean abs log-Z error ~0.01638, total source-work reduction ~18.99x. **This is not a 19x GPU speedup.** A later 243-frame block-0 live canary (56 heads) needed ~109 seconds for state build and ~58 seconds for traversal, ~170 seconds total; runtime optimization is unresolved.

Signed rank-8 curvature as the *primary accepted-node mass Hessian* was rejected: it kept routing decisions but substantially worsened log-mass error. The latest accepted research authority is second-order log-mass, first-order value dipole, mass-only routing, and cross-fit query-centroids (Cq=32/K=16). The runtime prototype's rank-8/monopole value mode is a narrower, non-equivalent implementation gate.

## Source provenance

- Mathematical source: `ASarych/ComfyNodes`, branch `research/h3-math-probe`, tip `542a0d3e68aff62cfd910a967b2e7cf3ac760d46`.
- Runtime / historical result source: branch `feature/apl-h3-runtime`, tip `79c5d494adc26a2410422f79a49355996e1948af`.
- **Fastest recorded completed HTRC-family video:** branch `research/h3-flat-live`, tip `571a7a3e6cbc7c76112ca21ea5d4f2a21a3d649f`, measured 2026-10-02; the flat source Git blobs are preserved verbatim.
- Original blobs are copied byte-for-byte by Git object SHA; README and supplementary proof / publication guides are new.

## Publication history and licensing

This public repository contains **files-only copies** of the selected original research blobs and **does not inherit the private ComfyNodes Git history**. Historical origin commits are listed for provenance; some original source links point to a private repository and are not publicly readable. Runtime timings are reported observations from its saved research state, not new independent benchmark runs.

The repository includes the [MIT license](LICENSE) created by its owner. Original experimental modules were copied from the author's own project; check any third-party notices and dependencies before redistribution of derivative integrations.

A standalone NumPy mathematical reference is now included in `src/htrc_reference/`, with 5 deterministic tests in `tests/` and a demo in `examples/`. Run `python -m pip install -e .`, then `python -m unittest discover -s tests -v` and `python examples/demo.py`. The reference illustrates exact tree identities with ordinary softmax leaves; unlike the H3 code it does not implement B16 antipodal RRC3 leaves or Triton traversal.

