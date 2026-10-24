# Flat HTRC: tree-free leaf expansion and near-field correction

**Scope.** This is the later, GPU-measured `flat_live` HTRC-family operator. It shares centroid-local softmax leaf compression with the hierarchical HTRC research, but it is **not** the same adaptive hierarchical runtime, nor the exact second-order / value-dipole research specification.

Original code: [`h3/htrc_flat.py`](../../h3/htrc_flat.py), [`h3/htrc_flat_triton.py`](../../h3/htrc_flat_triton.py), [`h3/htrc_flat_runner.py`](../../h3/htrc_flat_runner.py).

## Algorithm implemented

1. Split query rows into (normally 64-row) tiles. For tile `t`, compute expansion center `μ_t` from its query rows.
2. Partition source keys/values into 16-token leaves. At each tile center, obtain **exact finite-softmax statistics** for every leaf: log-partition `l_z`, scaled key mean `g` and mean value `vbar`.
3. For query `q` in this tile, approximate far-leaf logit by `l_z + (q-μ_t)·g`, with an optional isotropic second-order term. Approximate far-leaf output value as constant `vbar`. These are approximations away from the tile center.
4. Choose a fixed number of influential leaves according to the routing score; calculate their QK and PV contribution exactly and **remove those leaves from the far approximation**.
5. Keep exact sink-conditioning leaves and trailing partial-leaf keys where configured. Merge the far and near parts via numerically stable softmax mass/weighted sums.
6. The optimized path uses Triton kernels, tiled/chunked execution, reuse of temporary memory, and INT8 Q/K storage/dots in the fast configuration. CUDA-graph support exists in the runner, but the measured ComfyUI run kept graphs off due to `cudaMallocAsync` capture failures.

There is **no recursive hierarchy or dynamic traversal**. There are also no global worst-case attention-error proofs for the fixed near-field budget.

## Mathematical statement at a leaf center

For a leaf `B`, log-partition `L_B(q) = log Σ_{j∈B} exp(α q·k_j)` and conditional mean `v_B(q) = Σ_{j∈B} p_j(q)v_j`, with `α=1/√d`.

- `∇_q L_B(μ) = α Σ_j p_j(μ) k_j`.
- `L_B(q) ≈ L_B(μ) + ∇_q L_B(μ)·(q-μ)`.
- `v_B(q) ≈ v_B(μ)` in the recorded flat configuration.

The equalities at `μ` are exact finite-softmax identities; truncating the Taylor expansion and holding the value fixed are approximations. For dense attention, retaining all leaves exactly recovers the original result up to numerical rounding, subject to consistent masking/source semantics.

## Accuracy and status

The original `fast` configuration (256 exact leaves) had per-head relative output L2 error against dense attention on saved full-Q maps of roughly **9.9% median / 26.5% p90 / 36.7% max** (as documented in the runner); quality depends strongly on head, input, source split, and exact-leaf budget. Later INT8 Q/K changes were measured as virtually quality-neutral in an isolated paired operator sweep, but a full end-to-end parity/quality proof for the final video run is **not** provided.

See [actual video-run results](../evidence/FLAT_HTRC_END_TO_END_2026-10-02.md). Original experimental source requires torch, triton, NVIDIA GPU and environment-specific H3 capture/integration; no fresh standalone GPU validation is claimed.
