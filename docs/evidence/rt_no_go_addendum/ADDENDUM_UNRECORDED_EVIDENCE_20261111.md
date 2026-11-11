# Addendum to the HTRC-RT no-go — unrecorded evidence and open items (2026-11-11)

This addendum does **not** reopen `FINAL_NO_GO_HTRC_RT_20261025.md`. The RT decision stands.
It records (a) measurements that existed on the local machine but were never written into
this branch, (b) which report sources are no longer available, and (c) work that the no-go
points to but that has no recorded result anywhere in the repository.

No new GPU run was made for this addendum. Everything below is read from files found on
2026-11-11 under `D:\ComfyOutput\H3\math_probe\`, now copied byte-for-byte into
`results/evidence_local_20261025/`.

## 1. Committed evidence files

| File | Written | Backs |
|---|---|---|
| `htrc_rt_real_hierarchy.json` | 2026-10-25 20:58 | no-go blocker 5 (3D root AABB hold-out) |
| `htrc_real_root_gemm.json` | 2026-10-25 21:05 | no-go section "Existing optimized HTRC already uses GEMM" |
| `htrc_rt_real_rank8_three_way.json` | 2026-10-25 19:59 | no-go blocker 6, first bullet (real rank8 roots) |
| `htrc_rt_rank8_three_way.json` | 2026-10-25 19:39 | synthetic rank8 three-way, 2/5/10/20 % hit fractions |
| `htrc_rt_indexed_fused_validation.json` | 2026-10-25 18:42 | **not referenced by any report** — see section 2 |
| `ncu/available_gpu_metrics.txt`, `ncu/no_rt_tc_ncu.csv`, `ncu/no_rt_simt_ncu.csv` | 2026-10-25 01:05–01:21 | Nsight failure root cause — see section 4 |

Spot checks against the no-go text: the root GEMM medians 0.0786 / 0.0901 ms, the BF16
2-of-20,480 opening mismatch on block8, and the real-rank8 hot timings (RT broadphase
0.0732 ms, RT->CUDA 0.0946 ms, CUDA-only 0.0224–0.0272 ms, shader rank8 2.868 ms) all match
these files.

## 2. Unrecorded result: gather-free indexed fused attention kernel

`indexed_fused_validation.py` was run on the RTX 5070 Ti and its result was never summarised.
Shape: 16 tiles x 64 Q = 1,024 queries, 2,547 B16 blocks, D=128, FP16, 7-token tail,
synthetic Q/K/V, 40 repetitions, kernel-only CUDA events.

| B16 budget | Indexed fused kernel, median ms | Gather + cuBLAS with cached IDs, ms (second run in `rt_tensor_native_first_real_gpu_20261025.md`) | Ratio |
|---|---:|---:|---:|
| 2 % (51 blocks) | 0.0220 | 0.0989 | 4.5x |
| 5 % (128) | 0.0466 | 0.2196 | 4.7x |
| 10 % (255) | 0.0907 | 0.4671 | 5.2x |
| 20 % (510) | 0.1756 | 0.9089 | 5.2x |

Output relative L2 against the reference is ~2.6e-4 in FP16 (2.0e-3 in BF16) for permuted,
reversed, identical and duplicate ID lists; negative and past-end IDs are rejected.

Limits: the two columns come from different executables (Triton kernel vs native cuBLAS
pipeline) and synthetic data; p90 at 20 % is 0.345 ms; this is not a routed pipeline, not
production HTRC, and Tensor Core use is not verified. It answers Gate 3 of
`rt_tensor_architecture_decision_gate_20261025.md` (fused sparse design, option C) in the
positive, and it is independent of RT.

Related stage accounting from `ncu/no_rt_tc_ncu.csv` (5 % budget, only 2 repetitions, so
indicative only): gather 0.034 ms, QK 0.014 ms, **softmax 0.158 ms**, PV 0.015 ms, pipeline
0.221 ms. The gather + cuBLAS baseline is dominated by its separate global-memory softmax
kernel, not by the GEMMs or the gather.

## 3. Gates with code but no recorded result

- `rt_tensor_fixed_budget_gate.py` — deployable fixed per-head budgets without the per-query
  oracle count leak. No JSON, no report. This is the test that decides whether a cheap 128D
  mean-key or block-Gaussian selector is usable on the sparse heads (oracle: heads 1/55 need
  1.9 % / 4.6 % of B16 blocks for 80 % mass; heads 16/27 need 44 % / 33 %).
- `prestate_real_gate.py`, `prestate_bounds_compare.py`, `prestate_bound_probe.py` — 128D
  certified block-mass screening with sphere / coordinate-box / global-PCA-residual bounds.
  No JSON, no report.
- `centroid_value_gate.py` — marked non-authoritative in commit `6820c757`; no report.
- `run_hybrid_speed_windows.ps1 -ProfileOnly` — never run (no
  `htrc_rt_native_hybrid_profile_only.json`, no `available_gpu_metric_bases.txt`).
- Pass criterion 4 of `rt_state_elimination_feasibility_20261024.md` (independent
  head / call / layer validation) was not met: the 3D root proxy hold-out used head 1,
  centroid 0, 16 queries per capture. `attention_block_24_call_01/08.pt` and
  `attention_block_00_call_08.pt` are on disk and unused by this branch.
- Only a single K-only PCA projection was tested as the spatial encoding. A multi-subspace
  scheme (many 2D subspaces with hit accumulation, as in arXiv 2312.01712) was not. Given
  that the fused Triton screen runs in ~0.013 ms where OptiX needed 1.7–4.1 ms on the same
  2,547-leaf screen (non-equivalent timing scopes), this is unlikely to change the decision.

## 4. Nsight failure: root cause is known

`rt_tensor_native_first_real_gpu_20261025.md` lists the Nsight status as FAIL/INCONCLUSIVE
with unknown root cause. The preserved logs state it directly:

    ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
    Performance Counters on the target device 0.

It is a permission setting, not an unsupported metric or a parsing problem. Tensor Core
execution therefore remains unverified for every timing in this branch. Enabling counter
access is a system setting to be changed by the machine owner; it was not changed here.

## 5. Report sources that are no longer available

These JSON files are cited by reports in `results/` but are absent from
`D:\ComfyOutput` and `D:\HTRC-RT` as of 2026-11-11, so those reports can no longer be
re-derived from raw data: `htrc_rt_gpu_screen_*.json`, `htrc_rt_triton_screen_*.json`,
`htrc_rt_fullq_gate_*.json`, `htrc_rt_tensor_mass_80_oracle.json`,
`htrc_rt_native_hybrid_speed.json`, `htrc_rt_geometric_mass_gate.json`,
`htrc_rt_optix_smoke.json`, and the scalar / signed envelope and value-dipole reports.
No `h3_htrc_*` output of the earlier hierarchical research survives there either.

## 6. Where the no-go's recommendation stands in the rest of the repository

The no-go ends with "focus HTRC performance work on reducing frontier state construction
and scheduling without RT". The work prepared for exactly that has no recorded outcome:

- `research/h3-math-probe`, `research/H3_MATH_RESEARCH_STATE.md`: v30 measured state_build
  ~127.44 ms of ~439 ms per head. v31 (exact dataflow batch) is recorded as pending local
  GPU validation. The 2026-09-28 candidate battery (`research/h3-candidate-*`: solver
  variants, exact-small, adaptive Ritz, bottom-up residual family, Sol q-major layouts) is
  recorded as prepared with "no candidate accepted without local evidence"; the bottom-up
  rank16 `eigh` failure is open. No later commit on any branch records a matrix result.
- `research/h3-flat-live` (flat, the fastest completed HTRC-family run, 24.4–25.0 s/cycle):
  frozen at `571a7a3e` since 2026-10-02; no later branch modifies `h3/htrc_flat*.py`. Its
  cost is tile 2.64 + stats 1.07 + rowmax 0.43 ms/head, with top-k at 0.12 ms, which is why
  no candidate router (RT or otherwise) can help it. Open items in `h3/STATE.md`: the fp8
  far-field state slowdown (10.7 vs 6.4 ms, cause not localised), CUDA graphs disabled under
  `cudaMallocAsync`, and the per-head budget lever (worst heads 27 and 1). Flat remains
  ~3x slower in attention than Sol at equal error.

## 7. Hypothesis linking sections 2, 3 and 6 (not a result)

Heads differ strongly in sparsity. For heads like 1 and 55, an exact sparse path (cheap 128D
selector + the indexed fused kernel, no far field) may be both cheaper and more accurate
than a uniform flat or Sol setting, while dense heads like 16 and 27 keep the existing path.
The first test is the fixed-budget gate of section 3 on the existing capture; only if it
passes are a per-head mixed run and a full-model image/video quality check justified.
