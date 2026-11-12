# H3 Flat: temporal leaf-structure delta / look-ahead feasibility (2026-11-12)

**Research only.** Owner recovered a missing design decision from a lost temporary conversation: exploit the evolution of attention state across successive H3 denoising calls; **do not retain the entire K/V or per-query cache**. Try updating the *assembled leaf-related structure* from its changes, rather than paying for a full rebuild every time. This note is a new reconstruction of that proposal, **not** a transcript of the unavailable temporary chat.

Source: `experiment/flat-directflow-mutable-qkv-20261112@d201029cbbb0daf5009e92f0e149c1c83318bb52`. The successful mutable-QKV Direct-Flow CUDA Graph and its CPU/GPU gates are unchanged. The old stand-alone model-resident shadow is **SUPERSEDED / NOT REQUIRED** and is not a prerequisite.

## Correct Flat boundary

Flat has **fixed 16-token sequential leaf partitions**. It does NOT rebuild a hierarchical tree or k-means clustering. Expensive **per-query-centre, per-leaf** statistics are refreshed for every attention call/window:

- `MU`: query-centre mean, depends on current Q.
- `G` INT8 + `SG` FP32: weighted key slope and its row scale; depends on `MU` and K.
- `VB` BF16: weighted value; depends on `MU`, K, V.
- `LZ` FP32: leaf log-partition; depends on `MU`, K.
- `SEL -> topk -> IDX/EXCL`: near-leaf selection; depends on the full current Q and updated summaries.

Therefore simply copying the old leaf summary is **not** valid when Q/K/V change, even if 16-token leaf membership is unchanged. An exactly unchanged `MU` plus exactly unchanged 16 K/V rows permits exact **statistic** reuse, but selection and final output must still be recomputed for a new Q.

## Critical VRAM economics: only compact structure may persist

At **18,723 tokens, 56 distinct heads, 293 query centres, 1,170 leaves**:

| One H3 block, all query centres | Estimated persistence |
| --- | ---: |
| All previous `G8` + `VB16` + `LZ32` + `SG32` | **7.01 GiB** |
| Only selected near-leaf `IDX` as int32, 256/head/centre | **16.02 MiB** |

Keeping all previous G/VB/LZ/SG across 49 blocks/steps on RTX 5070 Ti 16 GiB is untenable. The proposed long-lived structure, if any, must be a **compact selected-leaf set and/or a much smaller per-leaf summary/delta**, never the complete query-dependent arrays. The 7.01 GiB figure excludes QKV, model weights, output and all other memory. Even an IDX cache across 49 blocks is not free (~785 MiB before metadata), so eviction/scope must be explicit.

## First feasibility implementation

`tools/benchmark_h3_flat_temporal_leaf_delta_ab.py` (this branch) measures two complementary opportunities. It does NOT install a new runtime or make quality assumptions.

**A. Exact compact selected-leaf change — PRIMARY:** Two QKV states are processed through the **same native HTRC Triton** leaf-stat, rowmax/topk and Tile/Merge kernels. The report counts unchanged **selected-leaf sets**, additions/removals and the number of changed **ordered `IDX` positions**. It demonstrates that applying an oracle position patch exactly reconstructs the fresh `IDX` byte-for-byte. **Important:** this is an *oracle size* measurement: it already knows the freshly computed second IDX; finding missing incoming leaves without recomputing every score is the unsolved fast-update problem. Even identical selected sets do **not** prove identical final attention output.

**B. Cheap aggregate-delta hypothesis — SECONDARY:** Given old `G8/SG/VB/LZ`, the newly observed K/V and new MU, form compact mean deltas per fixed 16-token leaf:

- `dK_l = mean(K_new[l] - K_old[l])`, `dV_l` likewise (computed once per leaf independent of Q centre).
- `g_old ≈ G8_old * SG_old`, `dMU = MU_new - MU_old`, `scale = 1 / sqrt(128)`.
- `g_hat = g_old + scale * dK_l`, and requantize the **same** INT8 formula.
- `VB_hat = VB_old + dV_l`.
- `LZ_hat = LZ_old + dot(dMU, g_old) + scale * dot(MU_new, dK_l)`.

These omit K/V–Q covariance terms and the errors from restoring old quantized G. **Approximation, not an exact update.** The probe then rebuilds `SEL/IDX/EXCL` from the predicted summaries and runs the **unchanged** HTRC Tile+Merge on new Q, K8/V/KS, comparing final output to the freshly rebuilt fixed-Flat output. Any differences mean **NO production acceptance**; no loosened `exact_leaves`, centre grouping, quality coefficients or silent fallback.

The test reports exact-dirty fractions, selected-set/ordered-ID churn, output rel-L2/max abs/bitwise equality, intermediate errors, peak memory and paired post-MU/post-K8 GPU+CPU timings **including** K/V mean-delta generation in the candidate. The benchmark intentionally excludes native QKV projection, new-MU computation, K8 quantization, model weights, actual denoising and image quality. The PyTorch delta prototype has multiple small launches, not an optimized fused Triton delta kernel; timing may lose even if a fused implementation could help.

### Input limitations and learning sequence

1. **Available immediately:** synthetic 4-head, 18,723-token base with `dense_drift`, `sparse_drift`, `uniform_leaf_drift`. These are *controlled numerical scenarios*, **NOT two actual denoising steps**.
2. **If an existing trusted QKV capture is available:** `--capture-a PATH` uses its 4 genuine captured heads as the base; the next state is still **synthetically perturbed**. This validates the input distribution/shape but not genuine cross-step temporal statistics.
3. **Actual next-step test (not yet available):** `--capture-a STEP_A.pt --capture-b STEP_B.pt` takes matching, ordered, distinct heads from the same block at consecutive steps. This is the only supported mode labeled `paired_real_QKV_captures`. The program verifies head IDs and shapes. We do not claim such a pair has been collected. No paid inference or long H3 workflow is triggered by this script.
4. If real-step chosen-leaf churn is low **and** a quality-preserving update or certified-bound/safe fallback can be found, implement a compact **GPU sparse updater** with measured end-to-end benefit. If most leaves change, or exact bounds require essentially the whole recomputation, record **NO-GO** and do not pursue.

### One Windows worktree, one result JSON

Use the *existing* HTRC repo root; **do not assume `D:\ComfyNodes` exists** or reuse the separate ComfyNodes repo. The following creates an isolated worktree and leaves the verified Direct-Flow checkout untouched:

```powershell
$src = 'D:\ComfyNodes-flat-split-20261111'
$wt = 'D:\HTRC-temporal-leaf-delta-20261112'
git -C $src fetch htrc
if ($LASTEXITCODE -ne 0) { throw 'HTRC fetch FAILED' }
if (Test-Path $wt) { throw "Destination exists: $wt" }
git -C $src worktree add --detach $wt htrc/experiment/flat-temporal-leaf-delta-20261112
if ($LASTEXITCODE -ne 0) { throw 'HTRC worktree FAILED' }
Push-Location $wt
try {
    & $py -m unittest discover -s tools -p test_h3_flat_temporal_leaf_delta_contract.py -v
    if ($LASTEXITCODE -ne 0) { throw 'CPU contract FAILED' }
    & $py -B tools\benchmark_h3_flat_temporal_leaf_delta_ab.py --heads 4 --tokens 18723 --centers 16 --rounds 5 --warmups 2
    if ($LASTEXITCODE -ne 0) { throw 'GPU research probe FAILED: inspect JSON' }
} finally { Pop-Location }
```

**Single report:** `D:\ComfyOutput\H3\math_probe\flat_temporal_leaf_delta_ab.json`. User may supply `--capture-a` to use their previously trusted QKV source and `--capture-b` for consecutive real captures. No separate JSON files, no `.pt` dumps, no model replacement.

## Acceptance and test status

- **CPU fixture:** seven CPU-only unittest checks for formula/shape, unchanged-state algebra, mean-delta, exact dirty-coverage, ordered position patch and numerical fail-closed labeling. Run in the authoring CPU container; **7/7 PASS**.
- **GPU native HTRC full stats, selected-leaf difference, Tile+Merge A/B:** **UNRUN** on user's RTX 5070 Ti.
- **Real paired denoising-step QKV and full 56-head model/resident benchmark:** **UNRUN / not available**.
- **Live H3/production:** unchanged; no integration, no quality coefficient, no global BEST promotion.

**Decision gate:** a `completed=true` synthetic feasibility JSON is **not** a performance or quality PASS. Report any `strict_bitwise_identity_pass=false` as a numerical discrepancy; decide next step only from observed output errors, IDX stability, update overhead and especially **real consecutive step** behavior.


## RTX 5070 Ti GPU result — 2026-11-12 (recorded, experiment NO-GO)

One uploaded combined report was completed with `errors=[]` and real
captured H3 QKV for **state A**, while **state B is synthetic** in each
scenario. No new H3 weights were loaded and no actual consecutive
denoising-step comparison or future prediction was run.
Evidence: [single combined JSON](evidence/flat_temporal_leaf_delta_ab_20261112.json).

| Synthetic B scenario | Fresh GPU ms | Observed mean-delta GPU ms | Delta slowdown | Exact reusable centre-leaf pairs | Selected-ID memberships replaced | Ordered IDX positions changed | Final rel-L2 / max-abs |
|---|---:|---:|---:|---:|---:|---:|---|
| dense_drift | 0.704640 | 11.275104 | 16.00x | 0.00% | 1.5137% | 48.9624% | 0.001941 / 0.52035 |
| sparse_drift | 0.423136 | 5.718400 | 13.51x | 89.04% | 0.1648% | 9.0576% | 0.000472 / 0.12281 |
| uniform_leaf_drift | 0.425440 | 4.877824 | 11.47x | 0.00% | 1.6235% | 50.1892% | 0.001493 / 0.20760 |

All three: `strict_bitwise_identity_pass=false`,
`eligible_for_quality_preserving_integration=false`, `all_cases_bit_exact=false`.
Timing covers post-MU/post-K8 stats+selection+Tile+Merge, NOT
projections, K8 preparation, model residency or full H3. Observed
mean-delta is an unoptimized approximate probe (not a Triton sparse
update), so this is a **NO-GO for this specific method**, not a
disproof of every temporal structure-prediction approach.

**Promising measured structure signal only:** sparse_drift had
45/64 centres with identical selected-leaf membership, with only 27
removal + 27 addition memberships (oracle membership patch 216 bytes
versus full 65,536-byte IDX for this small window). But the exact
ordered IDX position patch required 11,872 bytes (9.06% positions);
dense/uniform ordered patches replaced nearly half the positions.
Patch computation itself requires a fresh IDX in the present probe;
there is no selection-computation saving or online predictor.

For 56 heads and all query windows in ONE H3 block, persistent
G/VB/LZ/SG would cost an estimated **2.369 GiB**, whereas IDX alone
is ~**2.488 MiB** (excluding further cache/memory). Persisting full
summary state across 49 blocks or steps is not a 16 GiB solution.

**Owner research direction preserved:** test temporal leaf-set stability
on TWO ACTUAL consecutive post-RoPE H3 QKV captures of the SAME block
and prompt (same geometry), before proposing an online sparse-update
kernel or future-state predictor. The measured old-state/synthetic-B
oracle is not such a test. Do not rerun the rejected observed
mean-leaf-delta candidate without a new exactness mechanism.
No production integration, no quality coefficient, no BEST promotion.
