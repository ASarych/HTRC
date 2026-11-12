# H3 Flat Q preparation: shape-matched microprofile (2026-11-12)

## Result and scope

This is a synthetic GPU microprofile of the HTRC Flat query-preparation path, using the Priestess video attention shape: 18,723 tokens, 56 heads, head dimension 128, BF16 Q/K/V, 16 query tiles of 64 rows, 256 exact leaves, and a maximum group size of 11 heads. It ran 114 preparation calls (19 query windows × 6 head groups) on an RTX 5070 Ti. No model checkpoint, text encoder, denoising loop, video VAE, or video decoder was loaded.

Warm per-window/per-head-group stage medians:

| Stage | Median |
| --- | ---: |
| Q gather, cast to FP32, and MU reduction | 0.338 ms |
| K/V cache validation | 0.030 ms |
| Leaf statistics | 0.396 ms |
| Rowmax selection, top-k, and exclusion mask | 0.368 ms |
| **Sum of stage medians** | **1.132 ms** |

At 114 window/group passes, the stage-median sum is about **129 ms per full attention call**. This is derived from separate microprofile calls; it is not a measured end-to-end attention-call time. If the same cost repeated for 49 blocks across six denoising steps, the arithmetic extrapolation is about **38 seconds** of query preparation. That extrapolation is not a video benchmark result.

K/V cache build median was 0.406 ms per group and workspace allocation median was 0.088 ms per group. The profile included Triton JIT outliers in aggregate wall time; use the warm medians above for comparison. Stage timers synchronize CUDA and add measurement overhead.

## Interpretation limits

Q/K/V values were zero-filled synthetic tensors. Shapes, dtypes, cache use, workspace, Triton grids, and preparation functions match the measured path, but real tensor values, concurrent model residency, allocator pressure, and full-runtime overlap do not. The profile calls `prepare_batched_head_states`; it does not run the production `flat_attention_bthd` path, Tile/Merge, or a real denoising step. It therefore isolates preparation cost and does not prove end-to-end speedup or output quality.

## Source provenance

The HTRC checkout was based at `1743dd8e442d9caf8753eb1be3fd83ab46a0b50f`. At measurement time, these files had local, uncommitted changes and the profile imported that working-tree state:

- `h3/htrc_flat_head_batch.py` SHA-256 `0e028e0906bbd7e87dafca2e36ae051c19648682c967eed028f1ac1ba560c105`
- `h3/htrc_flat_live_backend.py` SHA-256 `566f315430f085f2db26182b498aec8e2f5563120bbd272cad061a767f9afbd1`
- `h3/htrc_flat_model_shadow.py` SHA-256 `74f232a4de4de330dcbac6e6cba08600809998905c9361b71f0fbf8252794b5c`

This evidence commit contains the Markdown interpretation and raw JSON only; it does not contain those code edits or the local profiling script. Checking out this branch alone will not reproduce the exact measured source state.

The detailed measured fields are preserved in [`H3_FLAT_QPREP_SHAPE_PROFILE_2026-11-12.json`](H3_FLAT_QPREP_SHAPE_PROFILE_2026-11-12.json).
