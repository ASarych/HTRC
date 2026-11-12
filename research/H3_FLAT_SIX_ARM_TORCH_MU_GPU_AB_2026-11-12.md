# Flat HTRC six-arm RTX 5070 Ti GPU PASS — Torch MU is accurate, not yet fast at 56 heads

Date: 2026-11-12. Tested on `f3eb44b8f0276ea4b437499ba662b9b6a0f2bb89`, research branch `experiment/flat-direct-qkv-kernels-20261112`.
Raw original user report: [flat_direct_qkv_six_arm_gpu_pass_20261112.json](evidence/flat_direct_qkv_six_arm_gpu_pass_20261112.json).

## Validation (one synthetic BF16 Q/K/V capture, NOT real model)

- `completed=true`, six arms, no `errors`, fixed `fast` 256 exact leaves, group1, no Q-quality coefficient, 18,723 tokens, 19 windows (short last: 291 Q rows / 5 dynamic tiles).
- **5320** state comparisons, **105** final outputs, **7** short-window vs old full-padded baseline. `IDX`, `EXCL`, quantized K and scales identical on both 4/56 cases. `native_torch_mu` intermediates Q/MU/G/VB/LZ/SG are bitwise equal to baseline in this run; 56-head final maximal output absolute deviation across all arms `3.725290298461914e-9` (not all arms individually).
- GPU event *sum of group medians* (NOT one model call, not entire video; cache-build costs excluded):

| Mode | 4 heads (ms) | 56 heads (ms) | 56-head throughput vs baseline |
|---|---:|---:|---:|
| `baseline` | 1.056480 | 7.416256 | 1.000000x |
| `direct_q` | 1.001376 | 8.246400 | 0.899333x |
| `direct_kv` | 1.021456 | 7.447984 | 0.995740x |
| `direct_qkv` | 0.994144 | 8.254368 | 0.898464x |
| `direct_q_torch_mu` | 0.978896 | 8.119296 | 0.913411x |
| `direct_qkv_torch_mu` | 0.970192 | 8.049008 | 0.921388x |

- Torch MU reduced the Direct QKV grouped event total from `8.254368` to `8.049008` ms (**2.49% less time**), but is still **8.53% more time** than baseline at 56 native heads. On 4 heads it is `1.088939x` faster.
- Per-11-head K/V cache: packed `132612480` bytes, native `27181440` bytes: **79.50% smaller additional cache**, excluding original model-owned BF16 K/V. Compute-only Direct K/V at 56 heads is effectively neutral (`0.995740x`).
- Cold build timings include JIT and vary significantly; cannot extrapolate from those numbers to a real 49-block, multi-step video.

## Decisions

1. Preserve Direct K/V as opt-in memory-saving path. `gather` remains the default until model-resident quality/VRAM pass.
2. Do NOT promote Direct Q or Direct QKV to production. Torch MU improves their cost but does not beat staged Q for grouped 56-head case.
3. Next engineering slice: experimentally lower Python dispatch overhead without changing numerical kernels. Benchmark one fixed Q/K/V geometry with eager vs CUDA Graph replay and explicit capture cost, peak/extra memory and numeric identity. This is a **dispatch feasibility probe** only: fixed pointers cannot be reused for new model layers/denoise steps by assumption.
4. Do not use runtime quality coefficients or reduce number of exact leaves/centers. Keep distinct real heads, one JSON per coherent A/B, and no silent fallback.

## Status

Six-arm GPU numeric A/B: **PASS_NUMERIC**; Torch MU 56-head performance: **NO_GO_PERFORMANCE**. Direct-Flow Graph: **NOT_IMPLEMENTED / UNRUN** at this evidence HEAD. Full model resident H3/video quality: **UNRUN**.
