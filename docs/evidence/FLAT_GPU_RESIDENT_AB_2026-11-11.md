# GPU-resident Flat HTRC grouped QKV preparation A/B — 2026-11-11

Evidence: user-run `flat_full_prep_gpu_qkv_ab_20261111-211712.json`
on NVIDIA RTX 5070 Ti (16 GiB). Candidate exact SHA
`84c6107c4f80ade7741ca435ed3038dd7fa5bdcc`, pinned
ComfyNodes BEST `e47cc7a310936547ab6c7d97bd07526eb6a135fe`
(source blob-equivalent HTRC `a8fe43e7aab8d9f951af7b4cba8fbd54f7a61e5c`).

Same GPU-resident captured QKV fed pinned per-head and batched grouped
path; 56 **logical** heads are 14 repeats of native 1/16/27/55,
not 56 distinct model heads. Mode fast; 16x64 = 1024 query rows
per head; 40,752 aligned K plus seven exact tail keys; groups
12+12+12+12+8. The GPU capture preload (119.411MiB) is outside
timings but remains allocated. No model checkpoint or denoise/video.
All values are synchronized host wall, **sum of five group medians**,
not an actual one-call 56-head model step.

| Work | Pinned BEST (ms) | Batched Triton (ms) | Ratio |
|---|---:|---:|---:|
| Preparation | 61.10760 | 14.58185 | 4.19066x |
| Tile + exact tail Merge | 16.85035 | 6.28880 | 2.67942x |
| Full grouped path | 77.97285 | 20.84815 | **3.74004x** |

- All five grouped final outputs bit-exact: rel-L2=0, max-abs=0,
  finite PASS. Four independent full original Flat head oracles PASS.
- Fused K8 exactly matches PyTorch: zero INT8 mismatches over 40,752
  rows for captured head 1; FP32 scale max absolute difference zero.
- No OOM/errors. Per-H12 grouped full speedups: 3.774x, 3.819x,
  3.779x, 3.834x; H8 3.409x.
- H12 incremental PyTorch peak allocated (median): 453.3125MiB
  pinned vs 492.0942MiB candidate (+38.7817MiB).
- Diagnostic-only GPU K/V gather in H12 preflight ~4.7–5.0ms;
  Triton K8 ~0.4–0.46ms, but the synchronized diagnostic wall
  is **not** the measured ABBA wall.

**Decision:** positive GPU-resident research result, NOT
production/cumulative BEST. It does not contradict earlier *CPU*
QKV A/B (509.192ms pinned, 545.401ms candidate, 0.93361x), since
CPU gather and H2D were inside those measurements for both paths.

Next: isolate batch-quant torch_gpu vs triton_gpu at the same
GPU-resident QKV; test bigger query chunks (e.g. 64 tiles),
capture genuinely different 56 heads, run with actual H3 checkpoint
resident and finally test complete denoising/video. Separate
query-cache branch `b75f517d80cee94cb867f35c07700e76412be237`
remains GPU UNRUN and must not be silently merged.
