# HTRC Flat Direct QKV — successful local A/B and next stage diagnostic

Date: 2026-11-12. Research branch \`experiment/flat-direct-qkv-kernels-20261112\`.

## Pinned RTX 5070 Ti evidence

Raw user-uploaded 4-/56-head report retained as
[\`flat_direct_qkv_ab_pass_20261112.json\`](evidence/flat_direct_qkv_ab_pass_20261112.json).

- \`completed=true\`, \`errors=[]\`; fixed \`fast\` math: 256 exact leaves and center group 1, no quality coefficient.
- 4 genuinely distinct synthetic heads: 456 state comparisons, 9 final output comparisons, 1 compact-vs-legacy tail test: PASS. 1.009360 ms baseline vs 0.972864 ms Direct QKV, **1.037514x** throughput.
- 56 genuinely distinct synthetic heads in groups of 11,11,11,11,11,1: 2,736 state comparisons, 54 final output comparisons, 6 compact-vs-legacy tail checks: PASS.
- 56-head sum of per-group GPU event medians (this is **not** an end-to-end model/video wall clock):
  - baseline 8.618320 ms
  - Direct Q 9.410160 ms (**9.19% more time**)
  - Direct K/V 8.614304 ms (effectively neutral)
  - Direct QKV 9.375168 ms (**8.78% more time**, 8.07% less throughput).
- \`IDX\`, \`EXCL\`, direct K8/KS exact for all checked cases; 56-head MU max_abs 2.98e-8, G int8 difference at most 1 LSB, VB 0.001953125, final max_abs 3.72529e-9 / rel_L2 3.55e-9.
- Additional BF16 K/V cache on each 11-head group: 132,612,480 bytes packed to 27,181,440 bytes native (**79.503% storage reduction**). This excludes original model Q/K/V and is NOT overall VRAM reduction. Cold cache-build and JIT times are noisy; warm repeated-group native cache-build can be faster, but do not extrapolate.
- Full and last partial query window tested; last 291 Q rows derive 5 active tiles out of 16 for this shape, and the compact baseline is bitwise equal to legacy full-padded baseline on real rows. Tile count is derived dynamically from window capacity/tile rows/center grouping, not hardcoded.

Decision: \`Direct K/V = PASS_NUMERIC / NEUTRAL_COMPUTE / POSITIVE_VRAM\`. Preserve as an opt-in research alternative. \`Direct Q and Direct QKV = PASS_NUMERIC / NO_GO_PERFORMANCE\` for the 56-head benchmark. None promoted to production. Model-resident video and 16 GiB checkpoint headroom remain UNRUN.

## New isolated stage profile, one JSON

The original benchmark has an opt-in \`--diagnose-only\` path, without a second report. It uses one 56-distinct-head BF16 capture but probes separately 1-, 4- and 11-head CUDA groups, full and short query windows, and all four existing arms.

Each measured call emits non-synchronizing CUDA event markers for query-gather-or-native-MU, KV cache hit, leaf_stats, rowmax_topk_exclusion, Tile and Merge. Per-stage medians and synchronized CPU wall medians are saved. Stage event intervals **include GPU idle time waiting for the host**, and event overhead; they are not pure kernel measurements.

This mode does NOT repeat full numeric parity and its \`completed=true\` means only diagnostic measurement finished. \`numerical_parity=NOT_REPEATED_IN_STAGE_DIAGNOSTIC\`. The preceding saved full A/B remains the numerical reference.

PowerShell from the existing correct HTRC/ComfyUI checkout with a proven Python CUDA+Triton runtime (use the same interpreter that produced the PASS report):

\`\`\`powershell
git fetch htrc
git switch --detach htrc/experiment/flat-direct-qkv-kernels-20261112
& $py -B tools\benchmark_h3_flat_direct_qkv_ab.py --diagnose-only --rounds 7 --warmups 2
\`\`\`

One result: \`D:\ComfyOutput\H3\math_probe\flat_direct_qkv_ab.json\`. Upload that JSON for Q stage bottleneck analysis. Do not activate any default Direct Q mode based on the present regression.

## Status

Uploaded GPU A/B PASS; new stage-profile code **CPU AST/UNRAN locally in this workspace, GPU UNRUN**, production/main unchanged.
