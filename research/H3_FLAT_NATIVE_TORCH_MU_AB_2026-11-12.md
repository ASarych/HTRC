# Flat Direct Q — separate ATen native-strided MU alternative

Date: 2026-11-12. Base \`aef423699ac49e4ca709b3278dfe4dc7a45e2a29\`.
Research branch \`experiment/flat-direct-qkv-kernels-20261112\`; no default change.

## Input: passed local stage diagnosis

Attached \`flat_direct_qkv_ab(3).json\` archived as
[\`flat_direct_qkv_stage_diagnostic_pass_20261112.json\`](evidence/flat_direct_qkv_stage_diagnostic_pass_20261112.json).
Report \`completed=true\`, CUDA RTX 5070 Ti, 56 distinct synthetic BF16 source heads, fixed fast 256 exact leaves / group1, 7 samples and 2 warmups per arm. **This is timing only; numerical parity was not repeated.**

For **11 heads, full 1,024-row window**, GPU stream stage-interval medians (not pure GPU kernel time):

| stage | baseline | Direct Q | delta |
|---|---:|---:|---:|
| Q gather/native MU | 0.590336 | 0.917984 | +0.327648 |
| Leaf Stats | 0.252448 | 0.244736 | -0.007712 |
| Rowmax + TopK exclusion | 0.188416 | 0.089984 | -0.098432 |
| Tile | 0.371232 | 0.358944 | -0.012288 |
| Merge tail | 0.033344 | 0.036608 | +0.003264 |

Direct Q saves ~0.118 ms in leaf selection/Tile/Stats on this grouping but increases Q/MU stage interval by ~0.328 ms, causing the ~0.21 ms interval sum regression. For 1 and 4 heads the larger Rowmax saving can dominate; at 11 heads it cannot. Stage intervals include CPU enqueue gaps, and are not a causal attribution to GPU arithmetic.

## New opt-in candidate

Use \`query_staging="native_torch_mu"\`: same native post-RoPE BF16 strided Q view as \`native_direct\`, but calculate FP32 center means with \`torch.mean(..., dtype=torch.float32)\` instead of the standalone Triton \`_direct_q_mu_kernel\`. It does not allocate a full FP32 Q tensor; it only allocates FP32 MU. The final partial center uses mean of true rows, as the existing control does. Downstream native-Q rowmax, Tile and exact-tail Merge kernels unchanged.

- Keep \`gather\` default, \`view_copy\`, \`native_direct\` (Triton MU) and \`native_torch_mu\` independent. No hidden fallback.
- Benchmark now supports six combinations: baseline, direct_q (Triton MU), direct_kv, direct_qkv (Triton MU), direct_q_torch_mu, direct_qkv_torch_mu. \`--diagnose-only\` measures 1/4/11 heads and both full/short windows. One JSON per run.
- \`--diagnose-only\` is timing only. For new candidate numerical acceptance, run default full A/B (4 and 56 genuinely distinct synthetic heads) and require zero errors, same IDX/EXCL and matching final output before considering performance.
- Exact leaves=256, group=1, no quality coefficient, no change to K8/KS, no production release. CUDA on this commit and real checkpoint/video parity **UNRUN**.

## Run in existing tested CUDA + Triton environment

\`\`\`powershell
cd D:\ComfyNodes-flat-split-20261111
git fetch htrc
git switch --detach htrc/experiment/flat-direct-qkv-kernels-20261112
& $py -B tools\benchmark_h3_flat_direct_qkv_ab.py --head-cases 4,56 --rounds 2 --warmups 1
\`\`\`

Use the **same** \`$py\` known to import torch+Triton from the previous PASS. One output: \`D:\ComfyOutput\H3\math_probe\flat_direct_qkv_ab.json\`. If and only if numerical parity PASSes, use \`--diagnose-only --rounds 7 --warmups 2\` to investigate stage savings, which will overwrite the same JSON. Save the first output before running diagnostics if both results are needed.

Do not interpret an isolated synthetic benchmark as end-to-end H3 model-resident speed.
