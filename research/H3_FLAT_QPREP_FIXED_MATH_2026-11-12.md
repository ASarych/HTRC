# HTRC Flat Q preparation — fixed math, NO quality coefficient

Date: 2026-11-12. Owner decision: **do not use any Q speed/quality
coefficient**. No Q=0/Q=0.1/Q=0.3 experiment or leaf-count reduction
is acceptable as a substitute for engineering a faster implementation.
The reported Q=0.3 image was rejected for quality.

## Recovered exact source

The evidence repo branch
experiment/flat-qprep-profile-20261112@c21fbb73b14d2de4f01f7374649da41d42d15218
now contains all three measured runtime source files AND the actual
tools/profile_h3_flat_qprep_shape.py script. Verified SHA-256 values:

| Uploaded file | Original SHA-256 | Source line endings | Verification |
|---|---|---|---|
| h3/htrc_flat_head_batch.py | 0e028e0906bbd7e87dafca2e36ae051c19648682c967eed028f1ac1ba560c105 | Windows CRLF | exact match |
| h3/htrc_flat_live_backend.py | 566f315430f085f2db26182b498aec8e2f5563120bbd272cad061a767f9afbd1 | Windows CRLF | exact match |
| h3/htrc_flat_model_shadow.py | 74f232a4de4de330dcbac6e6cba08600809998905c9361b71f0fbf8252794b5c | Windows CRLF | exact match |
| tools/profile_h3_flat_qprep_shape.py | 87601a16394010b5e98ece1527e846bc991678498e8267b45e1c018ad5ce3e98 | LF | exact match |

The GitHub connector presents normalized LF strings; rewriting
three recovered Python texts with CRLF reproduces their local
SHA-256 exactly. The source files themselves therefore match
the report; the earlier provenance blocker is resolved.
This new optimization starts from the uploaded evidence HEAD,
not the older HTRC branch.

## Stop changing quality settings to obtain speed

Live h3/htrc_flat_live_backend.py no longer has a \`quality\`
argument or dynamic \`cfg["quality"]\`, \`cfg["kern"]["group"]\`
and \`cfg["exact_leaves"]\` overrides derived from Q.
The obsolete h3/htrc_flat_q.py research helper is not
imported by the actual Flat GPU runtime; keep it only as
historical experiment evidence, never enable it in this path.

Selection retains the chosen fixed existing Flat mode
(\`fast\` = 256 exact leaves and group=1; \`balanced\` =
512 exact leaves and group=1). This does NOT claim either
approximate Flat configuration has mathematically identical
outputs to dense attention or user-approved video quality.
It only ensures **this optimization** does not degrade the
same fixed baseline. Acceptance still requires real video
fidelity/temporal checks compared with an acceptable reference.

## First isolated optimization: remove redundant Q packing

Observed real warmed Flat attention at 18,723 tokens/56
distinct heads: ~169.08 ms total wall; the instrumented
\`query_stats_selection\` interval is ~112.17 ms. The
synthetic shape-matched profile breaks preparation into
Q gather/FP32/MU ~0.338 ms, leaf stats ~0.396 ms,
rowmax/topk/exclusion ~0.368 ms per group-window
(medians, not additive wall proof).

Old path: \`index_select\` native heads, permute, make
BF16 contiguous, then FP32 cast (multiple allocations/copies).
New opt-in \`query_staging="view_copy"\`: only if source
native heads occupy consecutive positions, slice/permute
as a VIEW of the original CUDA tensor, create one FP32 Q
destination and copy/cast the native strided Q view into it.
This is a transport change ONLY. MU reduction, K8,
G/VB/LZ/SG, rowmax/topk/EXCL, Tile+R32 Merge and the
fixed \`exact_leaves\` remain identical.

- Default \`query_staging="gather"\` preserves the measured
  source for all model runs until the numerical and speed gate.
- If native CUDA indices are not consecutive, the experimental
  path raises an explicit error; it cannot produce a silent
  proxy or a different head order.
- Tool \`tools/benchmark_h3_flat_q_input_pack_ab.py\`
  tests one combined 4-head + 56-distinct-head experiment
  on nonzero synthetic native-strided Q/K/V at H3 shape.
- Every window/group checks **bitwise equality** of
  Q, MU, G, VB, LZ, SG, IDX and EXCL.
- First/middle/last windows additionally gate the full
  Tile+Merge output bitwise. Any mismatch fails closed.
- Paired GPU event measurements compare complete Q-prep
  before video inference. This is a synthetic source,
  **not** a quality verdict from a real H3 checkpoint.
- One final JSON, not one file per case.

## One Windows command, one JSON (no quality coefficient)

PowerShell in the existing HTRC worktree:

    cd D:\ComfyNodes-flat-split-20261111
    git fetch htrc
    git switch --detach htrc/experiment/flat-qprep-fixed-math-20261112
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat_head_batch.py h3\htrc_flat_live_backend.py tools\benchmark_h3_flat_q_input_pack_ab.py tools\test_h3_flat_qprep_fixed_math.py
    if ($LASTEXITCODE -ne 0) { throw "Python compile FAILED" }
    & $py -m unittest discover -s tools -p test_h3_flat_qprep_fixed_math.py
    if ($LASTEXITCODE -ne 0) { throw "Contract tests FAILED" }
    & $py -B tools\benchmark_h3_flat_q_input_pack_ab.py --tokens 18723 --heads 56 --head-cases 4,56 --group-cap 11 --mode fast --tiles 16 --rounds 2 --warmups 1
    if ($LASTEXITCODE -ne 0) { throw "Q input pack A/B or parity FAILED" }

One persistent output:
D:\ComfyOutput\H3\math_probe\flat_qprep_fixed_math_ab.json
with both embedded head cases, every-window exact-state
checks, representative final-output checks and GPU event
medians. No separate temporary JSONs.

Current CPU AST and GPU tests are UNRUN in this environment.
Do not switch live H3 from gather to view_copy until
GPU bit-equality and positive latency are observed.

## Next optimization decision after the gate

If native direct Q packing is exact and helps, integrate
as the same fixed-math input layout (without Q coefficient)
and run ONE matched real-H3 fixed-preset test with the
accepted visual reference. If neutral/negative, discard.

The next attention bottleneck is leaf stats and
rowmax/topk/exclusion plus repeated GPU launches. Consider
persistent allocation, launch fusion and sorted top-k
alternatives only if they reproduce the SAME top-k IDs,
exclusions and final output; don't reduce leaves or group
centres to gain speed. Validate real model VRAM/headroom
on RTX 5070 Ti, denoise-step runtime and video consistency.
Preserve full source SHA, exact commit, one combined report
per coherent experiment. BEST/production unchanged.
