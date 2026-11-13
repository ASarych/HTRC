# HTRC Flat research: in-Tile fused Vbar, old modes retained

Date: 2026-11-13. Base:
ASarych/HTRC experiment/flat-long-context-streamed-vb-20261113
@ 88a17eb35c9851adab9cc4bcd008d3c1ca9e3f05.
This is a separate research slice. Do not merge into main or change
ComfyNodes as part of this stage.

## Mode selection and non-regression policy

The old eager Flat and old proven Graph materialized-VB versions remain
the default and retain their compiled arithmetic. The streaming split
experiment also stays available. The new fused mode is OFF by default.

| APL_HTRC_FLAT_DISPATCH | APL_HTRC_DIRECTFLOW_VB_MODE | Selection |
|---|---|---|
| eager | irrelevant | Original eager Flat; unchanged |
| graph | unset or materialized | Original GPU Graph; default, measured |
| graph | split_stream | Prior experimental four-split producer/consumer |
| graph | fused_vb | New experimental direct VB-in-Tile specialization |
| graph | other | ValueError / fail closed, no silent fallback |

The mode participates in the Graph geometry/session key. The existing
first-call exact eager parity gate, finite graph pool memory cap and
cross-stream completion event remain. Only fast (256 exact leaves) and
balanced (512 exact leaves), centre group=1 and BF16 sources are accepted;
no Q quality coefficient and no shortcuts to exact near-field or tail.

## Source-code slice

- h3/htrc_flat_triton.py:
  opt-in FUSED_VB=1 specialization in existing _tile_kernel, keeping
  FUSED_VB=0 as default. New _fused_vbar_one_leaf builds a 16x16
  tensor-core computation from the real per-centre FP32 MU and 16 BF16
  K token rows; exponentiate and normalize leaf weights as original
  _stats_kernel; multiply those weights with 16 BF16 V token rows.
  The first tensor-core output row is taken as BF16 VB for one leaf,
  retained in register values (not stored). Exact far-leaf scoring,
  EXCL handling and Tile online-softmax/PV remain unchanged.
  This is correctness-first: the 16-row broadcast may be expensive,
  and the first CUDA numeric gate may discover rounding differences.
- h3/htrc_flat_head_batch.py:
  explicit launch_prepared_tile_batch_fused_vb entrypoint; only this
  path provides the real native BF16 K pointer/strides to _tile_kernel.
  The old launch_prepared_tile_batch defaults to fused_vb=False.
  SCORE_ONLY stats computes G/LZ/SG; the original selection produces
  the same IDX/EXCL. Full VB is a one-element placeholder.
- h3/htrc_flat_model_shadow.py:
  fused mode selects a one-element VB placeholder and no split scratch.
- h3/htrc_flat_directflow_live.py:
  explicit APL_HTRC_DIRECTFLOW_VB_MODE=fused_vb routing. No edits to
  h3/htrc_flat_live_backend.py or ComfyNodes; production remains eager.

## Very important: performance hypothesis may be NEGATIVE

Recomputing a whole leaf's native K and V costs two 16x128 BF16
matrices = 8192 bytes logically read for each centre/leaf pair;
avoiding the global BF16 VB write and read saves 2 x 256 bytes/pair.
The naive read/write tradeoff is therefore **up to 16x more source
bytes than the avoided VB traffic**, even before extra tensor-core
dots, warp occupancy or L2 cache effects. Relative to just the
avoided VB *write*, source reads are 32x.

At N=82,000 tokens, 56 heads, group1:
- C=1282 centres, L=5125 complete key leaves.
- P=56*C*L=367,934,000 centre/leaf pairs over a full attention call.
- Original VB write ~87.7 GiB logical, original VB read another
  ~87.7 GiB (one per centre/leaf); both may be satisfied partly by
  cache, not necessarily DRAM.
- Naive fused source K+V reads ~2807 GiB logical before any cache reuse.
  They are NOT simultaneous VRAM and not a measured DRAM transaction count.

The fused mode therefore might be SLOWER than materialized and
split_stream. It exists to establish an actual numerical/throughput
point of reference, not because it is presumed best. A truly competitive
future fused kernel must reuse leaf K/V across MANY centres (ideally
a 16-centre tensor-core batch) and avoid repeated global reads, or
combine selection/far with locality-preserving scheduling. Do NOT
run long H3 videos on this implementation without a positive smaller
kernel/full-attention A/B first.

## One-shot A/B and fail-closed checks

CPU-only source/AST gates (13 tests) in
tools/test_h3_flat_fused_vb_contract.py.

GPU full-attention experiment in
tools/benchmark_h3_flat_fused_vb_ab.py:
- two completely different post-RoPE-style BF16 Q/K/V banks;
- two modes: materialized Graph vs fused_vb Graph;
- on bank A each first full call independently gates against native
  eager HTRC for every head group;
- on bank B independently compare each Graph variant to native eager;
- then enforce full-output bitwise equality across modes for each
  bank, no stale QKV, and a warm full-call timing including K8,
  staging, far Tile, near, tail, merge and output writes;
- one JSON: D:\ComfyOutput\H3\math_probe\flat_fused_vb_ab.json.
- no model checkpoint or video is loaded.

Example on an isolated HTRC worktree:

    $src = "D:\ComfyNodes-flat-split-20261111"
    $wt = "D:\HTRC-fused-vb-20261113"
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    if (Test-Path $wt) { throw "Refuse to overwrite existing worktree" }
    git -C $src fetch htrc
    if ($LASTEXITCODE -ne 0) { throw "Git fetch FAILED" }
    git -C $src worktree add --detach $wt htrc/experiment/flat-long-context-fused-vb-20261113
    if ($LASTEXITCODE -ne 0) { throw "Worktree add FAILED" }
    Push-Location $wt
    try {
        & $py -m unittest discover -s tools -p test_h3_flat_fused_vb_contract.py -v
        if ($LASTEXITCODE -ne 0) { throw "CPU contracts FAILED" }
        & $py -B tools\benchmark_h3_flat_fused_vb_ab.py --tokens 15275 --heads 4 --mode fast --tiles 16 --rounds 3
        if ($LASTEXITCODE -ne 0) { throw "GPU full-output A/B FAILED" }
    } finally { Pop-Location }

CUDA Graph capture, GPU numerical gate and profiler timing are UNRUN
until executed by a CUDA environment. This implementation is not a
verified faster attention or Sol replacement.

## Follow-up if the expected bandwidth regression occurs

Do not hand-tune this very expensive one-leaf-per-tile GPU kernel.
Prefer redesign: stage selection-only scores, then schedule one
multi-centre cooperative far producer/consumer that reuses K/V across
centres (or a score+far fused kernel with cache-aware tiling), preserving
selection and exact softmax reductions. Evaluate the 40k/82k shape
before a Full HD 223k attempt. Keep original BEST intact.
