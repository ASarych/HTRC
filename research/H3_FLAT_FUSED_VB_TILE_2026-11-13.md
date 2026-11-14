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


## 2026-11-14 — RTX 5070 Ti strict GPU failure and diagnostic-only follow-up

The user executed the existing exact-HEAD fused_vb short A/B at
05de218c1e70ecec93b091fc572494d4ba1c7da2:
15,275 tokens, four distinct BF16 heads, Balanced/512, Tile32.
All **13 CPU source contracts passed**. The Graph eager/strict numerical
gate then stopped before any warmed performance measurement:

    AssertionError: Direct-Flow real H3 mismatch at heads [0,4):
    max_abs=0.00024414062

The benchmark wrote ONE flat_fused_vb_ab.json with
status=FAIL_CLOSED. The user supplied the terminal traceback,
not the full JSON payload. The observed maximum difference is
exactly 2^-12 in BF16 output units, but its distribution,
source stage, fraction of affected elements and actual runtime
speed are **NOT measured**. No tolerance or production behavior
has been relaxed. Do not treat a small maximum difference as
an acceptable quality change.

The existing _fused_vbar_one_leaf duplicates the centre MU into
16 rows to match the reference stats tensor-core geometry, then
reduces to one row. Even though the intended equations match
_stats_kernel, identical final BF16 rounding is not guaranteed
by source similarity; neither a rounding-only explanation nor
a kernel bug has yet been established.

### New single-JSON opt-in numeric diagnostic

tools/benchmark_h3_flat_fused_vb_ab.py --diagnose-numeric runs
the EXACT SAME first synthetic Q/K/V bank and fixed short geometry.
It uses Graph compare_only=True **only inside the isolated
diagnostic** for materialized and fused_vb, then independently
computes the untouched eager Flat output. Complete final BF16
tensors are compared: materialized vs eager, fused vs eager,
fused vs materialized. Record total/mismatched elements,
fraction mismatched, max-absolute error, relative L2,
per-head mismatch counts and the eight largest mismatching
token/head/feature values.

A materialized-vs-eager mismatch is an immediate error and
FAIL_CLOSED. Otherwise a fused-vs-eager mismatch yields
DIAG_NUMERIC_MISMATCH, numerical_gate=FAIL; an exact
match yields DIAG_MATCH_NOT_ACCEPTED, not PASS_NUMERIC.
The mode records performance_measured=false,
promotion_to_real_h3=PROHIBITED_DIAGNOSTIC_ONLY.
No model, VAE or video. No stage-local kernel oracle yet:
this diagnostic localizes the **output** error and whether
the original control is stable, not a specific tensor-core
operation.

The same safe PowerShell exact-HEAD runner supports
-Candidate fused_vb -DiagnoseNumeric -Rounds 5. It saves
ONE report at
D:\ComfyOutput\H3\math_probe\flat_fused_vb_ab.json,
overwriting the prior fail-only JSON. CPU source contracts
run before the GPU probe. Only the diagnostic call itself
uses compare_only; real/ordinary first-call bit-exact parity
and fixed STATS_BT=16 remain unchanged.

Current status: strict GPU gate **FAIL_NUMERIC**, GPU speed
**UNRUN**, new diagnostic GPU **UNRUN**, diagnostic source
contract **UNRUN**. Do not rerun the previous strict A/B
until mismatch is diagnosed. Do not promote or run H3 video.


## 2026-11-14 — completed numeric diagnostics; first bounded reduction-geometry correction

User-provided flat_fused_vb_ab.json from the diagnostic tool at
d0494448b374d42581c50883a31ceb4521f9fdbc:
DIAG_NUMERIC_MISMATCH, completed=true, numerical_gate=FAIL.
Synthetic four native BF16 heads, 15,275 tokens, balanced/512,
Tile32. Materialized Graph vs eager Flat: **7,820,800 / 7,820,800
bit-exact**, 0 differences. Fused-vs-eager AND fused-vs-materialized:
**622 / 7,820,800** differing final BF16 elements (**0.00795315%**).
Maximum |delta|=0.000244140625, relative L2=0.002056845%.
Mismatches per head: 151, 84, 223, 164. This is a narrow
output-level difference, NOT proof that output is acceptable or
that the error originates from BF16 rounding. GPU speed and
independent peak-VRAM were NOT measured. The ordinary strict
eager/Graph gate remains FAIL_CLOSED.

Source-level candidate explanation: _fused_vbar_one_leaf reduced
the first row of the [16,16] MU.K score tile into a 1D vector,
then used axis=0 softmax, whereas the original _stats_kernel
uses 2D per-row axis=1 max/sum/softmax and 16x16 W.V dot.
Both implement the same conceptual formula, but reduction layout
and compiler schedules differ and are not guaranteed bit-exact.

This single experimental kernel-only change replaces that 1D
softmax with the original per-row 2D sequence:

    a=tl.dot(mu_matrix,tl.trans(k),input_precision=PREC)*scale
    m=tl.max(a,axis=1)
    w=tl.exp(a-m[:,None])
    z=tl.sum(w,axis=1)
    w=w/z[:,None]
    vb=tl.dot(w,v,input_precision=PREC)

It still extracts row zero after the same 16-duplicate-MU
dot geometry. There is NO change to Tile far/near, exact
selection, quantization, G/LZ/SG, Graph routing, memory guards,
model or default materialized mode. The CPU source contract is
updated to check the rowwise operations and to reject the
former first-row 1D reduction.

This is a **hypothesis test**, NOT a proven fix. CUDA compile,
output bit-exactness and warmed performance of this particular
version are UNRUN. First run the existing strict full-output
fused_vb A/B; if the strict gate fails again, do NOT automatically
relax parity or rerun indefinitely. Close this naive in-Tile
candidate and prioritize centre-cooperative K/V reuse only
after independent architecture/traffic review. Do not run video.


## 2026-11-14 — corrected fused VB strict numeric PASS; performance baseline drift

User-provided flat_fused_vb_ab(1).json from the strict, output-producing
A/B on 86ca5ec2e05e140fa59164d4b52e2eeb36ae8b41.
Report status=PASS_NUMERIC, completed=true. Exact same synthetic geometry
as the previous failed/diagnostic gate: 15,275 tokens, four distinct
BF16 heads, Balanced (512 exact leaves), Tile32, five timed full-call
replays, two independent Q/K/V banks, RTX 5070 Ti. No H3 checkpoint,
VAE/video, or Sol comparison.

Both output banks are **bit-exact** between materialized Graph and
fused_vb Graph; each mode's first-bank first-call Graph output
and second-bank Graph output also independently match the native
unchanged eager Flat reference. The per-centre rowwise
max/sum/exp/divide correction therefore passes this shape's
strict numerical gate. No quality tolerance was relaxed.

| Metric | materialized | fused_vb |
|---|---:|---:|
| Full attention GPU events / call | 25.462938 ms | 14.105049 ms |
| Synchronized call wall / call | 25.854340 ms | 14.534740 ms |
| First-call extra reserved MiB (NOT isolated) | 320 | 0 |
| Capture + eager parity wall (cold/uncalibrated) | 19,649 ms | 5,997 ms |

This single fixed-order run reports **1.805x GPU-event** and
**1.7788x wall** materialized/fused speed ratios. But the separate
earlier split_stream experiment (same 15,275/4/Balanced/Tile32
control) measured materialized at **5.658144 ms GPU**. The current
materialized control is thus about **4.50x slower** than that earlier
independent run, despite unchanged materialized math and same
nominal geometry. Do not interpret the current 1.8x as reliable
speedup or attribute drift to one cause. All two-mode trials
ran materialized FIRST, then experimental variant, with only one
averaged block of five timed replays per mode.

The report's first-call 320/0 MiB is NOT independently measured
whole-model memory savings: allocator caches and private CUDA Graph
pools persist across arms. The fast candidate may still regress
on an ordinary/faster materialized baseline. The cold capture
wall values also cannot be interpreted as warmed speed.

**Numerical gate: PASS for this synthetic 4-head shape.**
**Speed benefit: UNCONFIRMED.** Do not promote fused_vb or
run model/video on the basis of these timings alone.

### Opt-in independent-capture ABBA confirmation

tools/benchmark_h3_flat_fused_vb_ab.py now supports
--paired-abba to compare 4 independent strict Graph captures
in materialized/fused_vb/fused_vb/materialized order.
All use the SAME two distinct BF16 Q/K/V banks. For each
capture, normal first-call eager/Graph parity remains REQUIRED,
as does independent bank-B eager equality. Full output for
both banks is checked across variants AND repeated captures.
Only CPU copies of reference outputs persist across blocks;
the GPU session cache is cleared between captures.

After each first-call parity gate, 5 additional untimed
full-attention replays warm the captured Graph, then 7
individual GPU-event and synchronized wall samples are
recorded per block. One JSON records all 28 samples,
four block medians and the speed-decision contract.
An exploratory candidate speed signal requires BOTH:
at least 5% faster median of block medians and BOTH fused_vb
block medians below BOTH materialized block medians.
Even a positive signal is NOT authorization to promote to H3.

The safe runner supports -Candidate fused_vb -PairedABBA
-Rounds 7; it writes exactly ONE separate report:
D:\ComfyOutput\H3\math_probe\flat_fused_vb_abba.json.
The old flat_fused_vb_ab.json is preserved. Default
two-mode A/B and numeric-diagnostic paths are unchanged.
Model-resident GPU and independent-process VRAM are UNRUN.

The new ABBA CPU source contracts, CUDA benchmark and
speed confirmation are UNRUN until executed on RTX 5070 Ti.


## 2026-11-14 — completed strict four-capture ABBA: fused_vb speed NO_GO

Evidence: user-provided flat_fused_vb_abba.json from
719db9ca11995adc43c1c1a93f59e26578f4d5c1.
status=PASS_NUMERIC, completed=true, paired_abba=true.
Exact shape: 15,275 tokens, four native distinct BF16 heads,
Balanced/512, Tile32, two independent QKV banks,
five additional post-capture warmups and seven timed GPU
events per each of FOUR independently captured sessions,
order materialized/fused_vb/fused_vb/materialized.
No model, VAE, video, or Sol.

| Capture block | Variant | GPU median ms | Wall median ms |
|---|---|---:|---:|
| 1 | materialized | 4.287584 | 5.842000 |
| 2 | fused_vb | 13.432864 | 14.973400 |
| 3 | fused_vb | 13.512032 | 15.298500 |
| 4 | materialized | 4.281728 | 5.934200 |

Median of the two independent GPU block medians:
**materialized 4.284656 ms**, **fused_vb 13.472448 ms**.
Ratio materialized/fused = **0.318031x**;
fused requires **3.14435x** materialized GPU time
(+214.43%). The two materialized medians agree to
~0.14%, and both fused medians are ~13.4-13.5 ms.
Built-in candidate_speed_signal=false and
consistent_faster_across_capture_blocks=false.

All four fresh-capture first-call tests return
PASS_FIRST_REAL_CALL with strict eager/Graph bit equality,
bank-B eager/Graph parity, and replay equality. Both
complete output banks are bit-exact across the variants
and repeated captures. Thus numeric correctness is
confirmed for this synthetic shape, while the candidate
speed hypothesis is conclusively rejected for it.

Earlier fixed-order 5-round report at the same nominal
shape measured materialized 25.462938 ms and fused
14.105049 ms, appearing to show 1.8x fused speedup.
Those timings were not reproduced after strict ABBA.
The earlier first-arm 25.46-ms anomaly's root cause
remains UNKNOWN; do not ascribe it to clocks or
allocator behavior without direct evidence. The
independent-capture result has stable per-mode medians.

**DECISION: fused_vb speed NO_GO / CLOSED.**
Preserve the experimental kernel for research but do
not enable it in the normal dispatcher, increase
bench rounds, or run real H3 model/video/40k for this
same naive recompute geometry. Keep materialized
Tile32 with STATS_BT=16 and stock eager fallback.
This evidence commit does NOT change runtime behavior.

VRAM remains unproven: first-call
extra_reserved_mib_non_isolated=320 / 0 / 0 / 44 for
the four blocks is affected by shared CUDA allocator
and Graph private pools. It is NOT an independently
isolated comparison. The materialized workspace's
BF16 VB component (for one full 32-tile, 4-head
15,275-token window) has a static shape
[4,32,954,128], about 29.81 MiB of raw data;
removing it does not by itself quantify live Graph
memory savings or explain the observed 9.19-ms
GPU slowdown.

### New architectural question — multi-centre K/V reuse, no new kernel yet

The old fused_vb Tile executes a per-leaf
_fused_vbar_one_leaf call inside each Tile's far
loop, repeatedly loading the SAME native K/V
leaf for different centres/Tiles. The split_stream
option also retains a separate VB producer and
consumer and was slower in the earlier short A/B.

Any alternative needs a distinct reuse hypothesis,
not another BT/BN sweep:

1. Preserve existing exact G/LZ/SG score generation,
   fixed IDX/EXCL selection and 512 exact near
   leaves (or 256 in Fast), native K/V and the
   original far online-softmax + tail + merge.
2. Propose a **centre-block by leaf-block** fused
   producer/consumer schedule: load each native K/V
   leaf into shared/register-local storage once for
   a bounded group of centres, compute per-centre
   BF16 VB with the SAME rowwise reduction, and
   immediately consume it for Tile far accumulation.
   Do not write global full VB. Avoid recomputation
   for each Tile/RB/replay split. Reuse must be
   explicit and measurable, not assumed from syntax.
3. First produce a traffic/CTA ownership design:
   number of K/V loads per (head, leaf, centre
   block), centres per CTA, register/shared-memory
   occupancy, split order, exact BF16 rounding and
   how different Q rows accumulate. The previous
   source shows one query centre per Tile in group1;
   group scheduling must preserve which MU/EXCL
   is attached to each query tile.
4. Only if the design predicts lower total work
   AND a practical 5070 Ti occupancy should a
   distinct opt-in kernel be prototyped. Gate on
   two-bank bit-exact full output + ABBA full-call
   GPU events, then separately isolated VRAM;
   only afterward consider 40k/model-resident.
   Do not equate isolated leaf-kernel benchmarks
   with attention throughput.

**No new CUDA implementation or GPU request is
made by this closure.** Real H3, video, VRAM
isolation and a cooperative implementation remain
UNRUN / NOT IMPLEMENTED. The ABBA result supersedes
the earlier provisional 1.8x speed interpretation,
not the earlier PASS_NUMERIC result.
