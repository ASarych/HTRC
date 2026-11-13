# HTRC Flat: universal vs long-context optimization and test gates

> **Numerical policy updated on 2026-11-13.** The quality oracle is
> **exact dense attention**, and numerical disagreements between Flat
> variants must be recorded as percent relative error, never aborting
> the experiment. The previous bit-exact gating described further below
> is retained only as historical design text, not the active matrix
> contract. Canonical implementation and procedure:
> H3_FLAT_DENSE_QUALITY_PERCENT_2026-11-13.md.


Date: 2026-11-13. Scope: ASarych/HTRC research branch only.
Source base: experiment/flat-long-context-fused-vb-20261113
@ 51aea61863920298e76922d2d902c38aae89b3df.
No ComfyNodes changes, no production/main promotion.

## Key owner rule

**Classify every optimization by expected scaling BEFORE testing.**
Do NOT use successful long-sequence throughput as evidence of a universal
benefit. Do NOT ignore short sequences for long-prioritized changes:
they still need exact-output regression and short speed/VRAM data.
Classification is a hypothesis, not a measurement and not a selector.

## Hypotheses by class

| Hypothesis class | Research work | Primary goal | Short regression? |
|---|---|---|---|
| UNIVERSAL_CANDIDATE | CUDA Graph launch/host scheduling | reduce launches at all N | Mandatory: yes |
| UNIVERSAL_CANDIDATE | native QKV staging, fused quant, redundant copies | reduce per-call overhead at all N | Mandatory: yes |
| UNIVERSAL_CANDIDATE | Stats SCORE_ONLY first pass | remove V dot+VB write independent of N; **only a stage** | Mandatory: yes |
| LENGTH_AMPLIFIED_CANDIDATE | 16/32/64 Graph window tiles | N-dependent replay/group cap crossover | Mandatory: yes |
| LENGTH_AMPLIFIED_CANDIDATE | split_stream VB | smaller peak VB but extra recomputation | Mandatory: yes |
| LENGTH_AMPLIFIED_CANDIDATE | fused_vb inside Tile | eliminate VB global buffer, possible huge extra K/V traffic | Mandatory: yes |
| LENGTH_AMPLIFIED_CANDIDATE | cooperative multi-centre K/V reuse | reduce repeated K/V traffic as number of centres grows | Mandatory: yes |
| CAPACITY_ENABLER | streaming QKV, INT8 K/V, block lifetime | enable large N within 16 GiB | Mandatory: yes |
| CAPACITY_ENABLER | workspace/head-group-aware scheduling | admission and head parallelism at large N | Mandatory: yes |

The "universal" class is **potential**, not universal proof.
The "length-amplified" class may also gain on short clips.
Once GPU results exist, record **measured winning domain and crossover
range** independently of the initial hypothesis. A long-only dispatch
threshold is forbidden without an actual same-machine A/B by length.

## Fixed test sizes

| Band | Tokens | Purpose |
|---|---:|---|
| Short A | 15,275 | Real H3 39-frame attention geometry; former Graph gate |
| Short B | 18,723 | Former Flat/Sol/dense comparison geometry |
| Medium | 40,759 | Early crossover search |
| Long | 82,000 | Scaling plus scratch-group/VRAM stress |
| Full HD memory gate | 223,605 | Analytical/model-resident admission only; NOT a GPU benchmark |

Avoid 223k synthetic full attention on the 5070 Ti until QKV
materialization is redesigned. The previous 362-frame attempt failed
on memory admission, and even Sol required an additional workspace.

## Mandatory two-part scorecard

1. **Quantitative quality measurement for every tested length.**
   Same native BF16 post-RoPE-style QKV, two independent QKV banks,
   same seed/heads/fast-or-balanced/tiles, same 256 or 512 near leaves,
   same tail handling. Compute FP32 exact dense reference across ALL keys
   for sampled Q rows, then error % per Flat variant. IDX/EXCL and
   equality to eager/materialized remain optional diagnostics, NOT gates.
2. **Performance and memory gate for every tested length.**
   Record warmed full-attention CUDA-event timing and synchronized wall,
   VRAM additional/reserved and group/head geometry. Do not compare
   a stage-only Stats microkernel with the full H3 attention. Separate
   stable short benefit, long speedup and VRAM capacity. For definitive
   performance, alternate modes across runs and report dispersion;
   the existing two-bank A/B runs one mode at a time and is explicitly
   preliminary.

A universal candidate needs short speed advantage (at both 15k/18k,
within measurement noise) plus no substantial slowdown at 40k/82k.
A length-amplified candidate can be slower at 15k, but must record
short dense-reference error % and show actual measured crossover at larger sizes,
not an inferred one. A capacity enabler can improve max feasible N
even when it does not improve short latency, but still must quantify
short error % against dense and document short latency penalty.

## Current code and one-report protocol

- tools/h3_flat_scale_protocol.py contains the versioned classification
  registry and mandatory SHORT->MEDIUM->LONG sequence.
- tools/benchmark_h3_flat_scale_matrix.py wraps existing
  tools/benchmark_h3_flat_streamed_vb_ab.py and
  tools/benchmark_h3_flat_fused_vb_ab.py. It does not modify
  kernels, eager Flat, Graph default, or native H3 model code.
- Every scope includes both short controls first:
  - short = 15,275 + 18,723;
  - scaling = short + 40,759 + 82,000;
  - capacity = scaling + analytic Full HD admission (never GPU).
- **Default is PLANNED_ONLY**, no accidental GPU work.
- Explicit --execute runs independent full-attention Graph variants
  against the SAME FP32 dense all-key, sampled-query reference.
  Numerical disagreement with eager/materialized never stops the matrix.
  Actual CUDA/runtime failures are recorded separately.
- Output is ONE JSON, per-case QUALITY_MEASURED / RUNTIME_FAILED /
  NUMERIC_NONFINITE_RECORDED, and all cases not executed remain UNRUN.
  No quality accuracy threshold or fabricated speed PASS.
  The 223605-token capacity row includes the existing conservative
  materialized-Flat source formula, explicitly NOT a true admission
  result or an estimate for fused_vb.
- This tool doesn't compare Sol, doesn't load HyperFlow, and doesn't
  claim a Full HD video run.

### GPU execution once source kernel correctness is ready

From an isolated checkout of the relevant research branch:

    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m unittest discover -s tools -p test_h3_flat_scale_protocol.py -v
    & $py -B tools\benchmark_h3_flat_scale_matrix.py --scope short --execute --variants split_stream
    # Scaling only after short numerical gate passes:
    & $py -B tools\benchmark_h3_flat_scale_matrix.py --scope scaling --execute --variants split_stream

Single report, overwritten at each invocation:
D:\ComfyOutput\H3\math_probe\flat_short_long_matrix.json

Use fused_vb only when its short GPU compile/parity and latency are
acceptable; the naive per-leaf helper has a strong *negative*
performance hypothesis. Neither new kernel has GPU approval yet.

## Status

- Previously real-model proven: Graph+materialized vs eager on 15k,
  six-step latents exact; speed gain at that size. Comparisons to Sol
  were earlier and not fully matched.
- split_stream: research implementation, dense quality %/GPU timing UNRUN.
- fused_vb: research implementation, dense quality %/GPU timing UNRUN.
- New classification protocol, plan-only matrix and CPU contracts:
  implementation complete; unit-test runtime status UNRUN until run.
- Multi-centre shared-K/V, fully fused SEL/Far, 223k streamed QKV:
  NOT IMPLEMENTED.
- No new GPU test claims or automatic production mode selection.
