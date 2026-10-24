# H3 Runtime Research State

> Branch-boundary note: unresolved HTRC mathematics is owned by
> `research/h3-math-probe`, whose canonical state is
> `research/H3_MATH_RESEARCH_STATE.md`.
> This runtime document describes live-prototype / canary integration state
> only. A full runner does not imply production readiness or transfer research
> authority into `feature/apl-h3-runtime`.

This file is the canonical handoff/state record for the active H3 runtime
research line on `feature/apl-h3-runtime`.

It is intentionally **not** a per-run diary. Update it when the research state
changes: stage transition, accepted/rejected direction, workload/gate change,
replay boundary change, or other decision that a fresh session must know.

Do **not** update it for an ordinary `test -> fix -> retest` loop that stays
inside the same stage and does not change the accepted state.

## Documentation rule

Update this file immediately when any of the following changes:

- current research stage / next executable step;
- accepted mathematical or runtime authority;
- validation workload or gate;
- replay/capture boundary;
- a hypothesis is accepted, rejected, or superseded;
- a blocker changes what work is executable;
- a runtime implementation becomes the new baseline.

A coherent implementation/test loop may contain many commits without touching
this file. The state file is updated once the loop resolves and the project
moves to the next step.

## Current accepted HTRC authority

- HTRC = Hierarchical Tilted Routed Collapse.
- accepted research authority:
  - mass: second-order log-mass;
  - value research authority: first-order dipole;
  - routing: mass-only;
  - cross-fit: even query rows train centroids, odd rows evaluate;
  - reference research point: Cq=32, K=16.
- current live runtime evaluator remains the bounded first implementation:
  `htrc_rank8_tail_monopole`.
- runtime forest: `root_kmeans_structural`.
- runtime factor path: `gram_cholesky`.
- traversal: `bucketed`.
- roots=80, fanout=4, query centroids=32, rank=8, leaf block=16.
- current default centroid chunk authority is 4.

The live value-law baseline does not supersede the accepted research value
authority; it is only the current runtime implementation gate.

## Current development / validation workload ladder

The research loop must not default to the largest production workload.

1. **39 frames** — default development/profiling/replay workload.
   Use for HTRC state-build/traversal iteration and memory/math debugging.
2. **124 frames** — intermediate scaling/quality validation when needed.
3. **243 frames** — full production-scale validation only after the 39-frame
   gate is healthy enough to justify the expensive run.

A 243-frame run must not silently become the normal research loop merely
because a launcher for it already exists.

## Replay authority

Primary iteration method after a real capture is offline replay, not repeated
full H3 startup.

Preferred boundary for current HTRC work:

`QKV/RoPE -> leaf geometry/exact-source split -> [REPLAY SNAPSHOT] -> HTRC state-build -> traversal`

The replay snapshot should contain the tensors consumed by HTRC after the
upstream projection/geometry stage:

- Q;
- compact leaf geometry;
- exact K/V source rows;
- layout/meta needed to reproduce the same HTRC call.

For changes restricted to HTRC state-build/traversal, reuse the snapshot and do
not reload H3 weights, VAE, reference encoder, or ComfyUI.

Take a new real capture only when the change crosses the replay boundary, for
example:

- QKV projection;
- RoPE;
- exact-source/layout semantics;
- leaf-geometry construction.

Historical raw `attention_block_*_call_*.pt` captures remain useful for
research because they contain Q/K/V and can rebuild current geometry, but they
may represent a different frame/token workload and must not be presented as a
243-frame validation result.

## Current runtime status

Latest validated full-head block-0 live HTRC gate before the current batching
change:

- 56/56 heads;
- q=76435 on the 243-frame workload;
- exact source tokens=3907;
- solved_fraction=1.0;
- bucket fallback=0;
- state-build ~108.8 s;
- traversal ~58.2 s;
- whole HTRC operator ~170.3 s;
- three 36x36 GPU eig failures were recovered by CPU FP64;
- Cholesky failures=0.

That run still executed centroid state construction effectively one centroid at
a time even though telemetry reported `htrc_centroid_chunk=4`.

## Current implementation change awaiting proper replay validation

The live state builder now uses the configured centroid work chunk:

`work_chunk = max(1, min(C, int(centroid_chunk)))`

with memory accounting updated to the same authority.

This change must be evaluated first on the cheap research workload/replay path,
not by repeatedly running the 243-frame full model.

## Current bottleneck

HTRC state-build remains the dominant cost.

The previous full-head run reported approximately:

- 8.38 million leaf eig matrices;
- 1.11 million Cholesky matrices;
- 5376 Cholesky batches.

The old 5376 count matches serial centroid execution:

`56 heads * 32 centroids * 3 hierarchical levels = 5376`.

Real centroid batching should reduce batch-launch count, but batching alone is
not expected to make production runtime sufficient.

## Current next research step

1. Establish/reuse a **39-frame** replay capture suitable for current HTRC.
2. Validate the real centroid-chunk=4 implementation offline.
3. Profile state-build structure from replay.
4. Determine which spectral/factorization work is truly centroid-dependent,
   head-dependent, or invariant/reusable.
5. Seek a mathematical/dataflow reduction of the millions of eig/Cholesky
   operations before broad kernel tuning.
6. Only after the 39-frame gate is materially improved, validate at 124 and
   then 243 frames.

## Session recovery rule

A fresh session must read this file before selecting an H3 runtime research
workload or next step.

Existing launchers/scripts are implementation tools, not research-state
authority. If a launcher and this state file disagree on the intended
development workload, this state file wins until explicitly changed and
documented here.
