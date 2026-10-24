# H3 RRC3 HTRC Hierarchical Curvature Compression Result — 2026-09-25

## Result

HIERARCHICAL_SIGNED_RANK8_CURVATURE = REJECTED AS PRIMARY NODE REPRESENTATION.

Authority:
- Cq=32;
- roots=80;
- fanout=4;
- mass-weighted recursive HTRC tree;
- open fractions {0.02,0.05};
- comparison: exact directional curvature, diagonal curvature, hierarchical signed rank8.

## Aggregate

### open=0.02
Exact:
- mean global-logZ abs error ~= 0.0163801
- mean TV ~= 0.0106756
- mean source reduction ~= 18.9856x

Diagonal:
- error ~= 0.0538543
- TV ~= 0.0256230
- reduction ~= 17.3415x
- root-open agreement ~= 0.9518

Rank8:
- error ~= 0.0591866
- TV ~= 0.0154679
- reduction ~= 19.0480x
- root-open agreement ~= 0.9808

### open=0.05
Exact:
- error ~= 0.0103954
- TV ~= 0.00724465
- reduction ~= 12.3845x

Diagonal:
- error ~= 0.0373387
- TV ~= 0.0202062
- reduction ~= 11.7675x
- root-open agreement ~= 0.9382

Rank8:
- error ~= 0.0456002
- TV ~= 0.0126027
- reduction ~= 12.1497x
- root-open agreement ~= 0.9648

## Interpretation

The important observation is that rank8 preserves routing decisions much better than it
preserves accepted-node mass values.

At open=0.02:
- root opening agreement is ~98%;
- source reduction is essentially unchanged;
- but global-logZ error degrades by ~3.6x.

At open=0.05:
- root opening agreement is ~96.5%;
- source reduction remains near the exact frontier;
- error degrades by ~4.4x.

Therefore the dominant failure is not tree selectivity.

The failure is using the same compact rank8 Hessian approximation for the actual
second-order accepted-node mass predictor.

This separates two roles that were previously coupled:
1. curvature as a ROUTING/CERTIFICATE signal;
2. curvature as a VALUE of the second-order node mass approximation.

A compact approximation may be good enough for (1) even when it is not accurate enough
for (2).

## Decision

CLOSE:
- hierarchical signed rank8 Hessian as the primary accepted-node second-order mass
  representation;
- lower rank2/rank4 sweeps: they cannot rescue a failure already present at rank8;
- diagonal Hessian as a primary accepted-node mass representation.

KEEP:
- exact HTRC second-order node law as offline authority;
- mass-weighted recursive routing;
- hierarchical rank8 as evidence that routing decisions can be approximated with high
  agreement.

## NEXT

Decouple routing from accepted-node mass evaluation.

Next gate should test a cheap approximate certificate while keeping the accepted-node
mass predictor exact under the existing offline authority, in order to answer:

  Can cheap routing preserve the exact HTRC tree frontier?

Candidate cheap routing signals, in order:
1. rank8 curvature ONLY for routing, exact second-order mass for accepted nodes;
2. child-disagreement / parent-vs-children predicted-mass residual;
3. scalar upper/lower curvature bounds.

If (1) preserves the exact frontier, the routing problem is solved independently.
Then derive a separate cheap accepted-node mass representation.

Do not conflate certificate compression and node-mass compression again.
