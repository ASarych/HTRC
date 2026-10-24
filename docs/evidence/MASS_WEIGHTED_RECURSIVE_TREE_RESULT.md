# H3 RRC3 HTRC Mass-Weighted Recursive Tree Result — 2026-09-25

## Result

MASS_WEIGHTED_HTRC_RECURSIVE_TREE = STRONG PASS.

Configuration:
- Cq=32;
- roots=80;
- fanout=4;
- even-query centroid training / odd-query evaluation;
- HTRC-native second-order node mass law;
- recursive opening;
- exact antipodal one-block leaves;
- certificate:

  score_N = predicted_mass_fraction_N * (exp(|q_N|/2)-1)

implemented in log space.

## Aggregate frontier

Across all 8 representative heads:

### open=0.02
- mean global-logZ abs error ~= 0.0163804
- mean TV ~= 0.0106758
- mean source reduction ~= 18.9872x
- mean visited nodes ~= 137.65/query
- mean exact leaves ~= 16.28/query

Excluding h55:
- error ~= 0.0165302
- reduction ~= 19.8466x

h55:
- error ~= 0.0153320
- TV ~= 0.0092246
- reduction ~= 12.9711x
- visited nodes ~= 196.36/query
- exact leaves ~= 48.91/query

### open=0.05
- mean error ~= 0.0103953
- mean TV ~= 0.0072443
- mean reduction ~= 12.3944x
- mean visited nodes ~= 218.82/query
- mean exact leaves ~= 47.70/query

### open=0.10
- mean error ~= 0.0070680
- mean reduction ~= 7.9405x

### open=0.20
- mean error ~= 0.0044438
- mean reduction ~= 4.4688x

## Comparison

Raw-key recursive Taylor-4 reference near 10x:
- mean error ~= 0.08783
- mean reduction ~= 11.28x
- h55 error ~= 0.27359
- h55 reduction ~= 6.19x

HTRC absolute-curvature certificate:
- mean error ~= 0.01950 at only ~=7.08x reduction;
- h55 ~=0.02388 at only ~=2.91x.

Mass weighting repairs the selectivity failure:
- open=0.02 is both much more accurate than raw Taylor-4 and much cheaper;
- h55 no longer forces broad opening.

## Interpretation

The source-side hierarchy is now supported by three independent pieces:

1. recursive topology beats fixed depth;
2. HTRC second-order node physics beats raw-key Taylor-4;
3. expected global error contribution is the correct opening objective, not raw
   curvature magnitude.

The current mathematical bottleneck is runtime representation of node curvature.

The offline authority computes exact directional second-order action by merging
block-local derivatives. Runtime cannot pay block-local 9D projections merely to decide
whether a coarse node should open.

## Decision

KEEP as current primary mass hierarchy:
- roots=80 / fanout=4;
- HTRC second-order node predictor;
- mass-weighted uncertainty certificate;
- open=0.02 as compute-biased authority;
- open=0.05 as quality-biased >=10x authority.

NEXT:
compress each node Hessian itself, locally per query centroid.

Do NOT return to a global query basis.

First compact-curvature gate:
- exact dense node Hessian is offline authority only;
- compare diagonal curvature and signed top-eigenmode approximations r={2,4,8};
- evaluate the SAME recursive tree/certificate using approximated q_N;
- preserve node mass prediction and topology;
- report final mass error and source reduction, not Hessian Frobenius error alone.

PASS requires a compact node representation that preserves most of the
open=0.02/open=0.05 frontier.
