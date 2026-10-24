# H3 HTRC-native recursive source tree — 2026-09-25

## Status

NEXT MASS GATE after PASS of the recursive Taylor-4 tree skeleton.

The topology/opening mechanism is retained:
- semantic source tree;
- roots=80;
- fanout=4;
- recursive query/node selective opening;
- exact leaf fallback.

Only the node physics changes.

## 1. Block authority at a query centroid

For one query centroid q0, every B16 antipodal block b already provides exact local
authority:

  Z_b(q0)
  g_b = grad log Z_b(q0)
  H_b = Hessian log Z_b(q0).

The block Hessian is not guessed.  It follows from the current antipodal pair law:

  H_b = E_pi[H_pair] + Cov_pi(g_pair).

## 2. Exact aggregate node derivatives

For a source-tree node N containing blocks b, define

  Z_N = sum_b Z_b

and block weights

  w_b = Z_b / Z_N.

Then

  g_N = sum_b w_b g_b

and

  H_N = sum_b w_b H_b + Cov_w(g_b).

This is the exact gradient and Hessian of log Z_N at q0.

Therefore a query residual delta has the node-local second-order model

  log Z_N(q0+delta)
    ~= log Z_N
       + delta^T g_N
       + 1/2 delta^T H_N delta.

No raw-key Taylor approximation is used in this node law.

## 3. Certificate

The first gate deliberately uses the local second-order correction itself:

  c2_N(delta) = | 1/2 delta^T H_N delta |.

This is not yet claimed to be a rigorous error bound.

It is a response-native refinement indicator:
- small c2 -> node is locally close to affine in log-mass;
- large c2 -> node response is curved and should be opened.

Also record the realized discrepancy between first and second order for diagnostic
purposes, but route only from c2.

If c2 alone is insufficient, the next certificate must use a third-order remainder
proxy or child disagreement.  Do not introduce those in the first gate.

## 4. Cross-fit protocol

Query centroids are learned only from even query rows.

Odd query rows are evaluation only.

For every odd query:
- assign to its even-trained centroid;
- start from 80 source roots;
- accept/open recursively using one absolute c2 threshold;
- a one-block leaf uses the exact antipodal block mass.

This preserves the HTRC held-out protocol.

## 5. Thresholds

Do not sweep tree geometry again.

Fixed:
- roots=80;
- fanout=4;
- Cq=32.

Calibrate c2 thresholds from TRAIN query/node values at the root level corresponding to
nominal root opening fractions:

  {0.02, 0.05, 0.10, 0.20}.

Apply the same absolute threshold recursively.

## 6. First numerical authority

Compare on odd queries:

1. exact antipodal block-mass aggregation;
2. accepted HTRC-native recursive second-order tree.

Report:
- global logZ mean/p95 absolute error;
- source-block distribution TV/KL;
- mean/p95 visited nodes;
- mean exact leaves;
- effective source reduction;
- h55 separately;
- comparison with the passed raw-key recursive Taylor-4 tree.

## 7. Cost accounting

This is an offline mathematical gate.

The first implementation may form dense H_N or otherwise use expensive block-level
derivative information during precompute.

Do NOT count that offline authority implementation as a runtime candidate.

If the node law passes, derive a compact node representation:
- low-rank signed curvature;
- diagonal + low-rank;
- or block-local sufficient-coordinate projection.

The runtime target is then one cheap node evaluation, not a dense D x D quadratic form.

## Decision gate

PASS if, near >=10x effective source reduction:
- HTRC-native tree materially improves mass error versus the raw-key recursive tree;
- h55 improves or at least does not require broad leaf explosion;
- recursion remains selective.

If PASS:
- retain HTRC node mass law;
- derive compact curvature representation;
- then add HTRC conditional-value monopole/dipole to accepted nodes.

If FAIL:
- keep the validated recursive topology;
- reject c2 as the node certificate;
- test child-disagreement / third-order remainder rather than changing tree geometry.
