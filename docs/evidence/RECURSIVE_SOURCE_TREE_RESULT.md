# H3 RRC3 HTRC Recursive Source-Tree Result — 2026-09-25

## Result

HTRC_NODE_LAW = STRONG PASS.
ABSOLUTE_CURVATURE_CERTIFICATE = REJECTED FOR SELECTIVITY.

Authority:
- Cq=32;
- roots=80;
- fanout=4;
- even-query centroids / odd-query evaluation;
- recursive selective opening;
- exact antipodal leaf mass;
- internal-node second-order HTRC mass law;
- certificate c2 = |0.5 delta^T H_N delta|.

## Aggregate

Across 8 representative heads:

open=0.02:
- mean global-logZ abs error ~= 0.01950
- mean TV ~= 0.01015
- mean source reduction ~= 7.08x
- mean visited nodes ~= 490.5/query
- mean exact leaves ~= 251.8/query

open=0.05:
- error ~= 0.01173
- reduction ~= 3.56x

open=0.10:
- error ~= 0.00736
- reduction ~= 2.24x

open=0.20:
- error ~= 0.00423
- reduction ~= 1.45x

Excluding h55 at open=0.02:
- error ~= 0.01887
- reduction ~= 7.68x.

h55 at open=0.02:
- error ~= 0.02388
- TV ~= 0.00684
- reduction ~= 2.91x.

## Comparison with raw-key recursive Taylor-4 tree

Previous passed skeleton point:

  roots=80, fanout=4, open=0.10

- mean logZ error ~= 0.08783
- mean source reduction ~= 11.28x
- h55 error ~= 0.27359
- h55 reduction ~= 6.19x

The HTRC-native second-order node law is dramatically more accurate:
- even the open=0.02 point has only ~=0.0195 mean error;
- h55 falls from ~=0.2736 to ~=0.0239.

Therefore the new node physics is validated.

The failure is computational selectivity:
the absolute curvature certificate opens too many nodes.

## Interpretation

c2 = |0.5 delta^T H_N delta| measures local log-mass curvature but ignores whether
that node carries enough predicted mass to matter to the global denominator.

A tiny-mass node with strong local curvature can trigger deep refinement even though
its contribution to total attention error is negligible.

This is the same distinction already identified in HTRC routing:
uncertainty alone is not the objective; expected global error contribution is.

## Decision

KEEP:
- recursive source-tree topology;
- HTRC exact aggregate node derivatives;
- second-order node mass predictor;
- exact leaf fallback.

CLOSE:
- absolute curvature c2 as the primary opening certificate.

NEXT:
test mass-weighted second-order uncertainty.

For node N with predicted log mass m_N and curvature magnitude c_N:

  u_N = exp(c_N)-1

and predicted absolute denominator-error scale:

  E_N = exp(m_N) * u_N.

For numerically stable routing inside one query, normalize by predicted total mass:

  s_N = softmax_mass_N * u_N.

Use one absolute contribution threshold recursively.

This targets nodes capable of changing the global denominator, rather than merely
nodes with large local curvature.

Do not alter tree geometry or node mass law.
