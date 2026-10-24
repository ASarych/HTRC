# H3 HTRC direct-from-leaf node-state feasibility — 2026-09-27

## Goal

The materialized-tree representation line is closed.  The remaining cost is
hierarchy-state mathematics.

An exactly lazy implementation of the current recursive rank8 compression does
not save compute: every one of the 80 root states is required before root
routing, and under the current recursive law every root depends on its full
descendant tree.  The dependency closure is therefore the full tree.

Real compute can be removed only if a node can be constructed directly from
leaf sufficient statistics, without recursively materializing all intermediate
node states.

## Direct node authority

At one query centroid q0, each B16 leaf b already exposes:

- log mass log Z_b;
- gradient g_b;
- conditional value v_b;
- exact leaf Hessian H_b in its <=8D physical span.

For any source-tree node N with descendant leaf set L_N:

    Z_N = sum_b Z_b
    w_b = Z_b / Z_N

The following are exact and associative, independent of tree shape:

    log Z_N = logsumexp_b log Z_b
    g_N     = sum_b w_b g_b
    v_N     = sum_b w_b v_b

The exact node log-mass Hessian is also available directly:

    H_N
      = sum_b w_b H_b
        + sum_b w_b (g_b-g_N)(g_b-g_N)^T.

This is the same exact mixture identity used by the recursive hierarchy before
rank truncation.

The candidate direct node state performs only one signed rank8 compression:

    H_N ~= U_N diag(lambda_N) U_N^T

instead of repeatedly truncating to rank8 at every intermediate level.

This does change the compact curvature representation relative to the current
hierarchical state, but it does not approximate node mass/gradient/value at the
centroid.  It may also be closer to the exact node Hessian because it postpones
the only rank truncation until the target node.

## Why this can unlock lazy state construction

If direct root states preserve the current routing/accepted-mass behavior well
enough, state construction can become:

1. evaluate leaf sufficient statistics at query centroids;
2. directly build the 80 root states;
3. route real queries;
4. only for opened roots, directly build their children from the same leaf
   statistics;
5. repeat for the actually opened frontier.

The computation then scales with requested node states rather than all
intermediate nodes in the forest.

This is a semantic change only in rank8 curvature compression.  It must be
gated before any runtime implementation.

## Broad numerical gate

Use block0/call1 historical raw capture, physical heads 1,16,27,55.

Per head:

- build the current accepted HTRC state;
- select the 8 query centroids with the most real current-call query rows;
- for each selected centroid:
  - construct direct-from-leaf state for all 80 roots;
  - compare exact centroid logZ/grad/value against current hierarchy;
  - compare direct rank8 quadratic action against:
    1. current hierarchical rank8 action;
    2. the exact direct dense Hessian action;
- on 512 evenly sampled real queries, compare current versus direct root routing
  at open fractions 2%,5%,10%;
- collect roots opened by the current state and test direct-from-leaf states for
  up to 32 of their child nodes per centroid;
- report root and opened-child action/prediction errors plus routing Jaccard,
  precision and recall.

The gate is mathematical/architectural, not a runtime benchmark.

## Decision rule

PROMISING if:
- direct logZ/grad/value matches current hierarchy to FP32 tolerance;
- direct rank8 is at least as faithful to its exact node Hessian action as the
  current hierarchical rank8 on actual residuals;
- root routing agreement remains high enough to justify a lazy prototype,
  especially on heads 27 and 55.

REJECT if one-shot direct rank8 materially destroys root routing/current-call
mass prediction.  In that case the next target must be a cheaper structured
root curvature law, not a lazy implementation of the same recursive hierarchy.
