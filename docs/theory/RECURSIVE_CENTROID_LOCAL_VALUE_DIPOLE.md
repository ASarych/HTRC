# H3 HTRC recursive centroid-local value dipole derivation — 2026-09-27

## Context

The active HTRC research authority already separates roles:

- mass: centroid-local second-order log-mass;
- value: centroid-local first-order conditional-value dipole;
- routing: mass-only.

Recent experiments closed two tempting shortcuts:

1. lazy V-leaf construction under the current bottom-up value law, because
   accepted internal nodes collectively cover 100% of descendant leaves;
2. one global query anchor per node, because a single global affine value law
   fails badly on difficult heads.

The correct question is therefore not whether value response can be made
global. It is whether the accepted **centroid-local** dipole can be aggregated
recursively through the source tree without changing its mathematics.

## Exact node value identities

For any child/source component i at centroid q0 define:

- mass Z_i;
- conditional value v_i = N_i / Z_i;
- log-mass gradient g_i = grad log Z_i;
- conditional-value Jacobian J_i = d v_i / d q.

For a parent node N with children i:

    Z_N = sum_i Z_i
    w_i = Z_i / Z_N

The exact parent conditional value is:

    v_N = sum_i w_i v_i

and the exact parent log-mass gradient is:

    g_N = sum_i w_i g_i.

Differentiate v_N with respect to q. Since

    d w_i = w_i (g_i - g_N)^T d q,

we obtain

    J_N
      = sum_i w_i J_i
        + sum_i w_i (v_i - v_N) (g_i - g_N)^T.

Because sum_i w_i (v_i - v_N) = 0, the g_N term cancels exactly:

    J_N
      = sum_i w_i J_i
        + sum_i w_i (v_i - v_N) g_i^T.

This is the exact recursive value-dipole merge law.

No descendant token/leaf replay is needed once each child exposes:

    (log Z_i, g_i, v_i, J_i).

The same scalar weights already used by the current bottom-up node mass/value
merge are sufficient.

## Numerator form

There is an equivalent additive representation that can be useful for
implementation.

Define the numerator derivative

    A_i = (1 / Z_i) d N_i / d q.

Then

    J_i = A_i - v_i g_i^T

and

    A_N = sum_i w_i A_i.

Therefore one may recursively aggregate either:

1. (v_i, J_i, g_i), using the covariance correction above; or
2. (v_i, A_i, g_i), where A is a pure mass-weighted child average and
   J_N = A_N - v_N g_N^T is reconstructed only where required.

The second form is algebraically simpler for tree construction because all
three parent quantities are mass-weighted child averages.

## Leaf sufficient-coordinate structure

The accepted antipodal B16 leaf already factors all query dependence through
nine exact local coordinates:

    dk = <delta q, kappa>
    beta = Q8^T delta q, beta in R^8.

The historical block-local sufficient-coordinate gate confirmed that this map
has effective rank exactly 9 on every tested block.

Consequently the leaf conditional-value Jacobian does not need a dense
128x128 representation. It can be stored exactly as

    J_leaf delta q
      = a_kappa * dk + A_beta * beta

with:

    a_kappa in R^Dv
    A_beta  in R^(Dv x 8).

This is an exact q-side rank <=9 representation of the leaf dipole.

The parent covariance term

    (v_i - v_N) g_i^T

adds q-side directions from child gradients. Therefore parent dipole rank is
not guaranteed to remain 9, but it is generated recursively from known
structured child directions. Its actual numerical rank must be measured before
choosing a compression law.

## Immediate numerical gate

Do not design a runtime kernel yet.

On the existing block0/call1 raw capture:

- physical heads: 1,16,27,55;
- use a small set of real HTRC query centroids;
- construct the accepted antipodal leaf (Z,g,v,J) authority;
- recursively merge J through the exact source tree using the identity above;
- independently flatten all descendant leaves for sampled internal nodes and
  compute the same node J directly;
- fail closed if recursive-vs-flat relative error exceeds FP32 tolerance;
- measure singular spectra of sampled internal-node J matrices by depth;
- report rank needed for 99%, 99.9% energy and approximation errors for
  r={4,8,16,24,32}.

This gate answers two questions at once:

1. can the accepted centroid-local value dipole be tree-aggregated exactly
   without semantic change?
2. does the recursive node dipole retain a compact q-side rank in practice?

Only after these are known should implementation choose between full structured
A/J propagation, rank-compressed node dipoles, or another exact factorization.
