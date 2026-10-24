# H3 RRC3 HTRC second-order tail — 2026-09-25

## Cross-fit result

The cross-fit HTRC validation used:
- even q rows only for query-centroid construction;
- odd q rows only for evaluation;
- mass routing only.

The direction survives out-of-sample validation.

Across representative heads excluding h55:
- Cq=32,K=64 gives mean HTRC->antipodal rel-L2 about 0.0809;
- Cq=32,K=32 gives about 0.1000;
- full-RRC3 block reduction is about 39.8x and 79.6x respectively;
- mean true top-K block recall is about 0.901 and 0.880 respectively.

The remaining error is strongly head-dependent. This indicates that routing is
not the only bottleneck: the first-order tail approximation itself is too weak
on several heads.

## 1. Move curvature from routing into the approximation

The previous curvature-aware routing score underperformed mass routing because
the curvature upper bound is conservative: it selects low-mass blocks that are
uncertain but unimportant.

Do not use that bound as the primary router.

Instead use the actual second-order Taylor term to improve every approximate
tail block.

For one block:

  f_b(q) = log Z_b(q).

At centroid q0, for delta=q-q0:

  f_b(q)
    ~= f_b(q0)
       + delta^T g_b
       + 1/2 delta^T H_b delta.

The first two terms are the current HTRC tail model.

## 2. Exact block Hessian identity

A block is a mixture of 8 antipodal pair masses z_p.

For each pair:

  ell_p(q)=log z_p(q)

  g_p = grad ell_p

  H_p = alpha'(a_p) c_p c_p^T
        + sech^2(h_p) r_p r_p^T.

With pair weights

  pi_p = z_p / Z_b,

the exact block log-mass Hessian is

  H_b
    = E_pi[H_p]
      + Cov_pi(g_p).

For a particular residual delta, no dense matrix is needed.

The quadratic scalar is

  delta^T H_b delta
    = E_pi[
        alpha'(a_p) (delta^T c_p)^2
        + sech^2(h_p) (delta^T r_p)^2
      ]
      + Var_pi(delta^T g_p).

That is exact at the centroid.

## 3. Important computational consequence

The second-order scalar for one query/block requires only pairwise scalar
projections:

  delta.c_p
  delta.r_p
  delta.g_p

for p=1..8.

No 128x128 Hessian is formed.

This is still more expensive than the desired final runtime if done for every
query/block directly, but it is the correct offline authority for deciding
whether second-order local response is worth compressing.

If it materially improves cross-fit quality, compress it later.

If it does not, do not spend time designing a runtime representation.

## 4. Compact runtime representations if validated

Candidate representations, in priority order:

### A. Query-cluster local PCA Hessian

For each query cluster, compute a small residual basis U_i with rank r=4..8.

Store for each block:

  Htilde_ib = U_i^T H_ib U_i   [r x r]

and

  gtilde_ib = U_i^T g_ib       [r].

For a query residual:

  y = U_i^T delta

then

  log Z_b(q)
    ~= logZ0
       + y^T gtilde
       + 1/2 y^T Htilde y.

This makes block scoring O(r^2) instead of O(8D).

### B. Diagonal projected curvature

Keep only diag(Htilde).

Cost becomes O(r) per block.

### C. Scalar curvature

Approximate

  1/2 delta^T H delta
    ~= 1/2 lambda_b ||delta||^2

using a fitted signed scalar rather than the previous absolute upper bound.

This is cheapest but should only be considered if A/B show that anisotropy is
small.

## 5. Tail value remains monopole in the first second-order test

Do not simultaneously add a value Jacobian.

First isolate whether mass reconstruction is the dominant remaining error.

Use:
- second-order log mass for all approximate blocks;
- exact full antipodal numerator/denominator for selected K;
- centroid conditional value V0 for the non-selected tail.

If denominator error collapses but output rel-L2 remains high, then the next
problem is the tail conditional value and a dipole/Jacobian term is warranted.

If both improve, no value Jacobian is needed.

## 6. Next experiment

One cross-fit sweep on the same capture:

  Cq={4,8,16,32}
  K={0,4,8,16,32,64}

Compare:
- first-order mass tail (existing authority);
- exact second-order log-mass tail;
- mass routing in both cases.

Report:
- HTRC->antipodal rel-L2;
- HTRC->exact rel-L2;
- denominator relative error;
- retrieved mass and top-K recall;
- per-head and ex55 aggregate.

Do not add runtime-specific approximations yet.

Decision:
- if second order materially improves difficult heads h0/h8/h16, derive a
  local-PCA compressed Hessian representation;
- otherwise abandon Taylor-order escalation and improve tail values/clustering
  instead.
