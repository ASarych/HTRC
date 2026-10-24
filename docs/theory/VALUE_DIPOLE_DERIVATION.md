# H3 RRC3 HTRC value-dipole derivation — 2026-09-25

## Result motivating this step

Cross-fit second-order HTRC materially improves denominator reconstruction.

For Cq=32, excluding h55, aggregate denominator error is already about:
- K=16: ~1.33%
- K=32: ~1.04%
- K=64: ~0.78%

while HTRC->antipodal rel-L2 remains materially larger.

This demonstrates that the dominant remaining tail error is no longer block
mass reconstruction. It is the conditional value approximation.

Therefore do not escalate log-mass Taylor order again.

## 1. Conditional block value

For block b:

  v_b(q) = N_b(q) / Z_b(q).

At centroid q0:

  v_b(q0 + delta)
    ~= v0_b + J_b delta.

The Jacobian is exactly

  J_b
    = (1/Z_b) grad_q N_b
      - v_b g_b^T

where

  g_b = grad_q log Z_b.

So the task reduces to deriving grad N for the corrected antipodal numerator.

## 2. Pair numerator

For one pair p in block b:

  t = q.kappa
  a = q.c
  h = q.r

  e(a)  = (1+a/2)^2 + eps
  de(a) = 1+a/2
  d2e   = 1/2

Value vectors:

  B = vbar + M
  D = dA
  M = mA.

Correct pair numerator:

  n(q)
    = 2 exp(t) [
        e(a) ( cosh(h) B + sinh(h) D )
        + de(a) cosh(h) M
      ].

Define

  F(q)
    = e(a) (ch B + sh D)
      + de(a) ch M.

Then

  n = 2 exp(t) F.

## 3. Gradient of F

Because:
  grad a = c
  grad h = r

and:
  grad e  = de c
  grad de = d2e c
  grad ch = sh r
  grad sh = ch r,

we obtain

  grad F
    =
      c ⊗ [
        de (ch B + sh D)
        + d2e ch M
      ]
    +
      r ⊗ [
        e (sh B + ch D)
        + de sh M
      ].

Let

  Cvec =
      de (ch B + sh D)
      + d2e ch M

  Rvec =
      e (sh B + ch D)
      + de sh M.

Then

  grad F
    = c ⊗ Cvec + r ⊗ Rvec.

## 4. Gradient of pair numerator

Product rule on exp(t):

  grad n
    = 2 exp(t) [
        kappa ⊗ F
        + c ⊗ Cvec
        + r ⊗ Rvec
      ].

This is a sum of only three rank-1 outer products per pair.

Summing 8 pairs:

  grad N_b
    = sum_p grad n_p.

No dense symbolic differentiation is needed.

## 5. Apply Jacobian without storing D x D

For one query residual delta:

  J_b delta
    =
      (1/Z_b) [grad N_b]^T delta
      - v_b (g_b.delta).

Using the rank-1 structure:

  [grad n_p]^T delta
    =
      2 exp(t) [
        (delta.kappa) F
        + (delta.c) Cvec
        + (delta.r) Rvec
      ].

Therefore the dipole correction for one query/block requires only scalar
projections:

  delta.kappa
  delta.c_p
  delta.r_p

plus vector combinations in value space.

No 128x128 Jacobian must ever be formed.

## 6. Exact local dipole evaluation

For the offline authority experiment, evaluate J delta exactly from the above
rank-1 representation for every non-retrieved block.

Tail block value becomes

  Vhat_b(q)
    = V0_b + J_b delta.

Use the already validated second-order block mass predictor.

Selected K blocks remain full antipodal RRC3.

This isolates whether first-order conditional value motion is the missing term.

## 7. Runtime compression if validated

Direct Jdelta still costs pair-level QxB work and is not the final runtime.

If the dipole materially improves quality, compress it in query-cluster residual
coordinates.

For query cluster i, compute PCA basis U_i in residual q-space, rank r.

Store:

  Jtilde_ib = J_ib U_i   [Dv x r]

For query residual:

  y = U_i^T delta

then

  J delta ~= Jtilde y.

Cost per approximate block becomes O(Dv*r), with r candidate 2..8.

The same residual basis can also compress the second-order mass Hessian:

  Htilde = U_i^T H U_i.

Then one shared local coordinate y supports:

  logZ ~= logZ0 + gtilde.y + 1/2 y^T Htilde y

  V ~= V0 + Jtilde y.

This is the natural local multipole representation.

## 8. Next experiment

Cross-fit, same even/odd split.

Use:
- second-order mass authority;
- mass routing;
- Cq={4,8,16,32};
- K={0,4,8,16,32,64}.

Compare tail values:
1. monopole V0;
2. exact local dipole V0 + Jdelta.

Report:
- HTRC->antipodal rel-L2;
- HTRC->exact rel-L2;
- denominator error (should be identical between value modes);
- per-head and ex55 aggregate.

Decision:
- if dipole materially improves output while denominator stays unchanged, move
  directly to local-PCA compression;
- if dipole does not improve difficult heads, source clustering/value model must
  change rather than increasing Taylor order.
