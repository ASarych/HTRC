# Exact identities versus claims

## Exact

For a finite exponential mixture, differentiating log(Z) gives the weighted key mean; differentiating again yields the weighted key covariance. The normalized vector-valued numerator gives value-Jacobian Cov(V,K). The parent laws in README follow by summing child partitions and differentiating the child mixture weights:
`∇w_i = w_i(g_i - g_parent)`.

These are derivations / directly checkable mathematical identities, not new universal theorems about algorithm complexity.

## Approximate / not proven universally

HTRC's centroid-local Taylor-2 log mass, Taylor-1 conditional-value dipole, adaptive mass-weighted opening score, low-rank curvature and source reduction estimates are approximations / empirical methods. In particular the curvature-based opening indicator is **not** a certified upper bound on normalized attention-output error.

## Tests that are still needed for open publication

- Cross-model, cross-prompt/call and held-out stress validation (not just alternating rows from a saved capture).
- Matched end-to-end GPU wall-clock + memory comparisons against current exact attention baselines, including state build, kernel launches and fallback.
- Quality metrics on final normalized vector output, not just log Z, routing agreement or source-work proxies.
- Independent runnable dependency/CI gate; a repository-level MIT license is present, but any third-party notices still require review.

See individual archived research documents in docs/theory and docs/evidence.
