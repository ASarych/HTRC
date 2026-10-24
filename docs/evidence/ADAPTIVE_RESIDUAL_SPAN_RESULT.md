# H3 RRC3 HTRC runtime-adaptive residual-span result — 2026-09-25

## Result

RUNTIME_ADAPTIVE_RESIDUAL_SPAN = MATHEMATICALLY_VALID / COMPUTATIONALLY_INSUFFICIENT.

Geometry:
- current-call Q only;
- Cq = 32;
- K = 16;
- 128 queries;
- 2547 source blocks;
- sufficient coordinates = [dk, beta8].

Across all measured heads:
- numerical residual rank = 96;
- full-rank exact factorization therefore requires r=96.

Aggregate excluding h55:
- exact-coordinate authority HTRC->antipodal ~= 0.02584482;
- r8  ~= 0.12069978, arithmetic ratio ~= 0.12535;
- r16 ~= 0.08490309, arithmetic ratio ~= 0.25070;
- r32 ~= 0.04703435, arithmetic ratio ~= 0.50140;
- r48 ~= 0.03364005, arithmetic ratio ~= 0.75209;
- r64 ~= 0.02742943, arithmetic ratio ~= 1.00279;
- r96 ~= 0.02584786, arithmetic ratio ~= 1.50419.

Interpretation:
- r32 gives about 2x arithmetic reduction but loses a substantial fraction of the
  authority gain;
- r48 is a modest quality/compute compromise (~25% arithmetic reduction);
- r64 nearly reaches authority but is already arithmetic break-even;
- exact r96 is ~1.50x more arithmetic than the direct 9D coordinate map.

Therefore the current-query residual span is high-rank enough that exact factorization
cannot accelerate the dense all-block coordinate map.

## Relation to older research

This confirms and unifies previous evidence:
- first-order query-tangent reconstruction failed;
- shared query basis needed R48/R64 for acceptable quality;
- HTRC response-active shared subspaces failed at low rank;
- cross-fit sufficient-coordinate action SVD failed to generalize.

These are manifestations of the same high-rank query-side geometry.

## Decision

CLOSE query/residual-rank compression as the primary HTRC acceleration mechanism.

Do not run more rank sweeps or new PCA/SVD variants on Q/residuals without a new
mathematical object.

Next research must reduce the number of source nodes/blocks whose local sufficient
coordinates are evaluated, rather than factorizing the dense all-block map.
