# H3 Historical Result Inventory — 2026-09-26

Purpose: prevent re-running already explored mathematics under new names.

## Recovered historical JSON families

Recovered Library artifacts show that the earlier H3/RRC3 math line already
tested the following families on representative heads, mostly around the
historical block-0 raw capture:

1. **Block-mass cumulant residual models**
   - format: `apl-h3-rrc3-block-mass-cumulant-residual-v1`
   - compared zero/static calibration, Gaussian m2, Taylor-log m2,
     cumulant3/4, Taylor4-log;
   - the recovered artifact explicitly uses
     `attention_block_00_call_01.pt`;
   - representative heads: 0,1,8,16,27,40,48,55.

2. **DCA response-direction family**
   - `apl-h3-rrc3-dca-response-directions-v1`;
   - even router rows for calibration, odd rows for evaluation;
   - response-direction count sweeps and grouped-source reduction.

3. **DCA multivariate response subspace**
   - `apl-h3-rrc3-dca-multivariate-response-subspace-v1`;
   - same representative-head set and even/odd split.

4. **DCA output-aware / output-metric keyspace**
   - `apl-h3-rrc3-dca-output-aware-subspace-v1`;
   - `apl-h3-rrc3-dca-output-metric-keyspace-v1`;
   - output-aware response geometry therefore predates the current
     softmax-aware rank-selection work and must not be treated as a new idea.

5. **K-side higher-moment direction selection**
   - `apl-h3-rrc3-dca-kside-t3-energy-v1`;
   - `apl-h3-rrc3-dca-kside-mixed-t3-t4-v1`;
   - K-side T3/T4 direction ranking was already explored.

6. **HTRC runtime adaptive residual span**
   - `apl-h3-rrc3-htrc-runtime-adaptive-residual-span-v1`;
   - rank sweep 8..128;
   - basis protocol used all current queries q-only;
   - coordinate system `dk_plus_beta8`;
   - measured sparse full-RRC3 interaction ratio near 0.00628
     (~159x reduction) in the historical setup.

## What is *not* new

Do not reopen these as nominally new directions without a materially different
mathematical question:
- generic cumulant / Taylor expansion of block mass;
- generic response-aware direction selection;
- generic output-metric keyspace;
- generic K-side T3/T4 mode ranking;
- generic adaptive residual low-rank span.

These may still be useful ingredients, but they already have historical
evidence and failure/success context.

## New evidence from the 2026-09-26 layer scan

The current all-head layer scan is materially different from the historical
representative-head work.

For sampled blocks the strongest alignment/logit role migrates:
- block 0: alignment head 55, logit head 55;
- block 8: alignment head 40, logit head 40;
- block 24: alignment head 40, logit head 0;
- block 40: alignment head 8, logit head 8;
- block 48: alignment head 40, logit head 27;
- block 49: alignment head 48, logit head 48.

Therefore the supported structural interpretation is not
"physical head 55 is globally special".

It is:
**a high-anisotropy/high-logit role exists and its physical-head carrier is
layer dependent.**

This is new relative to the recovered historical JSON set because those
artifacts used a fixed representative-head subset and did not perform an
all-head x multi-layer role scan.

## Consequence for next math

The next research question should not be another fixed-head variant of DCA,
cumulants, or response-aware mode selection.

The useful new question is whether the migrating role has a
**head-permutation-invariant signature** that can be detected cheaply from
K/Q geometry and whether the same signature predicts which already-tested
approximation family will succeed or fail in each layer.

Candidate signature variables already measured cheaply:
- alignment gain;
- centered-K RMS;
- leaf logit std;
- q norm;
- head rank within layer for those quantities.

Next gate:
1. correlate these structural metrics with approximation error using the
   existing cross-block/cross-call results;
2. test whether layer-local rank/percentile predicts error better than physical
   head identity;
3. only then design a new adaptive operator or routing rule.
