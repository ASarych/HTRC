# Flat Q-preparation: reduce launches and traffic, NOT quality

Date: 2026-11-12. Exact code base:
`experiment/flat-qprep-fixed-math-20261112@69bc7b35a8251029084dc690d2e9f40de5c84a0d`.
The user expressly rejected *all* Q speed/quality coefficient tuning.
No lowering `exact_leaves`, no artificial larger `center_group`,
no skipped query rows, no approximation introduced to buy speed.

## Observation from the real H3 video trace

For the user-run 56-native-head, 18,723 token H3 attention
(19 windows, 5 head groups; about 95 group-window iterations):
`profile_call_wall_ms=169.08`, instrumented
`query_stats_selection=112.17 ms`, Tile 17.21 ms,
Merge 2.95 ms and K/V+workspace setup 9.34 ms.

**These GPU event intervals include possible stream gaps** and
are NOT pure arithmetic kernel runtimes or an additive breakdown.
The source currently repeats a chain of:
Q index_select+permute+BF16 to FP32+mean,
`_stats_kernel` (G/VB/LZ/SG),
`_rowmax_kernel` (SEL),
`torch.topk`, int32 IDX conversion, EXCL clear/scatter,
Tile, Merge, BF16 output copy for every group-window.

The old zero-QKV shape microprofile has 0.338 ms
Q gather+cast+MU, 0.396 ms stats, 0.368 ms rowmax/topk/
exclusion per group-window. These are instrumented synchronized
medians, *not* full-model per-call durations. Re-read them after
the next real model-resident trace.

## Numerical acceptance is layered, not arbitrarily bitwise

- **Pure Q input transport / `view_copy`**: require bit-exact
  FP32 Q and all resulting states and output. There is no
  mathematical change, so any mismatch is a bug or missing
  deterministic assumption. Keep opt-in until GPU PASS.
- **Reordered FP32 reductions / Triton fusion**: do NOT require
  bitwise equal `MU/G/VB/LZ/SG/SEL`; use defined
  `rtol/atol`, rel-L2, max-abs and high-percentile errors
  with an unmodified fixed-Flat reference. Calibrate tolerances
  to baseline FP32/BF16 noise rather than choosing permissive
  thresholds for a desired speedup.
- **Near-field selected leaf IDs**: require same top-k set
  wherever scores are well separated. Treat near ties separately:
  log the selection-margin distribution and different IDs; do
  not silently pass changes that redirect meaningful attention.
- **Final attention**: compare every output row on representative
  real post-RoPE QKV; then run same-prompt/seed frames and
  temporal quality against an accepted reference.
  Passing a tolerance against approximate Flat does not prove
  native Dense/Sol fidelity. No quality regression is accepted.
- Freeze `exact_leaves`, centre grouping, precision paths,
  K8 and exact tail handling for each isolated engineering A/B.
  Record full config and commit SHA, one JSON per experiment.

## Optimization order by risk and likely payoff

1. **Q pack + MU fused Triton kernel.** Source is native
   GPU `[S,H,128]` view. Produce contiguous FP32
   `[H,centres*64,128]` plus FP32 per-tile MU in one
   launch, eliminating `index_select` and separate
   `torch.mean`, and avoiding intermediate BF16 gathers.
   Reuse fixed workspace, preserve partial-final-tile MU
   semantics. Test 4+56 unique real-style heads, full
   windows, EXACT Q and numerically equal MU.
2. **Reduce per-window Python dispatch / CUDA Graph trial.**
   Stable group workspace is already available. Profile
   kernel-only GPU occupancy vs enqueue gaps, then compare
   eager to captured/static-address per-group window replay.
   Capture one geometry for full windows, separate last
   partial; K/V cache is per actual attention call, not
   across denoise steps. Account for extra graph-pool VRAM
   on 16 GiB and JIT/cold-start cost, do not assume a win.
3. **Fuse leaf stats with rowmax SEL scoring** in a
   Triton experiment. `_stats_kernel` has g/lz/sg in
   registers; `_rowmax_kernel` currently reloads
   quantized G and LZ from global memory. Computing
   equivalent SEL before leaving the stats pass could
   remove a redundant G read and one kernel launch.
   BUT stats uses BT=16/BL=4 while rowmax needs Q rows
   and per-leaf maxima; the fused kernel might exceed
   register limits or reduce occupancy. Bench before
   calling it better, retain G/VB/LZ/SG for Tile/Merge.
4. **Combine selection packaging**, not leaf reduction.
   `torch.topk` -> int32 IDX -> zero EXCL -> scatter
   creates several launches and intermediates. Trial
   a Triton selection+IDX+EXCL stage, preserving the
   same kx and selected set. Sorting 1170+ leaves and
   emitting 256 exact indices may be slower than PyTorch;
   do not replace by an approximate top-k merely to win.
5. **Fused Merge -> final BF16 BTHD output** can remove
   an extra FP32->BF16 copy and transient output buffer;
   lower potential gain, work after dominant Q prep.

At each stage compare **complete full-attention call wall time**
and peak VRAM with checkpoint weights resident after numerical
gates. Use one unified JSON, don't extrapolate a 0.3 ms
microkernel saving to a 49-block, six-step video until observed.

## Next concrete implementation gate

The first isolated `view_copy` alternative and full-window
parity tool are already committed on the current branch.
GPU A/B was **UNRUN when this plan was written**. The subsequent
uploaded run PASSed bit-exact parity for 4+56 distinct native heads
but showed no useful preparation speedup (56-head sum-median ratio
`0.996173x`). Retain gather as default; do not polish view_copy.
Proceed with an opt-in direct-native QKV access experiment, then isolate
eager-vs-Graph dispatch, then test the combined Direct-Flow path.
No quality-coefficient search.

No change to production BEST, model weights, or ComfyNodes
loader. This note is design only; new fusion kernels UNRUN.


## Direct QKV + Direct-Flow follow-up (2026-11-12)

See [unified architecture and A/B gates](H3_FLAT_DIRECT_QKV_FLOW_PLAN_2026-11-12.md). These two methods are complementary but must be benchable independently. Neither is implemented or GPU tested by the uploaded view_copy PASS; production and global BEST unchanged.
