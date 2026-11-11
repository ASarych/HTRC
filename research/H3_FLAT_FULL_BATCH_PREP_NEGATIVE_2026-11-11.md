# H3 Flat 56 logical-head full preparation A/B: precision PASS, speed FAIL

Date: 2026-11-11. User GPU evidence:
`flat_full_prep_batch_ab_20261111-195815.json`.
Device: NVIDIA RTX 5070 Ti (16 GiB). Candidate:
`353a00f42e00e62b169f555c639bfe09808639a5`.
Pinned BEST: `e47cc7a310936547ab6c7d97bd07526eb6a135fe`.

## Scope and correctness

- One H3 block0/call1 capture, distinct raw heads 1/16/27/55.
- Each native head repeated 14x to 56 **logical work items**.
- Fast mode, 16 query tiles/head (1024 Q), 40752 full K + 7 tail.
- Streamed groups H12/H12/H12/H12/H8, budget 1024MiB.
- Both paths start from captured **CPU Q/K/V** and produce final fp32 attention.
- 5/5 final output parity PASS, each rel-L2=0, max_abs=0.
- Four independent full `flat_attention` oracle checks PASS (exact).
- No OOM, execution errors or missed logical heads.

## A/B: synchronized CPU+GPU full group wall medians

| Group | BEST full wall (ms) | New full wall (ms) | BEST/new | BEST prep (ms) | New prep (ms) | BEST Tile+Merge wall (ms) | New Tile+Merge wall (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H12 #1 | 110.658 | 127.479 | 0.8680x | 104.096 | 123.168 | 6.562 | 4.196 |
| H12 #2 | 108.329 | 126.770 | 0.8545x | 101.350 | 122.216 | 6.729 | 4.448 |
| H12 #3 | 109.291 | 126.393 | 0.8647x | 102.776 | 121.911 | 6.583 | 4.669 |
| H12 #4 | 109.010 | 121.201 | 0.8994x | 102.381 | 116.648 | 6.661 | 4.371 |
| H8 | 73.352 | 83.894 | 0.8743x | 68.592 | 80.694 | 4.907 | 3.130 |

Sum of per-group median synchronized wall times (not a single
end-to-end 56-head run): pinned **510.640ms**, batched
**585.737ms**, ratio pinned/new = **0.87179x**, new is
**14.71% slower**. Tile+Merge wall is faster in every group,
but preparation loses more than that improvement. Previous
3.3508x CUDA-event Tile+Merge alone remains valid and separate.

## VRAM pressure and diagnostic hypothesis

With 12 heads, prepared final state tensors are identical
(370.516MiB) yet peak PyTorch allocated rises from
453.313MiB to **785.522MiB**, +332.209MiB.
With H8, pinned 329.197MiB, batched 525.306MiB.
Code inspection suggests the batched preparation's whole-group
`kbf.float()`, `kf/KS` and contiguous group K/V staging
may cause very large temporary FP32 allocations and bandwidth.
This **hypothesis has NOT been isolated by stage profiling**.

## Decision

- Preserve head-flattened Tile and tail-Merge as accurate microbenchmark wins.
- **Reject promotion of the current group-wide preparation** to cumulative BEST.
- New next experiment should isolate K INT8 quantization cost and peak.
- Compare new fused quantization / reduced temporary buffers against
  current batched prep and pinned BEST, and record stage wall times.
- A 56-distinct-native-head capture, full Q, graph and actual H3
  model-resident performance remain unverified.
- Previous compact `dq8/sdq` query-cache experiment is still
  GPU UNRUN and must remain in research queue, not forgotten.

