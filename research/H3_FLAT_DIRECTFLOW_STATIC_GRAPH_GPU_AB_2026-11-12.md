# HTRC Direct-Flow fixed-pointer CUDA Graph GPU A/B PASS (2026-11-12)

Source of truth: original uploaded [JSON](evidence/flat_directflow_graph_static_pass_20261112.json).
Captured code base `8a4f7a6ab4ae26bfc15860fe7069424849c9db32`, RTX 5070 Ti (16GB), fixed fast: 256 exact leaves, group1, native BF16 QKV, 11 true distinct synthetic heads from a 56-head source, 18,723 tokens, 25 rounds, four warmups.

`completed=true`, `errors=[]`. For both 1,024-row and 291-row windows, `IDX`, `EXCL`, `MU` and `OUT` were **bitwise identical** between eager and CUDA Graph replay.

| Window | real rows | GPU event speedup | sync CPU+GPU speedup | graph capture wall ms | calculated break-even replays | measured positive reserved VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|
| full | 1024 | 1.0298× | 1.8251× | 7.380 | 12 | 0.000 |
| short | 291 | 1.4743× | 2.4328× | 4.153 | 8 | 0.000 |

* Full: eager GPU `0.731328` ms, graph GPU `0.710144` ms; eager sync wall `1.414300` ms, graph `0.774900` ms.
* Short: eager GPU `0.480128` ms, graph GPU `0.325664` ms; eager sync wall `0.947800` ms, graph `0.389600` ms.
* `graph_extra_reserved_mib` was **0.0 for both cases in the report** but is computed with `max(0, after_reserved-before_reserved)` using allocator reservation. This is **NOT** proof that a graph has zero private VRAM footprint. The short test even shows a negative signed reservation delta when calculated from before/after; it may reflect allocator cleanup. Need future measurement of total high-water memory with graph alive and memory cost of persistent mutable input copies.
* Capture break-even values count replay savings for the SAME fixed pointers and do not include Q/K/V transfer, K8/KS refresh, per-block/step shape management, nor graph pool amortization across model layers.
* This is synthetic single-group repeated-input performance, **NOT** model-resident, 56-group full-attention latency or real video. No claim of actual H3 acceleration is supported.

Decision: **PASS_NUMERIC / PASS_STATIC_REPLAY_FEASIBILITY**. Next isolate mutable-input graph with genuine QKV changes between replays. Keep all mathematical kernels untouched and no default production promotion.
