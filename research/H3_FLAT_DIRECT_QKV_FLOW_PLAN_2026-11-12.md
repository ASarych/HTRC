# Flat HTRC Direct QKV + Direct-Flow — unified design, independent gates

Date: 2026-11-12. Base: `d996053cf48da20c9836dd4427ee0b306e3aa796` on `experiment/flat-qprep-fixed-math-20261112`.
Scope: design-only research follow-up; NO new GPU kernels implemented in this commit.

## Motivation and evidence

The real HyperFlow H3 18,723-token / 56-native-head profile showed 169.08 ms call wall and a 112.17 ms instrumented `query_stats_selection` GPU event interval. GPU event intervals include launch/stream gaps; they are not pure arithmetic kernel runtimes. The subsequent fixed-math `view_copy` test from the uploaded `flat_qprep_fixed_math_ab.json` is numerically exact (1,064 state checks, 21 final output checks) but performance-neutral/negative (56-head six-group sum of preparation medians: gather 4.955920 ms vs view_copy 4.974960 ms). This closes view_copy UNRUN only, not the larger architectural proposal.

## Core separation

- **Direct QKV** changes how the same Q/K/V values are addressed and staged: consume native CUDA BHSD tensors with explicit strides (views allowed), avoid intermediate BF16 head gathering and FP32 Q global materialization where possible. Compute any necessary FP32 operands in registers. Retain exact original K8/KS semantics and the same MU/leaf stats, selection, Tile, tail and merge mathematics.
- **Direct-Flow** changes dispatch and buffer lifetime: preplanned per-head-group and per-query-window execution, fixed workspaces, possible CUDA Graph replay and/or a bounded producer-consumer queue with slots, without a Python round trip per stage/window. Direct-Flow must be able to run either staged or direct QKV.
- Flags should stay orthogonal for A/B: `qkv_source=staged|direct` and `dispatch=eager|graph|flow`; unsupported combinations fail explicitly, not silently fall back.

### Implementation sequence within one architecture (do NOT stack unverified changes)

1. Native layout contract: trace actual H3 Q/K/V strides from `capture_from_bhsd`. No assumption that permute makes data contiguous. Verify BF16/FP16/FP32 and batch=1, 56 unique heads, dim=128, same-token K/V, RoPE already applied.
2. Direct Q: add stride-aware Q loads to MU/rowmax/Tile and final-tail consumer, preserving partial last window's TRUE-row mean. Avoid `index_select -> permute -> contiguous -> FP32` intermediate. Where exact bit parity of reductions cannot be retained, report numerical deltas rather than claiming bitwise identity.
3. Direct K/V: when native sources are BF16 and layout permits, avoid KBF/VBF duplicates; write K8/KS with an exact fused kernel reading native strides; stats and exact near-field kernels consume original BF16 V and required K values. Keep K8/KS lifetime strictly within one attention invocation. No cross-step caches.
4. Direct output: optional separate gate for Merge writing the final BF16 BTHD destination at native strides, including the <=15 exact trailing K tokens and partial final Q window. Measure before keeping.
5. Graph path: allocate stable window/group workspaces; compare eager vs graph using identical direct QKV data, static geometry and two shapes for full/partial windows. No graph capture that allocates inside the replay. Check model-resident headroom on 16 GiB RTX 5070 Ti and record cold JIT/graph-pool cost.
6. Flow path: schedule producer(stats+select) and consumer(Tile+Merge) for bounded slots with explicit dependencies. Do not launch one monolithic register-heavy kernel. Prevent read/write overlap, races and deadlocks. Never assume that producing the next query's statistics can outrun the consumer without measurement.

## Constant mathematical and validation contract

- Fixed `fast` baseline: 256 exact leaves, centre group=1, leaf=16, 64-row tiles, native token order, same near-field selected set and exact tail; no Q quality coefficient.
- Preserve the same quantization and BF16 rounding rules, masks, softmax mass merge and output shape.
- **Gate A**: original staged-eager vs direct-eager, exact output on pure transport where expected; report FP32 reduction-order differences and near-tie top-k IDs separately. Fail closed on selection changes outside genuine tie ambiguity.
- **Gate B**: direct-eager vs direct-graph, identical tensor states/output and full attention-call wall + GPU events + CPU enqueue gaps. Fail closed on graph memory/OOM.
- **Gate C**: direct-graph vs direct-flow, identical outputs on every group/window and tail. Test sanitizer/races and end-to-end call wall, not just a microkernel.
- **Gate D**: 56 genuinely distinct real post-RoPE heads, model weights resident, full H3 denoise/video and accepted visual/temporal reference. Flat-vs-Flat parity does NOT establish dense/Sol quality.
- One combined JSON per coherent A/B: base & candidate exact SHAs, geometries, source strides, head uniqueness, rounds/warmup, cold/warm timings, CPU and GPU timings separately, kernel launch counts, memory peak and parity.
- Record all failed/UNRUN gates explicitly. No promotion to BEST/main or production until real gates PASS.

## Status

| Gate | Status |
|---|---|
| view_copy (GPU data transport) | PASS_NUMERIC / NO_GO_PERFORMANCE, uploaded JSON |
| Direct QKV direct-eager | NOT_IMPLEMENTED / UNRUN |
| Direct QKV + Graph | NOT_IMPLEMENTED / UNRUN |
| Direct QKV + Direct-Flow | NOT_IMPLEMENTED / UNRUN |
| Model-resident full H3/video parity, latency, 16 GiB VRAM | UNRUN |

Smallest first code slice is **Direct Q** (stride reads + MU), not Direct-Flow and not view_copy. Direct QKV and Direct-Flow can be delivered together as one architecture but must remain independently measurable before combination.
