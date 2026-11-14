# Research: native MiniMax-H3 one-pass compact QKV staging

**Scope:** HTRC Flat Graph research only. This is not production H3 attention,
not a tested quality replacement, and not a validated end-to-end speedup.
The default native-H3 Graph strategy remains `repeated_group`.

## Why

The first real 2026-11-13 ComfyNodes/NVFP4 canary measured a **single 8-head
group's** genuine 4096-row chunked fused `qkv_proj` on 40,759 tokens:
10 projection calls and 26.680 ms GPU. The original stock full attention
(QKV + norm/RoPE + attention + output projection) took 522.210 ms, but
those timings must not be directly compared. The previous Graph prototype
called the *full* fused projection again for each of 7 head groups: 70
projector calls per H3 attention, before the Graph's compute.

## Implemented design

- `h3/htrc_flat_h3_compact_once.py` uses the original model's
  `attn.qkv_proj(x[start:stop])` **exactly once per 4096-row chunk**,
  then Comfy's native fused Q/K RMSNorm+split-half RoPE.
  Original QuantizedTensor/NVFP4 checkpoint weights are never touched,
  sliced, dequantized or repacked.
- All Q/K/V heads are compressed into a temporary per-forward
  `torch.int8 [3,H,N,128]` staging tensor with FP32 scales
  `[3,H,N]`, one per token/head/Q-or-K-or-V component.
  The staging is explicitly *lossy* and makes no visual or parity claim.
- A group provider reads only the 8 selected heads. The existing
  Graph-owned K/V tensors `[1,8,N,128]` and Q window tensors
  `[8,P,128]` are filled by direct `torch.mul(...,out=...)`
  dequantization in BF16. **No additional QKV projector call** is
  made when moving among head groups or query windows.
- The packed staging is constructed fresh for every model forward.
  Never cache activations across timesteps, models or prompts.
- `h3/htrc_flat_h3_native_projection.py` accepts explicit research
  `projection_strategy="compact_int8_onepass"` and still defaults
  to `"repeated_group"`. Only the existing `compare_only=True`
  HTRC Graph path may consume this new group provider.
- An explicit staging cap `APL_HTRC_COMPACT_STAGE_CAP_MIB=1536`
  and a free-VRAM guard of packed stage plus 384 MiB fail closed.
  They do not override the Graph's separate allocator/headroom gates.

## Estimated storage at measured shape (N=40,759, H=56, D=128)

| Storage | Computed MiB |
|---|---:|
| All Q/K/V as BF16 | 1,671.756 |
| All Q/K/V as signed INT8 | 835.878 |
| FP32 per-row/head/component scales | 26.121 |
| **Persistent compact all-head stage** | **861.999** |
| Legacy single 8-head BF16 Q/K/V staging | 238.822 |
| Temporary 4096-row full fused BF16 output | 168.000 |

These are exact tensor byte counts, **not measured peak VRAM**.
Temporary FP32 packing work, graph-owned K/V buffers, graph pools,
attention output, quant kernels and transfer/allocator effects still
consume GPU memory. The compact stage is larger than a single group,
but avoids six additional *whole-model fused QKV passes*.

## Verification plan

1. CPU-only `python -B tools/test_h3_compact_onepass_cpu.py`:
   actual pack/decode implementations; exact projection call counts;
   two head groups; padded Q; fresh per-forward staging; cap failure;
   Graph default/policy structural checks.
2. ComfyNodes read-only canary: one stock H3 block and original denoise
   output, compact stage built once and all seven groups decoded.
   Record **build GPU ms**, **seven-group decode GPU ms**, **incremental
   peak allocated/reserved VRAM**, and persistent stage bytes in the
   same `h3_native_pre_qkv_canary.json`. Do **not** run Graph, generate
   a second video, or spend cycles on the earlier 4-5% comparison yet.
3. Only if measurements justify the storage tradeoff, profile faster
   fused packing/unpacking kernels. Then test full Graph in a separate
   opt-in research gate and **later** judge generated-video quality.
   Do not promote as a production replacement without those steps.

## State

CPU synthetic prototype (same pack/unpack logic) passed 3 tests on
a non-CUDA runner. Repository exact-source CPU preflight and actual
RTX 5070 Ti GPU timings have not yet run on this new HEAD.
Historical stock full-N-to-short-GEMM numerical mismatch is not a
blocking quality verdict and is not re-tested by this speed gate.

## 2026-11-14: first real compact one-pass GPU report

The user's `h3_onepass_diagnostic.txt` contains a successful
`MEASURED_NATIVE_SINGLEPASS_COMPACT_STAGE` one-block H3 canary,
from the genuine NVFP4 HyperFlow model on RTX 5070 Ti,
40,759 tokens / 56 heads / groups of 8 / 4096-row chunks.
The original model attention output was returned unchanged.

| Exact measured value | Result |
|---|---:|
| Original complete Kitchen attention, block 0 | 554.165 ms |
| Legacy one 8-head group fused QKV | 20.872 ms |
| **Compact all-56-head projection + Torch INT8 packing** | **84.151 ms** |
| **Decode all seven groups** (7 K/V groups, 70 Q windows) | **7.593 ms** |
| **Pack/build + 7-group decode** | **91.744 ms** |
| Packed INT8 + FP32 scale allocations | 903,871,584 bytes = 862.0 MiB |
| Incremental peak CUDA allocated during compact build | **1,254.874 MiB** |
| Incremental extra reserved allocator pool | 0 MiB |
| Actual H3 fused projector invocations for compact stage | 10 |
| Fused projector calls avoided vs 7-group repeated method | 60 |
| Original denoise first step | 32.25 s (stock attention; no Graph) |

An arithmetic seven-fold extrapolation from this run's
20.872-ms legacy one-group producer gives ~146.11 ms for
sequentially repeated QKV, before Graph. The measured
compact pack+unpack is ~1.59x faster against **that estimate**
(54.36 ms less) but takes ~836 MiB more peak memory
than the separately measured old one-group producer
(418.635 MiB extra). The seven-group repeated method
was not separately benchmarked end-to-end; these numbers
are NOT full model speedup or video quality evidence.

## Next experiment: one Triton pack instead of FP32 PyTorch intermediates

In the initial Torch staging path, every one of Q/K/V
calls `values.float()`, computes per-token/head absmax,
normalizes and rounds FP32, converts to new INT8, then
copies into the packed buffer. It makes a ~112 MiB FP32
temporary per full-size Q/K/V part and runs multiple
independent PyTorch kernels. The compact *persistent*
buffer is already only 862 MiB; its transient FP32 pack
path is the next speed/peak-memory target.

`h3/htrc_flat_h3_pack_triton.py` adds a **separate opt-in**
Triton kernel that jointly packs Q/K/V for 16 rows ×
4 heads per CTA. A single launch per QKV projector
chunk reads the genuine post-RMSNorm/RoPE contiguous BF16
fused output and writes INT8 activations with one FP32
scale per token/head/part into the EXISTING buffers.
It avoids creating full FP32 activation tensors.
The NVFP4 model weights and Comfy qkv_proj remain
unchanged. Pack scale uses the same absmax/127 clamp
and round-to-nearest scheme as the Torch fallback.
No source-relative/cross-step QKV activation caching.

Selection: `APL_HTRC_QKV_PACK_BACKEND=triton`.
Default remains `torch`. Fast path fails closed if:
- not on a real CUDA tensor or without native RoPE;
- native fused Q/K normalization returned different
  allocations rather than in-place views of the source;
- Triton package/round instruction unavailable;
- input or staging layout/dtype/shape changed.

The safe ComfyNodes canary preflights the Triton
import and source CPU tests **before** loading the
checkpoint. The kernel uses explicit `D=128` constexpr
and `triton.language.extra.libdevice.nearbyint` for
CUDA's ties-to-even quantization (rather than unsupported
`tl.round`). It asserts the normalized Q/K and original
V still alias the real fused output before running; any
future Comfy API change fails closed. The report explicitly
names `activation_packing_backend="triton"`, and the
launcher rejects a report naming the fallback backend.

A real GPU benchmark of the Triton fast path is
**UNRUN**; no speedup or memory reduction is claimed
until it runs. The previous 91.744-ms Torch figure
is the local control, not a guarantee.

## 2026-11-14 01:07 — Real output-producing HTRC Graph capture failure

The user's `h3_live_graph_diagnostic_010512.txt` contains the
real six-step HyperFlow NVFP4 **Live Graph** failure at block 0:

- `h3_native_graph_live.json`: `FAIL_CLOSED_NO_STOCK_FALLBACK`,
  `blocks_completed_first_cycle=0/49`,
  `complete_cycles=0`, `attention_calls=0`,
  `total_graph_gpu_event_ms=0`. No HTRC denoise/video result.
- Exact traceback:
  `htrc_flat_directflow_live._new_session` captures
  `perform()` using `with torch.cuda.graph(graph)`,
  then `htrc_flat_head_batch.prepare_batched_head_states`
  invokes `triton_module._stats_kernel`. The **first**
  Triton pointer is `MU.view(h*centers,dim)`, a transient
  `grouped_q.mean(...).contiguous()` CUDA tensor created
  **inside** the capture.
  CUDA driver reports `ValueError: Pointer argument (at 0)
  cannot be accessed from Triton (cpu tensor?)`.
  This message does **not** prove that the model QKV was
  CPU-resident: CUDA Q/K/V providers, warmups and workspace
  were already GPU-resident. A capture-private allocation
  / pointer lifetime incompatibility is the leading
  explanation.
- After failure the backend emits
  `CUDAMallocAsyncAllocator: Attempting uncaptured free of
  a captured allocation`, then Windows fatal
  `0x80000003`. This confirms a **capture/cleanup hazard**,
  not a numerical quality or model-weight issue.
- Native model load and 47/47 file-cache warmup were
  successful. Original attention was **not used as a
  fallback**, as explicitly required.

### Capture-safe research path (opt-in only)

`APL_HTRC_GRAPH_CAPTURE_SAFE_MU=1` changes the **HTRC
Direct-Flow capture staging**, not Comfy's global attention
or NVFP4 model weights:

1. `_new_session` preallocates persistent CUDA tensors
   `ws["MU"]`, `ws["_TOPK_VALUES"]`, `ws["_TOPK_IDS"]`
   **before** `torch.cuda.graph`.
2. `_one_window` opts into the already implemented native
   Triton `_direct_q_mu_kernel` on those stable pointers,
   instead of allocating a fresh `torch.mean` result in
   capture. When the environment knob is not set,
   `native_torch_mu` remains unchanged.
3. The top-k implementation explicitly writes FP32 values
   and INT64 indices into existing GPU workspace via
   `torch.topk(..., out=(topv,ids))`; `idx.copy_(ids)`
   casts into persistent INT32 stage without another
   short-lived GPU allocation.
4. Both source and runtime validate dimensions, dtype,
   device and contiguity before launching Triton.
   Any missing capture-safe buffer fails closed.
5. A separate *small* GPU test,
   `tools/smoke_h3_graph_capture_gpu.py`, executes two
   HTRC Graph capture/replay calls on 8192 tokens
   (balanced mode's 512 aligned 16-token leaves)
   and 8 heads, no H3 checkpoint, under a fresh process.
   The safe ComfyNodes launcher runs this BEFORE moving
   any user checkout or loading HyperFlow.
6. The import-free CPU source contract
   `tools/check_h3_graph_capture_safe_cpu.py` checks the
   real MU/topk preallocation, opt-in gating and smoke
   wiring without CUDA.

**Status:** Source-level repair authored on research
branch. CPU and small CUDA smoke on the user's exact
Windows/Triton build, full 40,759-token HTRC Graph,
whole denoise, peak VRAM and video quality are all
**UNRUN**. Do not claim this fixes the CUDA driver until
the small capture smoke and real one-block graph pass.
Original `torch` QKV pack remains selected for
the live model path.

### 2026-11-14: model-free smoke exposed optional `kern.group`

The first independent Windows CUDA Graph smoke correctly stopped
**before HyperFlow was loaded** with:

```text
KeyError: 'group'
htrc_flat_directflow_live.py, _new_session
centers = geom.tile_count // int(cfg["kern"]["group"])
```

This is **not** evidence of a CUDA pointer regression. The new
capture-safe preallocation mistakenly required an explicit
`cfg["kern"]["group"]`, while balanced/fast Flat config permits
the missing key and uses default `1` elsewhere in the same module.
It now uses
`int(cfg.get("kern", {}).get("group", 1))`.
The import-free CPU source gate explicitly rejects a reintroduction
of the direct indexed form. No Graph memory/performance or video
claim is possible from this early smoke failure.

The **next** small CUDA smoke may progress to new kernels and
may expose other independent capture issues. If that happens,
report the next exact traceback rather than declaring the entire
live path tested. CPU/real GPU execution on this new HEAD remains
**UNRUN**; the unchanged fallback policy is fail-closed.

## Next live denoise speed slice: eliminate per-block CUDA waits (2026-11-14)

The first completed six-step H3 Graph result in the user's
`h3_native_graph_live.json` is the new functional baseline:
**294/294 HTRC attention calls, 49/49 blocks, 6/6 cycles**,
`stock_attention_output_used=false`, no error, one video.
Cumulative attention CUDA events: **130,478.51 ms**;
per-block synchronization wall: **145,799.46 ms**.
First block's CUDA capture-heavy event: **9,769.74 ms**,
first-block wall: **16,030.95 ms**.
Observed CUDA allocated peak: **13,472.34 MiB**,
reserved peak: **14,368 MiB**.
Video quality is provisionally sufficient for continuing
speed research; exact dense parity has not been proven.

Two *independent* CPU/GPU barriers occurred for every one
of the 294 real attention calls:
- `CompactAllHeadsQKV._build` recorded an ending CUDA event
  and called `end.synchronize()` before Graph staging.
- The ComfyNodes live adapter recorded the entire attention
  call and called `stop.synchronize()` before returning
  the tensor to the next transformer operation.

This is **588 explicit barriers** in the prior full-video
integration; GPU can also have other barriers inside initial
HTRC Graph capture and verification. The difference between
145.8s CPU wall summed across attention calls and 130.48s GPU
events is **not** purely attributable to synchronization:
first capture, scheduling, allocations and Python work differ.

### New strict opt-in `APL_HTRC_COMPACT_DEFER_GPU_SYNC=1`

The genuine original NVFP4 `qkv_proj`, RMSNorm/RoPE,
per-token/head INT8 packed activations, Graph window
geometry and GPU dataflow remain unchanged.

`CompactAllHeadsQKV` now holds a `(start,end)` CUDA event
pair instead of synchronizing the end event **only** when
explicitly opted in. The internal native Graph return
has a private `_deferred_qkv_cuda_events` pair so the
ComfyNodes live adapter can read the genuine QKV GPU time
at the *end* of the 49-block transformer cycle.
No CUDA event or tensor is held across cycles after the
report is committed; the same pack buffer is refreshed
for every attention forward as before.

The default path still executes `end.synchronize()`,
returns immediate `build_gpu_ms`, and works as an
independent read-only GPU canary. The deferred path
returns `build_gpu_ms=None` until the owning live adapter
finalizes the events; report semantics explicitly distinguish
CPU submit wall time from wall including synchronization.

The paired ComfyNodes opt-in
`APL_H3_NATIVE_GRAPH_BATCH_TIMING=1` verifies the
all-49-block cycle and performs a single CUDA barrier
at its final block, with per-cycle GPU sums for QKV
and Graph+out_proj. Failure, missing events or partial
cycles reject success and must never fall back to stock.
The safe launcher sets both knobs together and runs CPU
functional/source gates before model loading.

**Measured speedup on RTX 5070 Ti: UNRUN.**
Previous ~415-ms warmed GPU attention per block is
a control, not a promised result. The new latency
comparison must use full six-step denoise and preferably
a matched stock backend; isolated QKV numbers alone
cannot justify an end-to-end speedup.

## 2026-11-14 04:19 — real six-step batched-sync A/B: REGRESSION

Actual user's `h3_native_graph_live(1).json` completed all
**294** output-producing HTRC attention calls with
`stock_attention_output_used=false` and `error=null`.
The new `BATCHED_49_BLOCK_GPU_EVENTS_WITH_ONE_SYNC_PER_CYCLE`
mode, however, did **not** accelerate the real pipeline.

| Measurement | Prior per-block sync | New per-cycle sync |
|---|---:|---:|
| 294 HTRC attention GPU-event sum | 130,478.511 ms | 184,781.478 ms |
| Mean GPU-event sum of warmed cycles 2–6 | 20,334.022 ms | 32,378.528 ms |
| CUDA max allocated | 13,472.342 MiB | 13,472.342 MiB |
| CUDA max reserved | 14,368 MiB | 16,384 MiB |

Regression is **+41.62%** in summed GPU-event time;
warmed steps are **+59.23%** slower. This is an
end-to-end *attention* measurement across all real
denoise blocks, not a QKV-only benchmark. It is NOT
a direct complete-video wall-time A/B because the
second run lacks an attached run-controller wall log.

New GPU event decomposition:
`QKV+INT8 pack 47,398.711 ms` (**25.65%**),
`Graph + native out_proj 137,382.766 ms` (**74.35%**).
By cycle, HTRC GPU-event totals are
22.889s, 28.244s, 19.874s, **50.961s**, 28.299s,
34.515s. The per-cycle first-to-last attention wall
spans sum to **609.430s** and include intervening model
work; they are NOT additive to denoise time.
Reserved memory reaches the entire 16,384 MiB card
reported by PyTorch by the second cycle. GPU allocator
pressure/offload and host/GPU pipeline interactions are
plausible contributors, but this JSON alone cannot
identify the unique cause. Do NOT claim that removing
CPU waits is always faster or that VRAM paging is proven.

**Decision:** Do not use cycle-batched synchronization
as the default on 16-GiB RTX 5070 Ti.
Keep `APL_HTRC_COMPACT_DEFER_GPU_SYNC=0` as the
known-working HTRC behavior. Its live ComfyNodes
launcher now also defaults to synchronous
`Attention.forward`, with optional telemetry collected
from the same already-synchronized CUDA events.
The full-cycle batch experiment requires a **separate**
explicit `-BatchTiming` flag and fails closed if
the PyTorch reserved-memory peak exceeds 15,360 MiB,
before propagating a partial result. No production
promotion or new performance claim. Exact dense parity
remains UNTESTED; video quality was provisionally
accepted on the prior synchronized path.

Next substantive target is the **Graph+out_proj** hot
path (three quarters of measured HTRC GPU time),
especially the cost of per-query-window Graph replay
and KV staging, not blind further QKV quantization.
Any proposed tiles/window/group experiment needs
an independent model-free memory gate before an
expensive six-step video.

## 2026-11-14 — research 32-tile full-Graph replay slice

The validated output-producing H3 path is still
`balanced / tile=64 / tiles=16 / 8-head group / per-block
CUDA sync`. The preceding batched-sync experiment
was **41.6% slower** and remains opt-in only.
This slice changes **only** the query-window batch
size for a separately selected **single real denoise
step**, without changing HTRC Flat math, exact leaf
count, real NVFP4 model QKV, RoPE, INT8 packing,
out_proj, or production backend.

The actual 40,759-token reference has 56 heads in
seven groups of eight. Fixed 64-query-row tiles:

| Research geometry | tiles16 control | tiles32 candidate |
|---|---:|---:|
| Windows per 8-head group | 40 | 20 |
| CUDAGraph replays per attention | 280 | 140 |
| Replays for 294 full-video attentions (illustrative) | 82,320 | 41,160 |
| Estimated 8-head persistent staged memory | 464.79 MiB | 756.65 MiB |
| Estimated stage increase | — | +291.86 MiB |

These are geometry and **analytical stage estimates**,
not measured kernel throughput or full VRAM peaks.
The private CUDA Graph capture pool is unmeasured,
so more expensive/worse GPU dispatches or VRAM pressure
remain possible. Fewer Graph replays do not imply
halved compute or doubled model performance.

The opt-in
`APL_HTRC_DIRECTFLOW_TILES32_ADMISSION=1`
performs a **pre-capture** check for exactly one
8-head Graph session using the existing
`graph_stage_head_bytes` model and actual
`torch.cuda.mem_get_info`. It requires
`free >= estimated_8_head_persistent_stage +
1024 MiB` before creating any graph-owned K/V
buffers and prints the actual free/stage estimates
on refusal. The downstream post-capture actual
allocator gate remains mandatory. The separate
ComfyNodes launcher sets
`APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB=1152` and
`APL_HTRC_DIRECTFLOW_MIN_FREE_MIB=768`
for the Tile32 research arm (Tile16 retains
768/256 MiB). The 1024-MiB early margin is
only an engineering guard, **not** a proof
of adequate physical VRAM on Windows.

Preflight layers, with **NO H3 model load first**:

1. Import-free
   `tools/check_h3_graph_tiles32_cpu.py` asserts
   true 40-vs-20 windows, 280-vs-140 replay count,
   persistent memory delta, CUDA-capture admission
   and fail-closed logic.
2. Previous real small CUDA Graph smoke for the
   currently selected branch remains unchanged.
3. New `tools/smoke_h3_graph_tiles_ab_gpu.py`
   runs genuine CUDA Flat graphs on identical
   random BF16 Q/K/V with 8319 tokens, 8 heads,
   **including the short padded tail**.
   Each geometry captures and replays twice.
   Same-mode replay and 16-vs-32 output must be
   **bit-exact**; otherwise it aborts the test
   before loading any HyperFlow weights.
   No production quality claim from synthetic QKV.
4. The ComfyNodes script
   `run_h3_native_qkv_canary_safe.ps1`
   requires explicit `-LiveGraph -Tile32`,
   refuses `-FullVideo` and `-BatchTiming`
   together with Tile32, and performs just
   one complete **49-block** H3 denoise forward
   after all preflights pass. It uses unique
   recovery paths and restores the original
   local ComfyNodes directory.

The live JSON reports the selected tile geometry
and calculated replay count per attention; per-block
real GPU times still include first Graph capture
where relevant. Compare warmed blocks 1–48 with
the old synchronized 16-tile control before
deciding whether a full six-step A/B is warranted.

**Execution status on RTX 5070 Ti:** CPU/GPU and
model Tile32 tests **UNRUN**. No optimization,
speedup, video parity or memory-fit claim.
