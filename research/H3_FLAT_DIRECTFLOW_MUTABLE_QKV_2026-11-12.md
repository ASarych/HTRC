# Direct-Flow with mutable Q/K/V: graph reuse across full 19-window call

Date: 2026-11-12. New branch: `experiment/flat-directflow-mutable-qkv-20261112`.
Base: `e25868966c16dfa58dc9bc1581decaa18c9c993e` — archived static-input graph GPU parity PASS.
This slice adds only a research benchmark + CPU static contracts. Default/live production unchanged.

## Question

Can 2 captured CUDA graphs (one per distinct full/short query-window geometry)
be replayed for **fresh Q, K, and V tensors**, without recapture, and yield
the same results as fresh native eager attention **after including all staging,
requantization, tail refresh and output copy costs**?

## Experiment

- Synthetic two fresh BF16 post-RoPE-style Q/K/V banks A and B, 11 genuinely
  distinct heads, 18,723 token positions, dim 128. Q, K and V ALL differ
  between banks; bank outputs MUST differ. Sources are CUDA-resident, and
  their pointers are not used by captured graphs.
- One persistent K/V BF16 staging pair per 11-head group, one K8 INT8/KS FP32
  cache with stable GPU addresses, and one persistent Q buffer/workspace per
  **distinct query window geometry** (16 full tiles and derived short geometry
  under default parameters).
- At each attention call: copy fresh K/V to staging, rerun the native-strided
  K8/KS quantizer on updated K, refresh exact 1..15 trailing K/V rows, and
  for each query window copy its Q to the correct buffer, replay the same
  captured graph, and copy real output rows to the full attention output.
- Eager uses exactly the same fixed Flat mathematical kernels, directly
  consumes each source bank without QKV staging, builds fresh K8/KS once for
  each call, and processes all 19 windows.
- A/B gates before timing: K8 and KS bitwise equality, MU/IDX/EXCL for first
  and final windows, entire [S,11,128] attention output bitwise equality
  separately for bank A and bank B; bank A output MUST differ from bank B.
- Timings (CUDA event, CPU submit and synchronized wall medians) cover the
  **full 19-window attention group** including Q/K/V updates, K8+KS rebuild,
  all window replays, and output copying. Graph capture and extra GPU reserve
  recorded independently. If Graph cannot beat eager after copies, report it,
  do not discount copy cost or hide its failures.

## Limits and guards

- Graph source tensor addresses are static only in staging, not model/source
  QKV addresses. This is a copy-based architecture; it may cost more VRAM or
  bandwidth than the original native eager path.
- Cache lifetime is one group/shape and must be externally scoped; not a
  global cache across unrelated model tensors or denoising steps. New K8 and
  tail contents are recomputed before EACH replay sequence.
- No model checkpoint loaded, no actual H3 49-block residency or video
  quality proven, no graph exec pointer updates, no production switch.
- Model-resident 16GB feasibility requires complete memory accounting after
  this synthetic test. Even if Graph wins in isolation, do NOT assume it can
  be resident alongside actual weights.
- No changes to exact leaves, center group, masks, selection, K8 rounding,
  Q coefficients, or performance-quality trade-offs.

## Windows PowerShell

Use the same proven `$py` CUDA+Triton interpreter.

```powershell
cd D:\ComfyNodes-flat-split-20261111
git fetch htrc
git switch --detach htrc/experiment/flat-directflow-mutable-qkv-20261112
& $py -m unittest discover -s tools -p test_h3_flat_directflow_mutable_contract.py
if ($LASTEXITCODE -ne 0) { throw "Mutable-input static contract FAILED" }
& $py -B tools\benchmark_h3_flat_directflow_mutable_qkv_ab.py --group-heads 11 --rounds 12 --warmups 4
```

ONE result: `D:\ComfyOutput\H3\math_probe\flat_directflow_mutable_qkv_ab.json`.
GPU parity + real copying/eager/replay timings **UNRUN**.


## Added executable CPU dataflow gates

In addition to AST checks, the CPU-only suite executes the benchmark's actual `copy_query_window` and `update_kv_stage` functions (extracted from the source via AST). The query test covers varied sequence lengths, head counts and tile capacities, including last-row repetition. A fake deterministic quantizer verifies changed K propagates into K8/KS and changed K/V into the exact tail. These tests do **not** execute Triton or prove CUDA Graph correctness. First-run Python unittest on the user's matching environment and GPU A/B remain **UNRUN**.

## Local 5070 Ti A/B: CUDA Graph PASS, CPU harness fixture FAIL (2026-11-12)

Raw local report archived as [`flat_directflow_mutable_qkv_gpu_pass_20261112.json`](evidence/flat_directflow_mutable_qkv_gpu_pass_20261112.json).

- **CUDA numerical: PASS** — `completed=true`, zero `errors`; 11 distinct synthetic BF16 heads, 18,723 tokens, 19 windows, **2 reused graphs**, two different sets of Q/K/V, different outputs for the two banks. Eager and replay match bit-for-bit for full output, K8, KS, and first/last MU, IDX, EXCL checks. This is NOT real-model or video parity.
- **Performance (full 19-window 11-head call, with copies/requant/tail/output; capture excluded):** eager GPU `15.406544` ms vs graph `14.003184` ms (**1.1002x**); eager CPU+GPU sync wall `15.614200` ms vs graph `14.377000` ms (**1.0861x**). CPU submit speedup `7.4206x` is NOT total model speedup.
- Captures took **3.4674 ms**; computed amortization break-even is **3 full calls** on these synthetic static geometries. Capture reserved GPU allocator memory rose **164 MiB**, peak delta **164 MiB**, persistent staged K/V **105447936 bytes**, Q staging **3784704 bytes**. This does not establish model-resident 16 GiB headroom.
- **CPU unittest status: FAIL** was reported independently by PowerShell, before the GPU command. It is not a contradiction: these are separate paths. Verified the failure in `test_cache_refresh_calls_requant_and_updates_tail_on_cpu`: its `FakeQuant.run` tried to copy `[H,N,D]` and `[H,N]` mock results into the **flattened** `[H*N,D]` and `[H*N]` buffers passed by `update_kv_stage`. Reproduced the shape mismatch with local CPU Torch.
- **Fix**: reshape the fake INT8 and scale results back to `[H*N,D]` and `[H*N]` before `.copy_`, with explicit ABI shape assertions. Real CUDA quantizer and inference math are untouched. A focused standalone CPU reproduction of the corrected fake layout passed; the full repository unittest on the user's interpreter is **UNRUN after this fix**.

Decision: **PASS_NUMERIC / PASS_MUTABLE_GRAPH_PROBE**, performance advantage measured at **1.086x in sync wall for one synthetic 11-head group**; CUDA Graph persistent staging memory must be evaluated with actual H3 weights before any production promotion. CPU harness fix must be retested independently.

## Local CPU contract gate complete (2026-11-12)

The owner ran the full CPU contract suite on the exact code HEAD
`b01a79c2c824e253b4b19d52a72c342dd572740b`:

```text
Ran 8 tests in 2.645s
OK
git rev-parse HEAD
b01a79c2c824e253b4b19d52a72c342dd572740b
```

Status: **CPU CONTRACT PASS (8/8)** on the tested code HEAD. The previous
CPU `UNRUN` notation above describes the historical state immediately after
the fixture fix, not the current result. The mutable-input CUDA Graph
synthetic GPU probe is independently **PASS_NUMERIC / PASS_MUTABLE_GRAPH_PROBE**
on the same code. No new GPU benchmark or real-model test was performed for
this documentation update.

**Next gate: real model-resident H3 (56 distinct heads, model weights resident
on RTX 5070 Ti 16 GiB) and full-call timing / peak VRAM / parity; UNRUN.**


## Historical model-shadow gate disposition — 2026-11-12

**Do not run** `tools/run_h3_flat_model_shadow.py` as a standalone prerequisite. The owner closed that task as **SUPERSEDED / NOT REQUIRED**, not PASS. Its independent JSON has not been confirmed, but actual output-producing H3 Flat runs with 56 distinct model heads are already archived in `research/H3_FLAT_QPREP_LAUNCH_EVIDENCE_2026-11-12/` (see [review](https://github.com/ASarych/HTRC/blob/d996053cf48da20c9836dd4427ee0b306e3aa796/research/H3_FLAT_QPREP_LAUNCH_EVIDENCE_2026-11-12/ANALYSIS_2026-11-12.md)). Q=0.3 was visually rejected.

The only distinct pending model gate for this branch is **mutable-QKV CUDA Graph integration and A/B inside actual H3**. Synthetic Direct-Flow A/B has GPU numerical PASS and local CPU tests 8/8 PASS; neither establishes real H3 Direct-Flow speed/VRAM/video parity. Do not recreate a shadow-only experiment.
