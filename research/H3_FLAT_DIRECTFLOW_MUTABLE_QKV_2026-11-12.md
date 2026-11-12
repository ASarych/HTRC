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
