# H3 Flat: model-resident shadow gate (2026-11-11)


## CLOSED / SUPERSEDED — do NOT rerun this standalone shadow gate (owner decision 2026-11-12)

**Decision:** The standalone `tools/run_h3_flat_model_shadow.py` gate is **CANCELLED AS UNNECESSARY**, not `PASS`. Do not schedule, execute, or request its `flat_model_resident_shadow.json` solely to establish that Flat works on real MiniMax H3 weights / 56 distinct model heads. Its separate run is **not verified**; keep the historical instructions below for provenance only.

**Stronger evidence already archived:** two completed, output-producing real HyperFlow H3 NVFP4 video runs with `attention_profile=htrc_flat_gpu`, 56 distinct native heads, 56 frames / 6 denoising steps / RTX 5070 Ti, in
[real-run evidence review](https://github.com/ASarych/HTRC/blob/d996053cf48da20c9836dd4427ee0b306e3aa796/research/H3_FLAT_QPREP_LAUNCH_EVIDENCE_2026-11-12/ANALYSIS_2026-11-12.md). The Fast Q=0 run completed; Q=0.3 also completed but its visual quality was rejected. These runs validate actual Flat output and model execution, **not** numerical equivalence to dense attention or user-approved final image quality.

**Still UNRUN:** integrating the new mutable-QKV Direct-Flow CUDA Graph into real H3, measuring its full denoising latency, memory headroom, and output/temporal quality. The old shadow test is not a prerequisite for this separate gate. No quality coefficients or reduction of fixed Flat leaves/centres are authorized.

---


Experiment only, never a production backend. Source branch: experiment/flat-model-resident-shadow-20261111.
Derived from the verified query-workspace gate: real 4-head 1.02571x and 56-logical proxy 1.03404x, all 520 group/window outputs exact.
The next missing proof is actual 56 distinct H3 heads with model weights resident.

## Proven source integration boundary

The canonical ComfyNodes real-weight attention benchmark is tools/run_h3_attention_backend_gate.py
from ASarych/ComfyNodes research/h3-flat-live@571a7a3e6cbc7c76112ca21ea5d4f2a21a3d649f.
It loads one MiniMax H3 checkpoint block via APLH3HybridLoader and intercepts
comfy.ldm.minimax.model.optimized_attention AFTER qkv_proj, RMSNorm and RoPE.

The new HTRC tools/run_h3_flat_model_shadow.py reuses that model loader.
The new h3/htrc_flat_model_shadow.py reads native BHSD [1,H,S,128] QKV as
zero-copy [S,H,128] permuted views and runs the validated Triton Flat K8
head batching on the first four, then ALL 56 distinct model heads.
KBF, VBF, INT8 K, FP32 KS are cached only within one attention call.
G/VB/LZ/SG, topk, exclusion and tile/merge are recalculated per query window.
Last partial Q tile uses only valid real Q rows for its MU, and 0..15 final K rows are exact.

The original Comfy attention output is computed first. It remains the ONLY
output sent into H3 out_proj: shadow cannot replace it. Status, native
quality rel-L2/max-abs/mean-abs, per-case elapsed times, memory, cache
build and group sizes go into ONE model-resident JSON, even on failure.
The global MiniMax attention function and in_training mode are restored.
CUDA OOM or unsupported shapes fail closed; never silently change production.

IMPORTANT: The first gate uses genuine model weights and genuine qkv_proj
with random/synthetic block input, synthetic timestep, and identity RoPE.
It is NOT a real denoising step, generated frame, or video quality test.
Approximate Flat need not match the original native backend exactly.

## One Windows run, one JSON

    cd D:\ComfyNodes-flat-split-20261111
    git fetch htrc
    git switch --detach htrc/experiment/flat-model-resident-shadow-20261111
    $py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
    & $py -m py_compile h3\htrc_flat_head_batch.py h3\htrc_flat_model_shadow.py tools\run_h3_flat_model_shadow.py tools\test_h3_flat_model_shadow_contract.py
    if ($LASTEXITCODE -ne 0) { throw "Compile FAILED" }
    & $py -m unittest discover -s tools -p test_h3_flat_model_shadow_contract.py
    if ($LASTEXITCODE -ne 0) { throw "Contract tests FAILED" }
    & $py -B tools\run_h3_flat_model_shadow.py --block 0 --sequence 37710 --head-cases 4,56 --mode fast --tiles 16 --budget-mib 1024 --reserve-mib 256 --workspace on

One retained report: D:\ComfyOutput\H3\math_probe\flat_model_resident_shadow.json.
No separate 4/56 JSONs, no QKV .pt dump, live group-by-group progress.
The tool automatically looks under the actual ComfyUI custom_nodes directory
for ComfyNodes/h3/hybrid_loader.py, or uses APL_H3_COMFYNODES_ROOT.
If missing, pass --comfynodes-root with the true installed ComfyNodes checkout.
The HTRC-only D:\ComfyNodes-flat-split-20261111 worktree is NOT the ComfyNodes loader.
Default --unet path is the previously proven premerged NVFP4 H3 checkpoint;
it must exist locally. --preflight-only verifies paths without loading model.

## After this gate

Only after a clean 56-real-head model-resident run, add a separate explicit
opt-in shadow hook to the APL-owned attention_runtime.py for a TRUE denoising
call (real prompt and RoPE), then follow the canonical 39-frame -> 124-frame
-> 243-frame validation ladder. Measure quality, motion and VRAM, not just
Flat vs Flat bit parity. No CUDA Graph or production BEST promotion before that.

Validation of this new model-resident GPU gate is UNRUN.