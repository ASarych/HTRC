# Original Tile steady-state admission for one far-VB .cg A/B/B/A

Date: 2026-11-15. Research-only branch
`experiment/flat-memory-to-speed-balanced-20261113`,
based on exact `7f9b295ac1fe64a01b1bd6b93f288fdcca5d5146`.

## Uploaded host/device attribution evidence

User-supplied `flat_tile_host_device_attribution.json`:
**89,657 bytes**, SHA-256
`b3481a6993bd12f8b58227c387ae4c2b16dc5863400017b1caaffd09976a341a`,
source exact HEAD
`7f9b295ac1fe64a01b1bd6b93f288fdcca5d5146`.
Full raw JSON stays with the user. The report has
`status=RECORDED_ORIGINAL_TILE_HOST_DEVICE_DIAGNOSTIC_ONLY`,
`completed=true`, 150 samples (60 + 60 with CUDA Events,
30 without), four BF16 heads, 15,275 tokens, original
Balanced/512 Tile32 2,048-query-row window.

Two independent QKV banks, pinned EXCL/IDX source SHA256,
native Graph/eager, unmodified Tile/PM/PL/PACC and Merge
raw-bit equality, all phase-end checks, and the original
255 registers/thread / 49,664 B/CTA / 0 compiler spills
PASS. There was **no candidate timing, no full attention
timing and no model/video inference**.

Medians of block medians:

| Phase | Wall ms/replay | Wall block spread | Event ms/replay | Event block spread | Enqueue ms/replay |
|---|---:|---:|---:|---:|---:|
| startup_host_gpu (12 x 5) | 0.4060451172 | 2.67982394% | 0.4048426300 | 2.67007903% | 0.0050109375 |
| settled_host_gpu (12 x 5) | 0.4079035156 | **0.23618414%** | 0.4066060036 | **0.24235802%** | 0.0050884766 |
| final_wall (6 x 5) | 0.4081986328 | **0.06911022%** | not measured | not measured | 0.0051166016 |

The first startup block median is 396.525 us wall and
395.353 us CUDA Event; block 12 is 407.151 / 405.910 us.
Both scales track the early upward transition. The first
raw sample is 383.689 us wall and 382.167 us GPU Event.
The last 12 event-instrumented block medians differ by
only 0.242% (max/min - 1); late no-event medians differ
by 0.069%. Early variation is thus not explained by
Python enqueue timing **alone**, whose median is ~5 us,
but CUDA Events include queued replay gaps and timing
overhead, not a purely isolated kernel cycle count.

Snapshots were 51.13 C / 44.58 W / 0% utilization before
and 72.64 C / 250.02 W / 99% after; these are *two*
snapshots, not concurrent telemetry in the startup phase.
**Do not claim temperature, clock throttling, scheduling
or a separate GPU client as the proven root cause.**
The nvidia-smi Comfy `standalone-env/python.exe` GPU
client was PID **26496**, identical to the report's
`process_identity.current_pid=26496`; the apparent
Comfy process was the benchmark itself. Other WDDM
clients were present, without proof that they issued
competing GPU kernels during sampling.

This supports pre-registered fixed settling BEFORE
small-effect A/B; it does **not** retroactively make
the two earlier unstable quiet-control runs pass,
nor prove a .cg improvement.

## Exactly one revised far-VB .cg speed experiment

The previously measured independent-capture .cg A/B/B/A
was `INCONCLUSIVE_TILE_ONLY` because the original
control captures differed by 32.327%. Existing candidate
was raw-bit exact, compiled 254 registers/CTA (control 255),
same 49,664 shared B/CTA and no spills. Its published
15.98% arithmetic apparent win was invalid due to unstable
control; worst cross-capture improvement was only 2.392%.
The separately measured mask-G 3.596% result is a
**closed NO_GO**, not a candidate for recombination.

This *new methodology*, not another random identical rerun,
uses the already implemented `far_vb_cg` cache policy
candidate. Kernel code, source selection, online softmax,
near/far loops and production defaults are unchanged.
Numerical and compiler-resource checks must pass first
for two distinct source QKV banks; no timing at all on
numeric/resource failure.

The new opt-in `--steady-precondition` flag performs:
1. Capture ONE original unchanged baseline Tile Graph after
   the accepted numeric/resource gates. Its output is
   checked against frozen bank-B PM/PL/PACC and original Merge.
2. Run exactly **24 blocks x 5 samples x 256 replays**
   of the baseline Graph using unprofiled host monotonic
   wall-clock and one CUDA synchronize per sample.
   All 120 raw samples are retained; no early exit, outlier
   removal, clock/power change, adaptive rewarm or candidate
   timing in this phase.
3. Use first 12 blocks as pre-defined settling evidence.
   Check **all last 12 block medians**: max/min spread
   <= **1.0%**, and median drift between first 6 and last
   6 of the late cohort <= **0.5%**. These are **new
   preconditioning thresholds**, NOT a change to the
   original 3% A/B inter-capture or 5% speed gates.
4. Revalidate baseline PM/PL/PACC and merged output *after*
   the entire preconditioning run, bit for bit.
5. If late controls do not meet BOTH thresholds,
   return `INCONCLUSIVE_PRECONDITION`, `completed=true`,
   zero A/B captures, and `UNRUN_UNSTABLE_CONTROL`.
   Never time the candidate in this case.
6. Only if the fixed gate admits, run the established
   four **independent** full Tile CUDA Graph captures in
   order control/.cg/.cg/control. Each has 5 warmups
   and 7 CUDA Event samples of 16 replays, plus strict
   pre/post capture/after-warmup raw-bit checks.
   Retain >=5% median *and worst-pair* speed improvement
   AND <=3% independent-capture median spread.
   No whole-attention or model/video claim.

A stable precondition is **necessary, not sufficient**
for speed admission. If captures still vary by >3%,
the decision stays `INCONCLUSIVE_TILE_ONLY`.
If .cg is faster by <5%, the final decision is
`NO_GO_TILE_ONLY`; do not iterate until a favorable
result. Only `PASS_TILE_ONLY_SPEED_GATE` makes this a
promising isolated **Tile** mechanism; independent
complete-attention, 56-head model-resident VRAM/quality
and video are still **UNRUN**.

## Files, execution, safety

- `tools/htrc_tile_steady_gate.py` -- pure fixed
  precondition protocol and fail-closed late-control gate.
- `tools/test_htrc_tile_steady_gate_cpu.py` -- five
  CPU tests, including fixed raw shape, corrupted data,
  unstable late-control rejection, deterministic
  fake CUDA Graph 30,720 replay count.
- `tools/benchmark_h3_far_vb_cg_tile_abba.py` --
  unchanged default behavior; new **opt-in** precondition
  before the old A/B/B/A capture loop.
- `tools/run_h3_far_vb_cg_steady_tile_abba_safe.ps1` --
  disposable exact-SHA clone, CPU/source checks, quiet
  GPU preflight, one JSON and explicit status validation.
- `tools/test_h3_far_vb_cg_steady_contract_cpu.py`
  -- integration/order/status and optional native
  PowerShell parser checks.

Windows runner requires exact pinned commit, a closed
ComfyUI server, Blender/Ollama closed and explicit
other-GPU-workloads-closed confirmation. Does not kill
processes or edit Comfy installations.

Run new runner only **once** after the complete source
commit has been pushed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_h3_far_vb_cg_steady_tile_abba_safe.ps1 -ExactHead <NEW_HEAD> -IClosedOtherGpuWorkloads -Rounds 7
```

Output exactly one user-held file:
`D:\ComfyOutput\H3\math_probe\flat_full_far_vb_cg_steady_tile_abba.json`.
The prior .cg and other diagnostics are not overwritten.

Local verification before publishing: the **exact Git
blob SHA** for helper and its five-unit-test module
was checked against the locally executed, source-identical
files; **5 CPU unit tests PASS**, Python compilation
PASS. Integrated exact-repo suite, native PowerShell parser,
RTX 5070 Ti GPU test **UNRUN** pending Windows execution.

Production, model/video and candidate speed admission:
**UNCHANGED / UNRUN**.

## 2026-11-16 — RTX 5070 Ti fixed steady-control run: inconclusive precondition

Source: user-provided \`flat_full_far_vb_cg_steady_tile_abba.json\`,
**33,927 bytes**, SHA-256
\`f46911955db21fb75bb3a1e61c6f86da83592c080c807216f8c21e6fc2f5b7ae\`,
tested at exact source HEAD
\`d871eba7d58f20d244f900aed04e067f4965a799\`.
Raw JSON is user-held and is not committed. Schema
\`htrc-far-vb-cg-tile-abba-v1\`,
\`status=INCONCLUSIVE_PRECONDITION\`, \`completed=true\`,
\`numeric_gate=PASS_TWO_BANK_FULL_TILE_MERGE_BIT_EXACT\`,
\`speed_measurement=UNRUN_UNSTABLE_CONTROL\`,
\`capture_blocks=[]\`, \`tile_only_speed=null\`.

The two different, pinned BF16 QKV source banks pass the
unchanged original Graph/eager, mirror and research
far-VB \`.cg\` raw-bit PM/PL/PACC/Merge checks.
All resource gates pass: control 255 registers/thread,
candidate 254, both 49,664 bytes shared/CTA, both zero spills.
The precondition control's post-run parity is also exact.
No candidate speed was sampled. These statements are
**numeric/resource evidence**, not performance evidence.

Exactly 24 original-only control blocks were recorded, each
with five wall-clock samples of 256 persistent CUDA Graph replays;
**120 raw samples, 30,720 replay calls**. All raw samples and
outliers remain present in the original user-held report.
The pre-registered decision uses the last 12 block medians
and both late half-medians:

| Metric | Observed | Required | Decision |
|---|---:|---:|---|
| First 12 block-median spread (descriptive) | 11.074111% | none | transitional |
| Last 12 block-median spread | **6.552325%** | <=1% | FAIL |
| Late blocks 13-18 median | 422.862695 us | - | first late half |
| Late blocks 19-24 median | 405.856445 us | - | second late half |
| Late half-median drift | **4.190213%** | <=0.5% | FAIL |

**Additional description, NOT a new admission decision:**
within blocks 19–24 alone, the block-median max/min
spread is **0.10594159%**, median **405.856445 us**.
Those last six were unusually stable, but discarding
blocks 13–18 would silently change the fixed experiment
after seeing the result. Do not do that, reinterpret the
whole run as passing, or issue an unregistered candidate
benchmark on the back of the final-six observation.
The first 12 control block medians ranged from about
404.935 to 449.778 us. The last 12 included swings
through block 17 before late stability appeared.

**Outcome: .cg speed remains UNRESOLVED, not NO_GO
on numerical parity and not PASS on speed.** The
previous \`.cg\` A/B/B/A was itself
\`INCONCLUSIVE_TILE_ONLY\` due to unstable control
capture medians (32.3267% spread); its apparent 15.98%
reduction was inadmissible. The present precondition
properly prevented a new misleading speed claim.

**Decision: STOP repeated .cg Tile microbenchmark retries.**
Do not relax the 1.0%/0.5% control precondition or the
3%/5% independent-capture speed gates; do not promote
\`.cg\` or combine it with the closed mask-G <5%
result. Keep the opt-in research code intact for
reproducibility. Revisit only if independently justified
environment control or materially different algorithm
makes a new experiment informative.

Return to the already identified whole-pipeline
bottlenecks: graph input Q/K/V staging, quantization,
top-k/state preparation, grouped CUDA Graph work,
and output Merge/write-back. See
\`H3_FLAT_MEMORY_TO_SPEED_BALANCED_2026-11-13.md\`.
The previously measured research-only 8-head grouping
was 4.34%/6.94% faster than legacy11 in the 15,275/
18,723-token *synthetic complete-attention* test,
but still lacks full-output and real-H3 model gates.
A future implementation should target a distinct
full-call mechanism with measured stage attribution,
not infer video speed from this Tile-only evidence.

Evidence-recording only: **no source, kernel, dispatcher,
production or benchmark change**; new GPU tests UNRUN.
