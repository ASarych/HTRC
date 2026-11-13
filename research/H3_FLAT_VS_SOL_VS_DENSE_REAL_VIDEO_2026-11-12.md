# Real-video comparison: HTRC Flat vs Sol chunked tau 0 vs dense, and the Full HD 15 s memory wall (2026-11-12)

Branch `experiment/flat-vs-sol-dense-fullhd-20261113`, based on
`experiment/flat-live-host-sync-20261112@ecfc94d`. No code of HTRC is changed here; this is
a record of real ComfyUI runs on the owner's RTX 5070 Ti 16 GiB and what they show.
Logs: `D:\ComfyOutput\H3\logs\HyperFlow_20261112-17*` ... `-18*`.

## Setup common to all runs

MiniMax H3 HyperFlow Flash 14B NVFP4 (`ref2va`), prompt and reference image of the Priestess clip,
seed 730251, 6 Euler steps, CFG 1, causal video decoder, 1280x736 cropped to 1280x720, 56 frames,
18,723 tokens, 56 heads. Each run is a fresh backend (cold first step). One run per configuration:
**no repeats, so run-to-run noise on step times is not measured.**

The HTRC runs use this branch's base (`flat-live-host-sync`, the per-window sync fix) through the
ComfyNodes adapter `h3/attention_htrc_flat.py`. That adapter still passed `quality=` and failed
against any HTRC checkout that dropped the Q coefficient; a local, uncommitted shim in a scratch
clone accepted `quality=0` for the first run, and the adapter was then edited to pass
`query_staging` / `kv_staging` instead.

## Sampler step times, 18,723 tokens

Step 1 includes kernel compilation and reading blocks from disk and is the same order for every
backend. Steps 2-6 (seconds):

| Attention | Steps 2-6 each | Sum | Run total |
|---|---|---:|---:|
| Sol chunked, tau 0 | 5.73 / 4.78 / 5.87 / 5.28 / 5.29 | 26.95 | 165.4 s |
| Dense `comfy_kitchen_int8` | 5.73 / 5.81 / 7.53 / 7.20 / 7.39 | 33.66 | 177.0 s |
| HTRC Flat GPU, gather, 32 tiles, budget 1024 MiB | 7.89 / 7.31 / 13.16 / 8.09 / 7.51 | 43.96 | 157.8 s |
| HTRC Flat GPU, native staging, 32 tiles | 8.95 / 8.11 / 9.13 / 9.03 / 10.11 | 45.33 | 205.5 s |

Step 1: Sol 40.97, dense 48.41, HTRC gather 54.91, HTRC native 57.82. The 13.16 s step 4 of the
gather run coincides with an "early transient contraction" memory warning. Run totals are
dominated by load, first step and decode and are not an attention comparison.

Reading: at this size HTRC Flat fast (256 exact leaves) is about 31% slower than dense and 63%
slower than Sol tau 0. The synthetic whole-call figure for the gather path was 118 ms at 32 tiles
(see `H3_FLAT_LIVE_HOST_SYNC_WINDOW_2026-11-12.md`); native staging was 12% faster there and not
faster in the model. That synthetic gain did not transfer.

## What "Sol tau 0" is

It is sparse, not dense. In the reference implementation (comfy-kitchen `sol_attn.py`) a key block is
computed exactly when the mean of its query-block / key-block proxy scores exceeds
`tau x sigma`; the other blocks enter the softmax as one pooled term each, and the diagonal +-1
and sink blocks are always exact. At tau 0 roughly the above-average half of the key blocks per
query block is exact (derived from the rule; the exact share was not measured on this clip).
Flat fast computes 256 leaves of 16 = 4,096 of 18,723 keys (22%) exactly.

## Frames against dense (same seed, same everything else)

PSNR / SSIM of the full clip, dense as reference (ffmpeg `psnr`, `ssim`):

| Video | PSNR avg | SSIM |
|---|---:|---:|
| Sol chunked tau 0 | 22.83 dB | 0.704 |
| Sol control (11-10 run) | 21.67 | 0.654 |
| HTRC Flat GPU, gather | 21.85 | 0.638 |
| HTRC Flat GPU, native | 21.82 | 0.638 |
| old Flat Q=0.1 (11-11 run) | 20.37 | 0.608 |

Two HTRC runs against each other (gather vs native, and against the 11-12 morning fast run) give
about 32 dB / SSIM 0.92. So run-to-run differences of the same method are of that size, and
differences to dense of ~22 dB are larger than noise but **say nothing about which clip is better**;
over 6 diffusion steps a small attention difference moves the whole trajectory. The owner judged the
new HTRC clip visually equal to the previous fast one. No visual verdict against dense or Sol is recorded.

## Full HD 15 s does not fit on 16 GiB with any of these backends

1920x1088 cropped to 1080, 362 frames, 223,605 tokens. Q, K, V and the attention output are each
3.0 GiB (BF16, 56 heads x 128). All three attempts stopped before the first block finished.

| Backend | Result | Numbers from the log |
|---|---|---|
| HTRC Flat GPU (`htrc_flat_gpu`) | denied by whole-block preflight | needs 17.31 GiB, free 9.58 GiB |
| Sol chunked tau 0 | admitted, then `torch.OutOfMemoryError` allocating the 3 GiB output in `sol_attn_chunked` | reserve estimate 7.63 GiB, free 9.58 GiB |
| flat provider with INT8 K/V (`sol_rrc3_live`, `flat_live`), runner cap 2048 MiB | denied | needs 9.90 GiB, free 9.55 GiB |
| same, `APL_H3_FLAT_RUNNER_CAP_MIB=1024` | admitted at 9.25 GiB, then `MemoryError: flat HTRC needs ~216 MiB at the smallest chunk, 0 MiB free` | free memory at runner build was 0 |

Dense attention was not tried at this size.

### Causes

1. **The Sol estimate omitted its own workspace.** `sol_attn_plan` for 223,605 tokens, 56 heads is
   6.225 GiB (INT8 Q/K/V, routing table, summaries); the estimate in `tools/h3_memory_math.py`
   counted 5.39 GiB in total. At 18,723 tokens the omission is 0.43 GiB, which is why nothing
   failed on short clips. Fixed in ComfyNodes commit `37c0103e` (size taken from `sol_attn_plan`,
   fallback closed form 1.7-6.8% above it). With the fix Sol's attention peak at 223,605 tokens is
   11.61 GiB, so the run would now be refused before start instead of failing inside the block.
   No video was run with the fix.
2. **The HTRC GPU path needs full BF16 Q, K, V and a full output.** 4 x 3.0 GiB = 12 GiB before any
   workspace; this formula is correct, the memory is simply not there.
3. **The INT8-storage flat provider's estimate is too low.** It passed admission at 9.25 GiB and had
   0 MiB free when it built its runner after the QKV stage. Which term is missing was not
   established; a diagnostic run that would have printed the free/allocated figures at that point was
   stopped at the owner's request.

### Not established

Whether quantised storage of K/V (INT8 or lower) is enough to make HTRC fit at this size. The
numbers above only show that INT8 storage got the requirement from 17.31 to about 9.3-9.9 GiB, which
is not enough with the model's streamed blocks resident. Rotation-based 4-bit storage was discussed
but nothing was measured; its memory saving would be about 1.5 GiB of K/V at this size and it would
not be faster with the current INT8/BF16 kernels.

## Where the time goes outside attention (18,723 tokens, HTRC gather run, 158 s total)

Load to sampler 23 s; step 1 55 s (12.5 s of it the first attention call, the rest waiting for block
reads at ~160 MB/s from the model disk); steps 2-6 44 s; causal decode 31 s. These are identical for
every attention backend and were not optimised.

## Next gates (all UNRUN)

- Repeat each configuration at least twice on one backend lifetime (`-KeepBackend`) to size the noise
  and take the cold first step out.
- Find the missing term of the INT8 flat estimate with the free/allocated figures at runner build.
- A 5 s Full HD clip (about 82,000 tokens) on dense, Sol and HTRC, to see whether Flat's slower
  growth overtakes dense before memory runs out.
