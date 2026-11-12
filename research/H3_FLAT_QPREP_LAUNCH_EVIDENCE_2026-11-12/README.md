# H3 Flat GPU launch evidence (2026-11-12)

This bundle preserves the two completed Priestess video-run reports and their runtime, event, GPU-memory, and host-memory logs. Local drive roots were replaced with labels such as `<COMFY_OUTPUT>` and `<USER_HOME>`; the logged measurements and model/run parameters remain. No credential-like patterns were found in the selected log files.

The ComfyUI `workflow.json` files are intentionally omitted because they contain the full prompt and reference-image path. Videos, latents, reference images, and model files are not included.

## Runs

- `flat_gpu_fast/`: stamp `20261112-002611-059`, HTRC Flat GPU fast, Q=0.0. Report compares against the earlier accepted Flat Q0.1 run.
- `flat_gpu_q03/`: stamp `20261112-004734-479`, same video setup with Q=0.3. Report compares against the Q=0.0 run.

Both are real ComfyUI runs, distinct from the synthetic shape-matched query-preparation microprofile. The Q0.3 run was rejected by the user for poor visual quality. Its recorded timing and memory values are retained as negative experimental evidence, not a recommended setting.
