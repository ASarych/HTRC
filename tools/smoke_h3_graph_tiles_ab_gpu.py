#!/usr/bin/env python3
"""Real small CUDA Graph A/B: 16 vs 32 query tiles, identical Q/K/V.

This is a model-free guard, NOT a real HyperFlow speed benchmark.
It captures each mode, replays the same input twice, and rejects
nonidentical BF16 outputs. Tail geometry is exercised deliberately.
All CUDA Graph pools are released between the A/B variants.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3 import htrc_flat_directflow_live as flow


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Tiles32 Graph A/B requires NVIDIA CUDA")
    if os.environ.get("APL_HTRC_GRAPH_CAPTURE_SAFE_MU") != "1":
        raise RuntimeError("Tiles32 A/B requires capture-safe MU buffers")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    # 8192 + 127: balanced 512-leaf region AND a short, padded tail
    # window in both tile16 and tile32 graphs.
    heads, tokens, dim = 8, 8319, 128
    torch.manual_seed(424242)
    first_q = torch.randn((1, heads, tokens, dim), dtype=torch.bfloat16,
                          device=device) * 0.35
    first_k = torch.randn_like(first_q) * 0.35
    first_v = torch.randn_like(first_q) * 0.35
    dummy = torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16, device=device)
    fake = dummy.expand(1, heads, tokens, dim)
    outputs = {}
    timings = {}
    previous = os.environ.get("APL_HTRC_DIRECTFLOW_TILES32_ADMISSION")
    try:
        with torch.inference_mode():
            def fill_kv(kstage, vstage):
                kstage.copy_(first_k)
                vstage.copy_(first_v)

            def fill_q(start, end, stage):
                real = end - start
                src = first_q[0, :, start:end, :]
                stage[:, :real, :].copy_(src)
                if real < stage.shape[1]:
                    stage[:, real:, :].copy_(
                        src[:, -1:, :].expand(
                            -1, stage.shape[1] - real, -1
                        )
                    )

            for tiles in (16, 32):
                # The special pre-capture allocation guard is only valid
                # for the 32-tile branch; keep 16-tile stock control intact.
                os.environ["APL_HTRC_DIRECTFLOW_TILES32_ADMISSION"] = (
                    "1" if tiles == 32 else "0"
                )
                flow.clear_session_cache()
                wall_start = time.perf_counter()
                current = None
                for replay in range(2):
                    produced = flow.graph_attention_bthd(
                        fake, fake, fake,
                        heads=heads, mode="balanced", tiles=tiles,
                        budget_mib=1024, reserve_mib=256,
                        workspace=True, compare_only=True,
                        query_provider=fill_q, kv_provider=fill_kv,
                        research_source_kind="native_h3",
                        research_group_size=heads,
                    )
                    torch.cuda.synchronize(device)
                    if (produced.device != device
                        or produced.dtype != torch.bfloat16
                        or tuple(produced.shape) !=
                           (1, tokens, heads, dim)
                        or not torch.isfinite(produced).all().item()):
                        raise RuntimeError(
                            f"Tile{tiles} CUDA Graph output invalid"
                        )
                    if current is not None and not torch.equal(
                        current, produced
                    ):
                        raise AssertionError(
                            f"Tile{tiles} two cached Graph replays differ"
                        )
                    current = produced.detach().clone()
                    del produced
                outputs[tiles] = current
                timings[str(tiles)] = {
                    "wall_two_calls_ms": round(
                        (time.perf_counter() - wall_start) * 1000, 2
                    ),
                    "window_count": (tokens + tiles * 64 - 1) // (tiles * 64),
                    "capture_and_replay_test": "TWO_IDENTICAL_CALLS_PASS",
                }
                flow.clear_session_cache()

            a = outputs[16]
            b = outputs[32]
            bit_exact = torch.equal(a, b)
            diff = (a.float() - b.float())
            max_abs = float(diff.abs().amax().item())
            relative_l2_percent = float(
                diff.norm().item() /
                max(a.float().norm().item(), 1.0e-20) * 100
            )
            report = {
                "experiment": "h3_graph_tiles16_vs32_gpu_guard_v1",
                "status": "PASS" if bit_exact else "FAIL_NONIDENTICAL_OUTPUT",
                "weight_checkpoint_loaded": False,
                "mode": "balanced", "heads": heads, "tokens": tokens,
                "reference_tiles": 16, "candidate_tiles": 32,
                "output_bit_exact": bool(bit_exact),
                "max_abs": max_abs, "relative_l2_percent": relative_l2_percent,
                "arms": timings,
                "cuda_allocated_peak_mib": round(
                    torch.cuda.max_memory_allocated(device) / (1 << 20), 2
                ),
                "cuda_reserved_peak_mib": round(
                    torch.cuda.max_memory_reserved(device) / (1 << 20), 2
                ),
                "real_h3_denoise_or_video_tested": False,
            }
            print(json.dumps(report, indent=2), flush=True)
            if not bit_exact:
                raise AssertionError(
                    "Tiles32 GPU Graph must be BIT-EXACT to tiles16 on "
                    "the same BF16 Q/K/V, including tail padding"
                )
    finally:
        if previous is None:
            os.environ.pop("APL_HTRC_DIRECTFLOW_TILES32_ADMISSION", None)
        else:
            os.environ["APL_HTRC_DIRECTFLOW_TILES32_ADMISSION"] = previous
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        flow.clear_session_cache()


if __name__ == "__main__":
    main()
