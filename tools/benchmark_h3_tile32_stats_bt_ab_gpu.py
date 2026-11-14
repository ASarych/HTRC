#!/usr/bin/env python3
"""One-file CUDA A/B of the real Tile32 Graph stats-kernel BT16 and BT32.

No model, VAE, or video is loaded. Both branches use the same BF16 Q/K/V
banks and same HTRC balanced, 512-exact-leaf materialized-VB math.
Compare bitwise ALL output rows on TWO different banks, then measure
warmed full-attention GPU time and dissect a representative window's
leaf_stats, selection/topk, tile and merge GPU event stages. The latter
is eager kernel-stage attribution, NOT a captured-Graph kernel profile.
Outputs exactly ONE persistent JSON with all pass/failure evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3 import htrc_flat_directflow_live as flow
from h3 import htrc_flat_head_batch as batch
from h3 import htrc_flat_model_shadow as shared
from h3 import htrc_flat_live_backend as live
from h3 import htrc_flat_windows as windows


def _speed_decision(bt16, bt32, paired_abba):
    """Pure arithmetic: source gate executes this without importing torch."""
    if not bt16 or len(bt16) != len(bt32):
        raise ValueError("Both modes require the same nonzero block count")
    if paired_abba and len(bt16) != 2:
        raise ValueError("ABBA requires two independent captures per mode")
    if any(not math.isfinite(x) or x <= 0 for x in bt16 + bt32):
        raise ValueError("GPU event block medians must be finite and positive")
    ratio = statistics.median(bt16) / statistics.median(bt32)
    consistent = max(bt32) < min(bt16)
    candidate = bool(
        (ratio >= 1.05 and consistent)
        if paired_abba else (ratio > 1.0)
    )
    return {
        "gpu_speed_ratio_bt16_over_bt32": ratio,
        "candidate_faster_on_gpu": candidate,
        "bt32_strictly_faster_across_all_captures": (
            consistent if paired_abba else None
        ),
        "candidate_criterion": (
            "ABBA: >=5% advantage and both BT32 block medians below both BT16"
            if paired_abba else "legacy: single-capture median advantage"
        ),
    }


@torch.inference_mode()
def emit(stage, tokens, heads, rounds, paired_abba=False, warmup_replays=0):
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU required; no CPU approximation")
    if (tokens < 2064 or tokens > 40759 or heads != 8
        or rounds < 2 or rounds > 12):
        raise ValueError("Guard only 2064..40759 tokens, exactly eight heads, 2..12 rounds")
    if warmup_replays < 0 or warmup_replays > 20:
        raise ValueError("Post-capture warmup replays must be within 0..20")
    if paired_abba and (rounds < 5 or warmup_replays < 5):
        raise ValueError("ABBA requires >=5 timed rounds and >=5 post-capture warmups")
    order = (16, 32, 32, 16) if paired_abba else (16, 32)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(20261114)
    shape = (1, heads, tokens, 128)
    banks = [
        tuple(torch.randn(shape, device=device, dtype=torch.bfloat16)
              * 0.35 for _ in range(3))
        for _ in range(2)
    ]
    marker = torch.zeros((1, 1, 1, 1), device=device, dtype=torch.bfloat16)
    placeholder = marker.expand(shape)
    best = {}
    blocks = {"16": [], "32": []}
    stage["partial_arms"] = blocks
    outputs = {}
    run_flags = (
        "APL_HTRC_TILE32_STATS_BT",
        "APL_HTRC_GRAPH_CAPTURE_SAFE_MU",
        "APL_HTRC_DIRECTFLOW_TILES32_ADMISSION",
        "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB",
        "APL_HTRC_DIRECTFLOW_MIN_FREE_MIB",
        "APL_HTRC_DIRECTFLOW_REPORT",
    )
    old = {name: os.environ.get(name) for name in run_flags}
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-bt32-") as temp:
            os.environ["APL_HTRC_GRAPH_CAPTURE_SAFE_MU"] = "1"
            os.environ["APL_HTRC_DIRECTFLOW_TILES32_ADMISSION"] = "1"
            os.environ["APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB"] = "1152"
            os.environ["APL_HTRC_DIRECTFLOW_MIN_FREE_MIB"] = "768"
            os.environ["APL_HTRC_DIRECTFLOW_REPORT"] = str(
                Path(temp) / "private_first_call.json"
            )
            def qfill(bank_index, start, end, destination):
                source = banks[bank_index][0][0, :, start:end, :]
                real = end - start
                destination[:, :real, :].copy_(source)
                if real < int(destination.shape[1]):
                    destination[:, real:, :].copy_(
                        source[:, -1:, :].expand(
                            -1, destination.shape[1] - real, -1
                        )
                    )

            for block_index, bt in enumerate(order):
                stage["phase"] = f"BT{bt}_capture_block_{block_index+1}"
                flow.clear_session_cache()
                os.environ["APL_HTRC_TILE32_STATS_BT"] = str(bt)
                # Capture a per-arm memory delta. The second arm retains
                # its predecessor's output copies and allocator cache:
                # absolute peaks are NOT a perfectly isolated VRAM A/B.
                baseline_allocated = torch.cuda.memory_allocated(device)
                baseline_reserved = torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
                selected = {"index": 0}

                def fill_q(start, end, destination):
                    qfill(selected["index"], start, end, destination)

                def fill_kv(k, v):
                    source = banks[selected["index"]]
                    k.copy_(source[1])
                    v.copy_(source[2])

                def run(index):
                    selected["index"] = index
                    return flow.graph_attention_bthd(
                        placeholder, placeholder, placeholder,
                        heads=heads, mode="balanced", tiles=32,
                        budget_mib=1024, reserve_mib=256,
                        workspace=True, compare_only=True,
                        query_provider=fill_q, kv_provider=fill_kv,
                        research_source_kind="native_h3",
                        research_group_size=8,
                    )

                for bank_index in (0, 1):
                    result = run(bank_index)
                    torch.cuda.synchronize(device)
                    if (tuple(result.shape) != (1, tokens, heads, 128)
                        or result.dtype != torch.bfloat16
                        or not torch.isfinite(result).all().item()):
                        raise RuntimeError(
                            f"BT{bt}: invalid full BF16 Graph output"
                        )
                    key = (bt, bank_index)
                    if key in outputs:
                        if not torch.equal(outputs[key], result):
                            raise AssertionError(
                                f"BT{bt} bank{bank_index}: output changed after recapture"
                            )
                    else:
                        outputs[key] = result.detach().clone()
                    del result
                if torch.equal(outputs[(bt,0)], outputs[(bt,1)]):
                    raise AssertionError(
                        "Two independent mutable Q/K/V banks returned identical attention"
                    )

                stage["phase"] = f"BT{bt}_post_capture_warmups"
                for _ in range(warmup_replays):
                    warmed = run(1)
                    torch.cuda.synchronize(device)
                    if not torch.equal(warmed, outputs[(bt,1)]):
                        raise AssertionError(
                            f"BT{bt}: post-capture warmup changed output"
                        )
                    del warmed
                stage["phase"] = f"BT{bt}_timed_block_{block_index+1}"
                timings = []
                for _ in range(rounds):
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    warmed = run(1)
                    stop.record()
                    stop.synchronize()
                    if not torch.equal(warmed, outputs[(bt,1)]):
                        raise AssertionError(f"BT{bt}: warmed Graph changed output")
                    timings.append(float(start.elapsed_time(stop)))
                    del warmed

                # The paired timing probe must not allocate eager diagnostics
                # between fresh captures: those allocations changed Graph
                # allocator reservation in earlier HTRC experiments.
                stages_ms = None
                stage_output_parity = None
                if not paired_abba:
                    # Attribute the GPU kernels of a SAME-INPUT full query
                    # window. This is an EAGER invocation of the EXACT kernels
                    # used inside the captured Graph, not Graph-replay timing.
                    state = flow._SESSIONS.get((8,))
                    if state is None:
                        raise RuntimeError(f"BT{bt}: missing 8-head capture session")
                    geometry = next(
                        g for g in windows.iter_query_windows(tokens, 32)
                        if g.tile_count == 32
                    )
                    item = state["graphs"][(geometry.tile_count, geometry.valid_rows)]
                    fill_q(geometry.start, geometry.end, item["q"])
                    tr = live._triton()
                    cfg = shared.flat_config("balanced", 32)
                    event_intervals = []
                    states, _, _, _, _ = batch.prepare_batched_head_states(
                        item["win"], state["ids"], cfg, tr,
                        tile_count=geometry.tile_count,
                        quant_backend="triton", input_cast="gpu",
                        kv_cache=state["cache"], query_workspace=item["ws"],
                        valid_query_rows=geometry.valid_rows,
                        query_staging="native_direct",
                        vb_mode="materialized",
                        stage_gpu_events=event_intervals,
                    )
                    tile_begin = torch.cuda.Event(enable_timing=True)
                    tile_end = torch.cuda.Event(enable_timing=True)
                    merge_end = torch.cuda.Event(enable_timing=True)
                    tile_begin.record()
                    partial, _ = batch.launch_prepared_tile_batch(
                        states, cfg, tr, buffers=item["ws"]["_partials"]
                    )
                    tile_end.record()
                    merged, _ = batch.launch_prepared_merge_batch(
                        states, cfg, tr, partial, tails=state["tail"],
                        output=item["ws"]["_out"], merge_rows=32
                    )
                    merge_end.record()
                    merge_end.synchronize()
                    stages_ms = {
                        name: float(a.elapsed_time(b))
                        for name,a,b in event_intervals
                    }
                    stages_ms["tile_kernel"] = float(
                        tile_begin.elapsed_time(tile_end)
                    )
                    stages_ms["merge_kernel"] = float(
                        tile_end.elapsed_time(merge_end)
                    )
                    expected_stages = {
                        "query_gather_gpu_mu", "kv_cache_hit",
                        "leaf_stats", "rowmax_topk_exclusion",
                        "tile_kernel", "merge_kernel",
                    }
                    if expected_stages - stages_ms.keys():
                        raise RuntimeError(
                            f"BT{bt}: incomplete GPU kernel-stage event attribution: "
                            f"{sorted(expected_stages - stages_ms.keys())}"
                        )
                    selected_window = merged.reshape(
                        heads, geometry.padded_rows, 128
                    )[:, :geometry.valid_rows, :].permute(1,0,2).to(
                        torch.bfloat16
                    )
                    if not torch.equal(
                        selected_window,
                        outputs[(bt,1)][0,geometry.start:geometry.end]
                    ):
                        raise AssertionError(
                            f"BT{bt}: eager stage kernels differ from captured Graph output"
                        )
                    stage_output_parity = True
                record = {
                    "capture_block_index": block_index + 1,
                    "post_capture_warmup_replays": warmup_replays,
                    "gpu_event_ms_warmed_full_attention": timings,
                    "median_gpu_event_ms": statistics.median(timings),
                    "one_full_window_eager_kernel_stages_gpu_ms": stages_ms,
                    "one_full_window_bit_exact_to_captured_graph": stage_output_parity,
                    "eager_stage_attribution_status": (
                        "SKIPPED_ABBA_ISOLATION" if paired_abba else
                        "EXECUTED_AND_BIT_EXACT"
                    ),
                    "first_call_graph_report_stats_bt": json.loads(
                        (Path(temp) / "private_first_call.json").read_text(
                            encoding="utf-8"
                        )
                    ).get("stats_kernel_bt"),
                    "cuda_peak_allocated_mib": round(
                        torch.cuda.max_memory_allocated(device) / (1 << 20), 2
                    ),
                    "cuda_peak_reserved_mib": round(
                        torch.cuda.max_memory_reserved(device) / (1 << 20), 2
                    ),
                    "cuda_peak_allocated_delta_from_arm_baseline_mib": round(
                        (torch.cuda.max_memory_allocated(device)
                         - baseline_allocated) / (1 << 20), 2
                    ),
                    "cuda_peak_reserved_delta_from_arm_baseline_mib": round(
                        (torch.cuda.max_memory_reserved(device)
                         - baseline_reserved) / (1 << 20), 2
                    ),
                    "memory_scope": (
                        "per-arm reset_peak; prior bank outputs persist, "
                        "private Graph pool and allocator cache are not "
                        "independently isolated"
                    ),
                }
                if record["first_call_graph_report_stats_bt"] != bt:
                    raise RuntimeError("Graph report mislabeled stats BT")
                blocks[str(bt)].append(record)
                if not paired_abba:
                    del states, partial, merged, selected_window, item, state
                flow.clear_session_cache()
        for bt in (16, 32):
            per_block = blocks[str(bt)]
            expected = 2 if paired_abba else 1
            if len(per_block) != expected:
                raise AssertionError(f"BT{bt}: incomplete capture blocks")
            medians = [row["median_gpu_event_ms"] for row in per_block]
            best[str(bt)] = {
                **per_block[0],
                "blocks": per_block,
                "block_medians_gpu_event_ms": medians,
                "median_gpu_event_ms": statistics.median(medians),
                "gpu_event_ms_warmed_full_attention": [
                    x for row in per_block
                    for x in row["gpu_event_ms_warmed_full_attention"]
                ],
            }
        comparisons = {}
        for index in (0,1):
            a=outputs[(16,index)]
            b=outputs[(32,index)]
            equal=bool(torch.equal(a,b))
            diff=(a.float()-b.float())
            comparisons[f"bank_{index}"]={
                "bit_exact":equal,
                "max_abs":float(diff.abs().amax().item()),
                "relative_l2_percent":float(
                    diff.norm().item()/
                    max(a.float().norm().item(),1e-20)*100
                ),
            }
        stage["partial_parity"] = comparisons
        if not all(v["bit_exact"] for v in comparisons.values()):
            raise AssertionError(
                "Stats BT32 must be BIT EXACT to BT16 for BOTH full outputs"
            )
        decision = _speed_decision(
            best["16"]["block_medians_gpu_event_ms"],
            best["32"]["block_medians_gpu_event_ms"],
            paired_abba,
        )
        return {
            "status":"PASS_NUMERIC",
            "model_checkpoint_loaded":False,
            "video_or_vae_executed":False,
            "scope":(
                "CUDA Graph full attention isolated ABBA"
                if paired_abba else
                "CUDA Graph full attention + eager component GPU attribution"
            ),
            "tokens":tokens,"heads":heads,"head_dim":128,
            "mode":"balanced","tiles":32,"exact_leaves":512,
            "rounds":rounds,
            "paired_abba":paired_abba,
            "capture_order":list(order),
            "post_capture_warmup_replays_per_block":warmup_replays,
            "outputs":comparisons,
            "arms":best,
            "gpu_speed_ratio_bt16_over_bt32":decision[
                "gpu_speed_ratio_bt16_over_bt32"],
            "candidate_faster_on_gpu":decision["candidate_faster_on_gpu"],
            "speed_decision":decision,
            "cuda_device":torch.cuda.get_device_name(device),
            "model_resident_h3":False,
            "promotion_to_real_h3":"UNRUN_NOT_AUTHORIZED",
        }
    finally:
        # Release captured graphs before attempting to free GPU bank
        # buffers; no model weights or installed Comfy checkout touched.
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        flow.clear_session_cache()
        for name in run_flags:
            if old[name] is None:
                os.environ.pop(name,None)
            else:
                os.environ[name]=old[name]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens",type=int,default=40759)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--paired-abba",action="store_true",
                   help="4 independent captures: BT16, BT32, BT32, BT16")
    p.add_argument("--warmup-replays",type=int,default=0,
                   help="additional untimed full attention replays per capture")
    p.add_argument("--output",type=Path,default=Path(
        r"D:\ComfyOutput\H3\math_probe\flat_tile32_stats_bt_ab.json"
    ))
    args=p.parse_args()
    started=time.perf_counter()
    phase={"phase":"init"}
    report={
        "experiment":"h3_tile32_stats_bt16_vs_bt32_cuda_ab_v1",
        "status":"FAIL_CLOSED",
        "completed":False,
        "error":None,
    }
    try:
        result=emit(phase,args.tokens,args.heads,args.rounds,
                    paired_abba=args.paired_abba,
                    warmup_replays=args.warmup_replays)
        report.update(result)
        report["completed"]=True
    except BaseException as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        report["failed_phase"]=phase["phase"]
        report["partial_arms"]=phase.get("partial_arms",{})
        report["partial_parity"]=phase.get("partial_parity",{})
        report["traceback_tail"]=traceback.format_exc(limit=8)
        raise
    finally:
        report["elapsed_seconds"]=round(time.perf_counter()-started,2)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report,indent=2)+"\n",encoding="utf-8"
        )
        temporary.replace(args.output)
        print(json.dumps({
            "status":report["status"],
            "phase":phase["phase"],
            "report":str(args.output),
            "error":report["error"],
        }),flush=True)


if __name__ == "__main__":
    main()
