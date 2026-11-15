"""GPU far-index pack cost only; exact two-bank content gate; one JSON.

NO packed attention candidate, NO speedup inference. Only the optional
single-kernel ordered index construction is timed after strict validation.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from analyze_h3_far_pack import reference_pack, verify_pack_outputs
from profile_h3_far_exclusion import ENV_KEYS

NB = 954
HEADS = 4
TILES = 32
SPLITS = 4
LPS = 256
FAR_BN = 64
TOKEN_COUNT = 15275


def validate_request(args):
    if not 7 <= args.rounds <= 12 or not 5 <= args.warmups <= 12:
        raise ValueError("Requires 7..12 timing samples and 5..12 warmups")
    if args.replays_per_sample != 16:
        raise ValueError("Pinned timing uses exactly 16 Graph replays per event")


def _alloc(torch, device):
    packed = torch.empty((HEADS, TILES, SPLITS, LPS),
                         dtype=torch.int32, device=device)
    counts = torch.empty((HEADS, TILES, SPLITS),
                         dtype=torch.int32, device=device)
    return packed, counts


def _launch(excl, packed, counts, pack_kernel):
    import torch
    if (tuple(excl.shape) != (HEADS, TILES, NB) or
        excl.dtype != torch.int8 or not excl.is_contiguous()):
        raise ValueError("Pinned contiguous int8 EXCL tensor required")
    pack_kernel[(HEADS * TILES, SPLITS)](
        excl, packed, counts, NB=NB, SPLITS=SPLITS,
        LPS=LPS, num_warps=4, num_stages=1)


def _check(torch, packed, counts, reference, phase):
    torch.cuda.synchronize(packed.device)
    try:
        return verify_pack_outputs(packed.detach().cpu().tolist(),
                                   counts.detach().cpu().tolist(), reference)
    except (ValueError, AssertionError) as exc:
        raise AssertionError(f"{phase}: {exc}") from exc


def run_gpu(args):
    import torch
    from htrc_far_pack_triton import pack_ordered_far_index
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("Pinned CUDA SM120 GPU is required")
    if "RTX 5070 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Pinned NVIDIA RTX 5070 Ti is required")
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.manual_seed(42026)
    shape = (1, HEADS, TOKEN_COUNT, 128)
    banks = [tuple(torch.randn(shape, device=device, dtype=torch.bfloat16)
                   for _ in range(3)) for _ in range(2)]
    previous = {key: os.environ.get(key) for key in ENV_KEYS}
    report = {"banks": [], "capture_blocks": []}
    full_outputs = []
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-far-pack-") as tmp:
            env = {
                "APL_HTRC_TILE32_STATS_BT": "16",
                "APL_HTRC_DIRECTFLOW_GROUP": "11",
                "APL_HTRC_DIRECTFLOW_WARMUPS": "4",
                "APL_HTRC_DIRECTFLOW_VB_MODE": "materialized",
                "APL_HTRC_DIRECTFLOW_GROUP_POLICY": "legacy",
                "APL_HTRC_DIRECTFLOW_REPORT": str(Path(tmp) / "first_call.json"),
                "APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP": None,
                "APL_HTRC_GRAPH_CAPTURE_SAFE_MU": "0",
                "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB": "768",
                "APL_HTRC_DIRECTFLOW_MIN_FREE_MIB": "256",
                "APL_HTRC_DIRECTFLOW_TILES32_ADMISSION": "0",
            }
            for key, value in env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            flow.clear_session_cache()
            second_bank_state = None
            second_bank_reference = None
            for bi, bank in enumerate(banks):
                out = flow.graph_attention_bthd(
                    *bank, heads=HEADS, mode="balanced", tiles=TILES,
                    budget_mib=1024, reserve_mib=256, workspace=True)
                torch.cuda.synchronize(device)
                if bi == 0:
                    gate = json.loads((Path(tmp) / "first_call.json").read_text(encoding="utf-8"))
                    if (gate.get("status") != "PASS_FIRST_REAL_CALL" or
                        gate.get("all_groups_bit_exact") is not True):
                        raise AssertionError("First complete eager/Graph gate failed")
                if tuple(out.shape) != (1, TOKEN_COUNT, HEADS, 128):
                    raise AssertionError("Full attention output shape drift")
                full_outputs.append(out.detach().cpu().clone())
                session = flow._SESSIONS.get((HEADS,))
                if session is None or (TILES, 2048) not in session["graphs"]:
                    raise AssertionError("Pinned full Tile32 Graph window missing")
                state = session["graphs"][(TILES, 2048)]["states"]
                excl, ids = state["EXCL"], state["IDX"]
                if (tuple(excl.shape) != (HEADS, TILES, NB) or
                    tuple(ids.shape) != (HEADS, TILES, 512) or
                    excl.dtype != torch.int8 or ids.dtype != torch.int32):
                    raise AssertionError("Pinned selection state shape/dtype drift")
                ref = reference_pack(excl.detach().cpu().tolist(),
                                     ids.detach().cpu().tolist(),
                                     splits=SPLITS, bn=FAR_BN, expected_near=512)
                if (ref["selection"]["current_far_dot_blocks"] != 1920 or
                    ref["packed_indices"] != 56576):
                    raise AssertionError("Unexpected far mask or packing geometry")
                packed, counts = _alloc(torch, device)
                _launch(excl, packed, counts, pack_ordered_far_index)
                audit = _check(torch, packed, counts, ref, f"bank{bi}_eager")
                report["banks"].append({
                    "bank": bi,
                    "selection_mask_sha256": ref["selection"]["selection_mask_sha256"],
                    "exact_pack_validation": audit,
                    "theoretical_packed_far_blocks": ref["packed_blocks"],
                    "current_far_blocks": ref["selection"]["current_far_dot_blocks"],
                    "fully_excluded_original_blocks": ref["selection"]["fully_excluded_blocks"],
                })
                if bi == 1:
                    second_bank_state = excl
                    second_bank_reference = ref
                del packed, counts, out
            if torch.equal(full_outputs[0], full_outputs[1]):
                raise AssertionError("Two independent QKV banks have identical full outputs")
            if (report["banks"][0]["selection_mask_sha256"] ==
                report["banks"][1]["selection_mask_sha256"]):
                raise AssertionError("Two independent banks have identical selection masks")
            eager = live.flat_attention_bthd(
                *banks[1], heads=HEADS, mode="balanced", tiles=TILES,
                budget_mib=1024, reserve_mib=256, workspace=True,
                query_staging="native_torch_mu", kv_staging="native_direct",
                dispatch="eager")
            if not torch.equal(eager.cpu(), full_outputs[1]):
                raise AssertionError("Independent bank-B full eager/Graph parity failed")
            del eager, full_outputs

            # Four independent Graph captures of the *same* ONE pack kernel.
            # 16 replays per GPU event reduce the noise of a tiny kernel.
            for block in range(1, 5):
                packed, counts = _alloc(torch, device)
                _launch(second_bank_state, packed, counts, pack_ordered_far_index)
                _check(torch, packed, counts, second_bank_reference, f"block{block}_pre_capture")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _launch(second_bank_state, packed, counts, pack_ordered_far_index)
                torch.cuda.synchronize(device)
                for _ in range(args.warmups):
                    graph.replay()
                _check(torch, packed, counts, second_bank_reference, f"block{block}_warmup")
                samples = []
                for _ in range(args.rounds):
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.replays_per_sample):
                        graph.replay()
                    stop.record()
                    stop.synchronize()
                    samples.append(float(start.elapsed_time(stop)) / args.replays_per_sample)
                post = _check(torch, packed, counts, second_bank_reference, f"block{block}_post_timing")
                report["capture_blocks"].append({
                    "capture_index": block, "independent_capture": True,
                    "replays_per_gpu_event": args.replays_per_sample,
                    "gpu_event_ms_per_pack_samples": samples,
                    "median_gpu_event_ms_per_pack": statistics.median(samples),
                    "output_prefix_validation": post,
                })
                del graph, packed, counts
                gc.collect()
            medians = [b["median_gpu_event_ms_per_pack"] for b in report["capture_blocks"]]
            reported_median = statistics.median(medians)
            report.update({
                "completed": True, "status": "PASS_PACK_COST_ONLY",
                "first_complete_graph_eager_bit_exact": True,
                "bank_b_complete_graph_eager_bit_exact": True,
                "two_distinct_bf16_qkv_banks": True,
                "different_selection_masks_per_bank": True,
                "pack_cost": {
                    "capture_medians_ms_per_pack": medians,
                    "median_of_capture_medians_ms_per_pack": reported_median,
                    "min_capture_median_ms": min(medians),
                    "max_capture_median_ms": max(medians),
                    "hardware_timing_scope": "ONE_PACK_KERNEL_PER_ONE_2048_ROW_WINDOW",
                    "samples_per_capture": args.rounds,
                    "replays_per_sample": args.replays_per_sample,
                    "does_not_include_packed_far_gathers_or_attention": True,
                },
                "memory": {
                    "packed_index_capacity_bytes": HEADS * TILES * SPLITS * LPS * 4,
                    "count_bytes": HEADS * TILES * SPLITS * 4,
                    "total_output_bytes": HEADS * TILES * SPLITS * (LPS * 4 + 4),
                    "packed_suffix_is_uninitialized_and_must_be_masked": True,
                },
                "no_packed_attention_kernel": True,
                "no_attention_candidate_timing": True,
                "production_modified": False,
            })
            return report
    finally:
        try:
            flow.clear_session_cache()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--replays-per-sample", type=int, default=16)
    args = parser.parse_args()
    report = {
        "schema": "htrc-far-pack-cost-v1", "status": "FAIL_CLOSED",
        "completed": False, "probe": "GPU_ORDERED_FAR_INDEX_PACK_COST_ONLY",
        "source_tokens": TOKEN_COUNT, "heads": HEADS, "tiles": TILES,
        "far_bn": FAR_BN, "splits": SPLITS, "near_budget": 512,
        "source": "real prepared full Graph EXCL/IDX on two independent QKV banks",
        "timing_scope": "ONE_PACK_KERNEL_NOT_FULL_ATTENTION",
        "attention_candidate": "UNRUN", "h3_model_or_video": "UNRUN",
        "production_promotion": "PROHIBITED_RESEARCH_ONLY",
    }
    try:
        validate_request(args)
        actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         text=True).strip()
        if actual != args.expect_head:
            raise RuntimeError(f"Expected exact HEAD {args.expect_head}; got {actual}")
        report["source_head"] = actual
        report.update(run_gpu(args))
    except BaseException as exc:
        report["status"] = "FAIL_CLOSED"
        report["completed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"[:1800]
        report["traceback_tail"] = traceback.format_exc()[-4600:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"one_json={args.output} status={report['status']}")
    return 0 if report["status"] == "PASS_PACK_COST_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
