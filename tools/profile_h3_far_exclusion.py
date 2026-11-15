"""Read-only two-bank Graph exclusion/near-selection shape probe; ONE JSON.

No candidate kernel, GPU speed benchmarking, H3 model/video or production change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from analyze_h3_far_exclusion import summarize_exclusions

ENV_KEYS = (
    "APL_HTRC_TILE32_STATS_BT", "APL_HTRC_DIRECTFLOW_GROUP",
    "APL_HTRC_DIRECTFLOW_WARMUPS", "APL_HTRC_DIRECTFLOW_VB_MODE",
    "APL_HTRC_DIRECTFLOW_GROUP_POLICY", "APL_HTRC_DIRECTFLOW_REPORT",
    "APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP",
    "APL_HTRC_GRAPH_CAPTURE_SAFE_MU", "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB",
    "APL_HTRC_DIRECTFLOW_MIN_FREE_MIB", "APL_HTRC_DIRECTFLOW_TILES32_ADMISSION",
)


def run_gpu() -> dict:
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("Exact NVIDIA SM120 CUDA GPU required")
    if "RTX 5070 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected RTX 5070 Ti GPU")
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.manual_seed(42026)
    shape = (1, 4, 15275, 128)
    banks = [tuple(torch.randn(shape, device=device, dtype=torch.bfloat16)
                   for _ in range(3)) for _ in range(2)]
    old = {key: os.environ.get(key) for key in ENV_KEYS}
    rows = []
    outputs = []
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-far-mask-") as tmp:
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
            for bi, bank in enumerate(banks):
                output = flow.graph_attention_bthd(
                    *bank, heads=4, mode="balanced", tiles=32,
                    budget_mib=1024, reserve_mib=256, workspace=True)
                torch.cuda.synchronize(device)
                if bi == 0:
                    gate = json.loads((Path(tmp)/"first_call.json").read_text(encoding="utf-8"))
                    if gate.get("status") != "PASS_FIRST_REAL_CALL" or gate.get("all_groups_bit_exact") is not True:
                        raise AssertionError("Full first-call eager/Graph parity failed")
                if tuple(output.shape) != (1, 15275, 4, 128):
                    raise AssertionError("Unexpected full attention output shape")
                outputs.append(output.detach().cpu().clone())
                stage = flow._SESSIONS.get((4,))
                if stage is None or (32, 2048) not in stage["graphs"]:
                    raise RuntimeError("Expected full materialized Tile32 Graph window missing")
                state = stage["graphs"][(32, 2048)]["states"]
                excl = state["EXCL"].detach().cpu()
                ids = state["IDX"].detach().cpu()
                if tuple(excl.shape) != (4, 32, 954) or tuple(ids.shape) != (4, 32, 512):
                    raise AssertionError("Graph EXCL/IDX shape changed from pinned admission")
                selection = summarize_exclusions(
                    excl.tolist(), ids.tolist(), splits=4, bn=64,
                    expected_near=512)
                if selection["current_far_dot_blocks"] != 128 * 15:
                    raise AssertionError("Unexpected far-loop block count")
                rows.append({"bank": bi, "selection": selection})
                del output, excl, ids
            if torch.equal(outputs[0], outputs[1]):
                raise AssertionError("Independent QKV banks produced same output")
            if rows[0]["selection"]["selection_mask_sha256"] == rows[1]["selection"]["selection_mask_sha256"]:
                raise AssertionError("Two independent QKV banks yielded identical selection masks")
            eager_b = live.flat_attention_bthd(
                *banks[1], heads=4, mode="balanced", tiles=32,
                budget_mib=1024, reserve_mib=256, workspace=True,
                query_staging="native_torch_mu", kv_staging="native_direct",
                dispatch="eager")
            if not torch.equal(eager_b.cpu(), outputs[1]):
                raise AssertionError("Bank B complete eager/Graph parity failed")
            del eager_b
            # No timing of any new candidate is performed here.
            return {"status": "PASS_FAR_EXCLUSION_GEOMETRY", "completed": True,
                    "gpu": torch.cuda.get_device_name(0),
                    "two_distinct_bf16_qkv_banks": True,
                    "different_selection_masks_per_bank": True,
                    "first_complete_graph_eager_bit_exact": True,
                    "bank_b_complete_graph_eager_bit_exact": True,
                    "banks": rows,
                    "all_mask_id_invariants_passed": True,
                    "no_new_kernel": True, "no_candidate_tile_kernel_timing": True}
    finally:
        try:
            flow.clear_session_cache()
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expect-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = {"schema": "htrc-far-exclusion-geometry-v1",
              "status": "FAIL_CLOSED", "completed": False,
              "probe": "READ_ONLY_FAR_EXCL_AND_NEAR_TOPK",
              "tokens": 15275, "heads": 4, "tiles": 32,
              "exact_leaves": 512, "far_bn": 64, "splits": 4,
              "source": "native materialized full-Graph prepared selection",
              "timing_scope": "NO_NEW_KERNEL_BENCHMARK",
              "ab_ba_speedup": "UNRUN", "h3_model_or_video": "UNRUN",
              "production_promotion": "PROHIBITED_RESEARCH_ONLY"}
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        if actual != args.expect_head:
            raise RuntimeError(f"Exact HEAD mismatch {actual}")
        report["source_head"] = actual
        report.update(run_gpu())
    except BaseException as exc:
        report["status"] = "FAIL_CLOSED"
        report["completed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"[:1800]
        report["traceback_tail"] = traceback.format_exc()[-4500:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"one_json={args.output} status={report['status']}")
    return 0 if report["status"] == "PASS_FAR_EXCLUSION_GEOMETRY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
