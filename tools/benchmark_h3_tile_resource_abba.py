"""Research-only Tile32 resource/bit-exact micro A/B/B/A, one JSON.

Reuses the existing materialized Graph session, prepared state, Tile launcher
and exact kernel. Graph ABBA times *one Tile window*, not whole attention.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
PIN = (15275, 4, "balanced", 32, 512)
CANDIDATES = ("late_q", "stage1", "rb2", "rb2_late_q", "near_pv_acc")
ENV_KEYS = (
    "APL_HTRC_TILE32_STATS_BT", "APL_HTRC_DIRECTFLOW_GROUP",
    "APL_HTRC_DIRECTFLOW_WARMUPS", "APL_HTRC_DIRECTFLOW_VB_MODE",
    "APL_HTRC_DIRECTFLOW_GROUP_POLICY", "APL_HTRC_DIRECTFLOW_REPORT",
    "APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP",
    "APL_HTRC_GRAPH_CAPTURE_SAFE_MU", "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB",
    "APL_HTRC_DIRECTFLOW_MIN_FREE_MIB", "APL_HTRC_DIRECTFLOW_TILES32_ADMISSION",
)


def validate_request(candidate: str, tokens: int, heads: int,
                     rounds: int, warmups: int) -> None:
    if candidate not in CANDIDATES:
        raise ValueError("Only explicitly allowed one-window Tile variants are admitted")
    if (tokens, heads) != PIN[:2]:
        raise ValueError("Pinned 15275 tokens and 4 distinct heads only")
    if not (7 <= rounds <= 12 and 5 <= warmups <= 12):
        raise ValueError("ABBA requires 7..12 rounds and 5..12 warmups")


def compiled_resources(kernel) -> dict:
    meta = getattr(kernel, "metadata", None)
    regs = getattr(kernel, "n_regs", None)
    if regs is None:
        regs = getattr(meta, "n_regs", None)
    shared = getattr(meta, "shared", None)
    spills = getattr(kernel, "n_spills", None)
    if spills is None:
        spills = getattr(meta, "n_spills", None)
    if regs is None or shared is None:
        raise RuntimeError("Compiled Tile lacks register/shared metadata")
    return {"registers_per_thread": int(regs), "shared_bytes_per_cta": int(shared),
            "compiler_spills": int(spills) if spills is not None else None,
            "warps_per_cta": 4}


def resource_estimate(metrics: dict, regs_per_sm: int,
                      shared_per_sm: int, threads_per_sm: int) -> dict:
    regs = metrics["registers_per_thread"]
    shared = metrics["shared_bytes_per_cta"]
    if min(regs, shared, regs_per_sm, shared_per_sm, threads_per_sm) <= 0:
        raise ValueError("Nonpositive resource data")
    block_threads = 128
    limits = {"registers": regs_per_sm // (regs * block_threads),
              "shared": shared_per_sm // shared,
              "threads": threads_per_sm // block_threads}
    admitted = min(limits.values())
    return {"optimistic_resident_ctas_per_sm": admitted,
            "source_resource_limits": limits,
            "meets_both_3cta_necessary_bounds": (
                regs * block_threads * 3 <= regs_per_sm and
                shared * 3 <= shared_per_sm),
            "note": "Compiler/SRAM source upper bound, not achieved occupancy; allocations are granular"}


def abba_decision(control: list[float], candidate: list[float]) -> dict:
    if (len(control), len(candidate)) != (2, 2) or any(
        not math.isfinite(x) or x <= 0 for x in control + candidate
    ):
        raise ValueError("ABBA needs two positive independent block medians each")
    base = statistics.median(control)
    trial = statistics.median(candidate)
    consistent = max(candidate) < min(control)
    signal = bool(consistent and base / trial >= 1.05)
    return {"median_control_ms": base, "median_candidate_ms": trial,
            "control_over_candidate_ratio": base / trial,
            "candidate_faster_in_both_independent_captures": consistent,
            "at_least_5pct_faster_and_consistent": signal,
            "admission": "PROMISING_TILE_ONLY" if signal else "NO_GO_TILE_ONLY"}


def combined_resource_gate(compiled: dict, estimate: dict, *,
                           registers_per_sm: int, shared_bytes_per_sm: int) -> dict:
    """Necessary only: compiler resources permit >=3 CTAs; not a measured occupancy/speedup.

    No timing is admitted when either resource ceiling fails or compiler spill
    metadata is missing/nonzero. The installed GPU's actual SM limits are used.
    """
    regs = compiled["registers_per_thread"]
    shared = compiled["shared_bytes_per_cta"]
    spills = compiled.get("compiler_spills")
    if min(regs, shared, registers_per_sm, shared_bytes_per_sm) <= 0:
        raise ValueError("Invalid compiler or hardware resources")
    max_regs = registers_per_sm // (3 * 128)
    max_shared = shared_bytes_per_sm // 3
    both = (regs <= max_regs and shared <= max_shared
            and estimate["meets_both_3cta_necessary_bounds"]
            and estimate["optimistic_resident_ctas_per_sm"] >= 3)
    no_spills = (spills == 0)
    admitted = bool(both and no_spills)
    return {
        "status": "ADMIT_TILE_TIMING_ONLY" if admitted else "NO_GO_RESOURCE_GATE",
        "admitted_for_tile_timing": admitted,
        "necessary_3cta_resource_bounds_met": bool(both),
        "compiler_spills_zero": no_spills,
        "registers_per_thread": regs,
        "shared_bytes_per_cta": shared,
        "compiler_spills": spills,
        "required_registers_per_thread_at_most": max_regs,
        "required_shared_bytes_per_cta_at_most": max_shared,
        "registers_excess": max(0, regs - max_regs),
        "shared_bytes_excess": max(0, shared - max_shared),
        "optimistic_ctas_per_sm": estimate["optimistic_resident_ctas_per_sm"],
        "note": "Necessary allocation bounds only; not measured achieved residency, occupancy or speedup",
    }


def _alloc_partials(torch, item) -> dict:
    old = item["ws"]["_partials"]
    return {k: torch.empty_like(old[k]) for k in ("pm", "pl", "acc")}


def _verify_partials(torch, baseline: dict, candidate: dict, phase: str) -> None:
    for name in ("pm", "pl", "acc"):
        if not torch.equal(baseline[name], candidate[name]):
            raise AssertionError(f"{phase}: non-bit-exact partial {name}")


def _kern(torch, batch, states, cfg, tr, buffers, candidate: str):
    kwargs = ({"research_resource_variant": candidate}
              if candidate != "control" else {})
    _, compiled = batch.launch_prepared_tile_batch(
        states, cfg, tr, buffers=buffers, **kwargs)
    return compiled


def run_gpu(args) -> dict:
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_live_backend as live
    from h3 import htrc_flat_model_shadow as shared
    from h3 import htrc_flat_head_batch as batch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("Pinned NVIDIA SM120 only")
    if "RTX 5070 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Pinned RTX 5070 Ti only")
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.manual_seed(42026)
    shape = (1, 4, 15275, 128)
    banks = [tuple(torch.randn(shape, device=device, dtype=torch.bfloat16)
                   for _ in range(3)) for _ in range(2)]
    props = torch.cuda.get_device_properties(0)
    reg_sm = int(props.regs_per_multiprocessor)
    shared_sm = int(props.shared_memory_per_multiprocessor)
    thread_sm = int(props.max_threads_per_multi_processor)
    report = {"bank_partial_parity": [], "compiled": {}, "abba_blocks": []}
    previous = {name: os.environ.get(name) for name in ENV_KEYS}
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-tile-resource-") as tmp:
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
            cfg = shared.flat_config("balanced", 32)
            tr = live._triton()
            outputs = []
            ref_bank_b = None
            item_b = None
            for bi, bank in enumerate(banks):
                outcome = flow.graph_attention_bthd(
                    *bank, heads=4, mode="balanced", tiles=32,
                    budget_mib=1024, reserve_mib=256, workspace=True)
                torch.cuda.synchronize(device)
                if bi == 0:
                    gate = json.loads((Path(tmp) / "first_call.json").read_text(encoding="utf-8"))
                    if gate.get("status") != "PASS_FIRST_REAL_CALL" or gate.get("all_groups_bit_exact") is not True:
                        raise AssertionError("Existing first-call full eager/Graph parity gate failed")
                if outcome.shape != shape[:1] + (shape[2], shape[1], shape[3]):
                    raise AssertionError("Unexpected Graph attention output shape")
                outputs.append(outcome.detach().cpu().clone())
                session = flow._SESSIONS.get((4,))
                if session is None:
                    raise RuntimeError("Missing existing materialized four-head session")
                item = session["graphs"].get((32, 2048))
                if item is None:
                    raise RuntimeError("Missing full 32-tile attention window")
                states = item["states"]
                a = _alloc_partials(torch, item)
                b = _alloc_partials(torch, item)
                ka = _kern(torch, batch, states, cfg, tr, a, "control")
                kb = _kern(torch, batch, states, cfg, tr, b, args.candidate)
                torch.cuda.synchronize(device)
                _verify_partials(torch, a, b, f"bank{bi}")
                report["bank_partial_parity"].append({"bank": bi, "pm_pl_pacc_bit_exact": True})
                for name, kernel in (("control", ka), (args.candidate, kb)):
                    meta = compiled_resources(kernel)
                    if name in report["compiled"] and meta != report["compiled"][name]:
                        raise AssertionError("Compiled resources changed across banks")
                    report["compiled"][name] = meta
                if bi == 1:
                    item_b = item
                    ref_bank_b = {k: a[k].clone() for k in ("pm", "pl", "acc")}
                del a, b
            if torch.equal(outputs[0], outputs[1]):
                raise AssertionError("Two independent QKV banks returned identical full outputs")
            eager_b = live.flat_attention_bthd(
                *banks[1], heads=4, mode="balanced", tiles=32,
                budget_mib=1024, reserve_mib=256, workspace=True,
                query_staging="native_torch_mu", kv_staging="native_direct",
                dispatch="eager")
            if not torch.equal(eager_b.cpu(), outputs[1]):
                raise AssertionError("Existing bank B full eager/Graph parity failed")
            del eager_b, outputs
            torch.cuda.synchronize(device)
            # Compile/bit-exact resource admission happens BEFORE ANY
            # independently captured Tile timing of the combined candidate.
            # All two-bank PM/PL/PACC and full eager/Graph parity gates have
            # completed before this point, including the separate bank-B eager check.
            if args.candidate == "rb2_late_q":
                report["resource_estimates"] = {
                    name: resource_estimate(report["compiled"][name], reg_sm,
                                            shared_sm, thread_sm)
                    for name in ("control", args.candidate)
                }
                gate = combined_resource_gate(
                    report["compiled"][args.candidate],
                    report["resource_estimates"][args.candidate],
                    registers_per_sm=reg_sm, shared_bytes_per_sm=shared_sm)
                report["resource_gate"] = gate
                report["sm120_limits"] = {
                    "registers_per_sm": reg_sm,
                    "shared_bytes_per_sm": shared_sm,
                    "threads_per_sm": thread_sm,
                }
                report["tile_launch_geometry"] = {
                    "control_grid": [128, 4, 1],
                    "candidate_grid": [128, 4, 2],
                    "control_ctas": 512,
                    "candidate_ctas": 1024,
                    "candidate_rows_per_cta": 32,
                    "accounting": "One 2048-query-row window; CTA count is not achieved occupancy",
                }
                if not gate["admitted_for_tile_timing"]:
                    report["status"] = "NO_GO_RESOURCE_GATE"
                    report["completed"] = True  # Complete resource gate, NO benchmark.
                    report["tile_only_speed"] = {
                        "admission": "UNRUN_RESOURCE_GATE",
                        "samples_collected": 0,
                        "reason": "Candidate not admitted for Tile timing",
                    }
                    report["full_graph_first_call_eager_bit_exact"] = True
                    report["full_graph_bank_b_eager_bit_exact"] = True
                    report["distinct_qkv_banks"] = True
                    report["source_bank_used_for_abba"] = None
                    # abba_blocks remains []; never substitute profiler
                    # timing or invent GPU-event results in this branch.
                    return report
            states = item_b["states"]
            for block_idx, arm in enumerate(("control", args.candidate,
                                             args.candidate, "control"), 1):
                buffers = _alloc_partials(torch, item_b)
                _kern(torch, batch, states, cfg, tr, buffers, arm)
                torch.cuda.synchronize(device)
                _verify_partials(torch, ref_bank_b, buffers, f"block{block_idx}_pre_capture")
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    _kern(torch, batch, states, cfg, tr, buffers, arm)
                torch.cuda.synchronize(device)
                for _ in range(args.warmups):
                    graph.replay()
                torch.cuda.synchronize(device)
                _verify_partials(torch, ref_bank_b, buffers, f"block{block_idx}_post_warmup")
                ms = []
                for _ in range(args.rounds):
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    stop.record()
                    stop.synchronize()
                    ms.append(float(start.elapsed_time(stop)))
                _verify_partials(torch, ref_bank_b, buffers, f"block{block_idx}_post_timing")
                report["abba_blocks"].append({
                    "capture_index": block_idx, "arm": arm,
                    "gpu_event_samples_ms": ms,
                    "median_gpu_event_ms": statistics.median(ms),
                    "independent_capture": True,
                    "numeric": "PM_PL_PACC_BIT_EXACT",
                })
                del graph, buffers
                gc.collect()
            by_arm = {name: [x["median_gpu_event_ms"] for x in report["abba_blocks"]
                       if x["arm"] == name] for name in ("control", args.candidate)}
            report["tile_only_speed"] = abba_decision(by_arm["control"],by_arm[args.candidate])
            report["resource_estimates"] = {
                name: resource_estimate(report["compiled"][name], reg_sm,
                                        shared_sm, thread_sm)
                for name in ("control", args.candidate)
            }
            # RB2 deliberately doubles the CTA grid, not the memory footprint
            # of PM/PL/PACC: each CTA owns disjoint 32-query-row slices.
            splits = int(cfg["splits"])
            if splits != 4:
                raise AssertionError("Pinned research requires four leaf splits")
            rb = 2 if args.candidate in ("rb2", "rb2_late_q") else 1
            report["tile_launch_geometry"] = {
                "control_grid": [128, splits, 1],
                "candidate_grid": [128, splits, rb],
                "control_ctas": 128 * splits,
                "candidate_ctas": 128 * splits * rb,
                "candidate_rows_per_cta": 64 // rb,
                "accounting": "One 2048-query-row window, four heads; CTA count is not measured occupancy",
            }
            report["sm120_limits"] = {"registers_per_sm": reg_sm,
                                      "shared_bytes_per_sm": shared_sm,
                                      "threads_per_sm": thread_sm}
            report["full_graph_first_call_eager_bit_exact"] = True
            report["full_graph_bank_b_eager_bit_exact"] = True
            report["distinct_qkv_banks"] = True
            report["source_bank_used_for_abba"] = 1
            if args.candidate == "near_pv_acc":
                report["near_pv_acc_semantics"] = {
                    "control": "acc * alpha + tl.dot(p_bf16, v_bf16)",
                    "candidate": "tl.dot(p_bf16, v_bf16, acc=acc * alpha)",
                    "near_field_only": True,
                    "far_field_unchanged": True,
                    "same_RB1_CTA_grid": rb == 1,
                    "requires_bit_exact_partials": True,
                    "note": "FP32 accumulation order may differ, so exact GPU parity is mandatory",
                }
            report["status"] = "PASS_NUMERIC_TILE_ONLY"
            report["completed"] = True
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, default="near_pv_acc")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--expect-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "htrc-tile32-research-resource-abba-v1",
              "status": "FAIL_CLOSED", "completed": False,
              "candidate": args.candidate, "tokens": PIN[0], "heads": PIN[1],
              "mode": PIN[2], "tiles": PIN[3], "exact_leaves": PIN[4],
              "window_only_query_rows": 2048,
              "reference": "materialized Tile32 / Stats BT16 / RB1",
              "timing_scope": "ONE_TILE_KERNEL_WINDOW_NOT_FULL_ATTENTION",
              "ncu_used": False, "h3_model_or_video": "UNRUN",
              "production_promotion": "PROHIBITED_RESEARCH_ONLY",
              "not_a_full_operator_speedup": True}
    try:
        validate_request(args.candidate,PIN[0],PIN[1],args.rounds,args.warmups)
        actual = subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],
                                         text=True).strip()
        if actual != args.expect_head:
            raise RuntimeError(f"Exact HEAD mismatch {actual}")
        report["source_head"] = actual
        report.update(run_gpu(args))
    except BaseException as exc:
        report["status"] = "FAIL_CLOSED"
        report["completed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"[:1800]
        report["traceback_tail"] = traceback.format_exc()[-4500:]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"one_json={args.output} status={report['status']}")
    return 0 if report["status"] in ("PASS_NUMERIC_TILE_ONLY", "NO_GO_RESOURCE_GATE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
