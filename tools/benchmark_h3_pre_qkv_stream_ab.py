"""One-JSON BF16 synthetic pre-QKV projection + Flat Graph A/B.

A: full fused [Q,K,V] projection, then Graph with 8-head groups.
B: K/V per-group staging and lazy per-window Q (previous control).
C: direct projection of K/V into stable Graph-owned stage; lazy Q.
D: direct Graph-owned K/V and one full Q projection per 8-head group.
All calls time projection + full Graph attention, not just replay.
No MiniMax checkpoint, real Q/K norm/RoPE, NVFP4, or video; no promotion.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = Path(r"D:\ComfyOutput\H3\math_probe\flat_pre_qkv_stream_ab.json")
ENV_KEYS = (
    "APL_HTRC_DIRECTFLOW_GROUP_POLICY",
    "APL_HTRC_DIRECTFLOW_GROUP",
    "APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP",
    "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB",
    "APL_HTRC_DIRECTFLOW_VB_MODE",
    "APL_HTRC_DIRECTFLOW_REPORT",
)


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def gpu_rounds(call, count):
    import torch
    samples = []
    for _ in range(count):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        started = time.perf_counter()
        out = call()
        b.record()
        b.synchronize()
        samples.append({
            "gpu_ms": float(a.elapsed_time(b)),
            "sync_wall_ms": (time.perf_counter()-started)*1000.0,
        })
        del out
    return {
        "rounds": count, "samples": samples,
        "gpu_ms_median": statistics.median(x["gpu_ms"] for x in samples),
        "sync_wall_ms_median": statistics.median(
            x["sync_wall_ms"] for x in samples),
        "includes": "QKV projection + full Graph Flat + final BTHD output",
    }


def compare_sample(control, candidate):
    import torch
    delta = candidate.float()-control.float()
    return {
        "sampled_queries": int(control.shape[1]),
        "bit_exact": bool(torch.equal(control, candidate)),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2_percent": float(
            delta.norm().item() /
            max(control.float().norm().item(), 1.0e-30) * 100.0),
    }


def run_case(tokens, args, row, write):
    import torch
    import torch.nn.functional as F
    from h3 import htrc_flat_directflow_live as flow
    from h3 import htrc_flat_pre_qkv_stream as pre

    torch.manual_seed(args.seed + tokens)
    x = torch.randn((tokens, args.hidden_width),
                    device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((3*args.heads*128, args.hidden_width),
                          device="cuda", dtype=torch.bfloat16)
    weights.mul_(1.0/math.sqrt(args.hidden_width))
    memory_source_mib = (x.numel()*x.element_size() +
                         weights.numel()*weights.element_size()) / 2**20
    row["resident_x_weight_mib"] = round(memory_source_mib, 3)
    row["full_qkv_theoretical_mib"] = round(
        tokens*3*args.heads*128*2/2**20, 3)
    row["max_projected_piece_mib"] = round(
        args.chunk_rows*2*args.group_heads*128*2/2**20, 3)
    positions = torch.linspace(0, tokens-1, steps=min(args.sample_queries, tokens),
                               dtype=torch.long).round()
    reference_sample = None

    def full_call():
        qkv = F.linear(x, weights).view(tokens, 3, args.heads, 128)
        q = qkv[:, 0].permute(1, 0, 2).unsqueeze(0)
        k = qkv[:, 1].permute(1, 0, 2).unsqueeze(0)
        v = qkv[:, 2].permute(1, 0, 2).unsqueeze(0)
        return flow.graph_attention_bthd(
            q, k, v, heads=args.heads, mode=args.mode,
            tiles=args.tiles, budget_mib=args.budget_mib,
            reserve_mib=args.reserve_mib, workspace=True,
            compare_only=True)

    def projected_call(kv_mode, q_mode):
        return pre.projected_graph_attention(
            x, weights, heads=args.heads, group_heads=args.group_heads,
            chunk_rows=args.chunk_rows, mode=args.mode, tiles=args.tiles,
            budget_mib=args.budget_mib, reserve_mib=args.reserve_mib,
            kv_mode=kv_mode, q_mode=q_mode)

    def streamed_call():
        return projected_call("separate", "window")

    def direct_kv_lazy_q_call():
        return projected_call("graph_owned", "window")

    def direct_kv_group_q_call():
        return projected_call("graph_owned", "group")

    prior = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        os.environ["APL_HTRC_DIRECTFLOW_GROUP_POLICY"] = "reuse_aware"
        os.environ["APL_HTRC_DIRECTFLOW_GROUP"] = str(args.group_heads)
        os.environ["APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP"] = str(
            args.group_heads)
        os.environ["APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB"] = str(
            args.max_extra_mib)
        os.environ["APL_HTRC_DIRECTFLOW_VB_MODE"] = "materialized"
        with torch.inference_mode():
            for arm, call in (("full_qkv", full_call),
                              ("streamed_kv_lazy_q", streamed_call),
                              ("direct_kv_lazy_q", direct_kv_lazy_q_call),
                              ("direct_kv_group_q", direct_kv_group_q_call)):
                result = {"status": "RUNNING", "model_video": "UNRUN"}
                row["arms"][arm] = result
                write()
                try:
                    flow.clear_session_cache()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    baseline_alloc = torch.cuda.memory_allocated()
                    baseline_reserved = torch.cuda.memory_reserved()
                    result["baseline_allocated_mib"] = baseline_alloc / 2**20
                    result["baseline_reserved_mib"] = baseline_reserved / 2**20
                    with tempfile.TemporaryDirectory(prefix="htrc-pre-qkv-") as t:
                        gate = Path(t) / "first_graph_gate.json"
                        os.environ["APL_HTRC_DIRECTFLOW_REPORT"] = str(gate)
                        torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                        output = call()
                        torch.cuda.synchronize()
                        result["cold_call_wall_ms"] = (
                            time.perf_counter()-started)*1000
                        result["cold_peak_allocated_extra_mib"] = (
                            torch.cuda.max_memory_allocated()-baseline_alloc) / 2**20
                        result["cold_peak_reserved_extra_mib"] = (
                            torch.cuda.max_memory_reserved()-baseline_reserved) / 2**20
                        result["cold_peak_allocated_absolute_mib"] = (
                            torch.cuda.max_memory_allocated()/2**20)
                        result["cold_peak_reserved_absolute_mib"] = (
                            torch.cuda.max_memory_reserved()/2**20)
                        if gate.exists():
                            detail = json.loads(gate.read_text(encoding="utf-8"))
                            result["first_graph_status"] = detail.get("status")
                            result["first_graph_extra_reserved_mib"] = detail.get(
                                "extra_reserved_mib")
                            result["memory_trace"] = detail.get("memory_trace")
                        sample = output[:, positions, :, :].float().cpu()
                        del output
                        if arm == "full_qkv":
                            reference_sample = sample
                        else:
                            result["parity_to_full_projection"] = compare_sample(
                                reference_sample, sample)
                        del sample
                        for _ in range(args.warmups):
                            warm = call()
                            del warm
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        result["timing"] = gpu_rounds(call, args.rounds)
                        result["warm_peak_allocated_mib"] = (
                            torch.cuda.max_memory_allocated()/2**20)
                        result["warm_peak_reserved_mib"] = (
                            torch.cuda.max_memory_reserved()/2**20)
                        result["warm_peak_allocated_extra_mib"] = (
                            torch.cuda.max_memory_allocated()-baseline_alloc)/2**20
                        result["warm_peak_reserved_extra_mib"] = (
                            torch.cuda.max_memory_reserved()-baseline_reserved)/2**20
                        result["status"] = "MEASURED"
                except BaseException as exc:
                    result["status"] = "RUNTIME_FAILED"
                    result["error"] = f"{type(exc).__name__}: {exc}"[:1600]
                finally:
                    write()
                    flow.clear_session_cache()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
    finally:
        for key, old in prior.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        del reference_sample, weights, x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--tokens", default="15275,18723")
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--group-heads", type=int, default=8)
    ap.add_argument("--hidden-width", type=int, default=1024)
    ap.add_argument("--chunk-rows", type=int, default=4096)
    ap.add_argument("--tiles", type=int, default=16, choices=(16, 32, 64))
    ap.add_argument("--mode", choices=("fast", "balanced"),
                    default="balanced")
    ap.add_argument("--budget-mib", type=int, default=1024)
    ap.add_argument("--reserve-mib", type=int, default=256)
    ap.add_argument("--max-extra-mib", type=int, default=768)
    ap.add_argument("--sample-queries", type=int, default=64)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    try:
        lengths = tuple(int(v) for v in args.tokens.split(","))
    except ValueError:
        ap.error("--tokens must be comma-separated positive integers")
    if (not lengths or any(n < 16 for n in lengths) or
        args.heads < 1 or args.group_heads < 1 or
        args.heads % args.group_heads or args.group_heads > 32 or
        args.hidden_width < 1 or args.chunk_rows < 1 or
        args.sample_queries < 1 or args.rounds < 2 or args.warmups < 0):
        ap.error("Invalid tokens/head groups/projection/benchmark arguments")

    report = {
        "experiment": "h3_pre_qkv_stream_graph_ab_v2",
        "status": "PLAN_ONLY",
        "scope": "synthetic BF16 fused projection + full Flat Graph",
        "same_input_across_arms": True,
        "weights": "BF16 dense [Q(all),K(all),V(all)]",
        "h3_rmsnorm_rope": "UNRUN_NOT_IMPLEMENTED",
        "nvfp4": "UNRUN_NOT_IMPLEMENTED",
        "checkpoint_model_denoise_video": "UNRUN",
        "hard_graph_extra_cap_mib": args.max_extra_mib,
        "heads": args.heads, "group_heads": args.group_heads,
        "hidden_width": args.hidden_width, "chunk_rows": args.chunk_rows,
        "warnings": [
            "Full QKV baseline is separate from streamed candidate; "
            "peak stats include allocator/Graph buffers and input source.",
            "streamed_kv_lazy_q keeps separate group K/V and copies into Graph.",
            "direct_kv_lazy_q/direct_kv_group_q project K/V directly into "
            "the stable Graph buffers; extra K/V source copies are avoided.",
            "Group-Q creates only one 8-head Q tensor and reuses it per window.",
            "All Graph-owned producer modes are research-only compare_only.",
            "Different GEMM shapes may change BF16 rounding/top-k; "
            "sample quality and not an assumed bit parity is reported.",
            "No H3 real-model quality, RoPE/norm or NVFP4 support claimed.",
        ],
        "cases": [{"tokens": n, "status": "UNRUN",
                   "arms": {"full_qkv": {"status": "UNRUN"},
                            "streamed_kv_lazy_q": {"status": "UNRUN"},
                            "direct_kv_lazy_q": {"status": "UNRUN"},
                            "direct_kv_group_q": {"status": "UNRUN"}}}
                  for n in lengths],
    }
    def write():
        save(args.output, report)
    write()
    if not args.execute:
        return
    import torch
    if not torch.cuda.is_available():
        report["status"] = "CUDA_UNAVAILABLE"
        write()
        raise RuntimeError("CUDA is needed to run the pre-QKV Graph A/B")
    for row in report["cases"]:
        row["status"] = "RUNNING"
        write()
        run_case(row["tokens"], args, row, write)
        row["status"] = (
            "MEASURED" if all(a["status"] == "MEASURED"
                              for a in row["arms"].values())
            else "MEASURED_WITH_FAILURES")
        write()
    report["status"] = (
        "MEASURED" if all(row["status"] == "MEASURED"
                          for row in report["cases"])
        else "MEASURED_WITH_FAILURES")
    write()


if __name__ == "__main__":
    main()
