"""One-JSON, opt-in GPU A/B of HTRC Balanced Graph head-group policies.

Default mode writes an analytical PLAN, no CUDA. --execute runs full
Flat attention for legacy 11-head Graph and reuse-aware Graph on the
SAME distinct-head native BF16 Q/K/V input, short cases first.
Numerical mismatch is measured as % vs all-key FP32 dense reference at
sampled query positions; it NEVER cancels the experiment. Not a real
model/video or Sol timing. No promotion to production from this tool.

  python -B tools/benchmark_h3_graph_group_policy.py --execute --scope short
  python -B tools/benchmark_h3_graph_group_policy.py --execute --scope scaling
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tools"))
from h3 import htrc_flat_graph_group_plan as planner
from h3_flat_quality_reference import (
    exact_dense_sampled_queries, quality_against_exact, query_positions,
    normalized_difference_percent,
)

OUTPUT=Path(r"D:\ComfyOutput\H3\math_probe\flat_graph_group_balanced_ab.json")
SHAPES={"short":(15275,18723),
        "scaling":(15275,18723,40759,82000)}
POLICIES=("legacy","reuse_aware")


def _atomic_report(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",
                   encoding="utf-8")
    tmp.replace(path)


def _timing(call,rounds):
    import torch
    measured=[]
    for i in range(rounds):
        start=torch.cuda.Event(enable_timing=True)
        stop=torch.cuda.Event(enable_timing=True)
        start.record()
        before=time.perf_counter()
        out=call()
        stop.record()
        stop.synchronize()
        measured.append({
            "gpu_ms":float(start.elapsed_time(stop)),
            "synchronized_wall_ms":(time.perf_counter()-before)*1000,
        })
        del out
    return {
        "rounds":rounds,"samples":measured,
        "gpu_ms_median":statistics.median(x["gpu_ms"] for x in measured),
        "wall_ms_median":statistics.median(
            x["synchronized_wall_ms"] for x in measured),
        "scope":"warmed complete HTRC Flat attention, QKV refresh included",
    }


def run_case(row, args, write):
    import torch
    from h3 import htrc_flat_directflow_live as flow
    tokens=row["tokens"]
    torch.manual_seed(args.seed+tokens)
    # One unique, not duplicated/repeated, head bank reused by both arms.
    bank=tuple(torch.randn((1,args.heads,tokens,128),
                           device="cuda",dtype=torch.bfloat16)
               for _ in range(3))
    positions=query_positions(tokens,min(args.sample_queries,tokens))
    exact=exact_dense_sampled_queries(*bank,positions,query_batch=8)
    saved={}
    env_keys=(
        "APL_HTRC_DIRECTFLOW_GROUP_POLICY",
        "APL_HTRC_DIRECTFLOW_GROUP",
        "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB",
        "APL_HTRC_DIRECTFLOW_VB_MODE",
        "APL_HTRC_DIRECTFLOW_REPORT",
    )
    original={name:os.environ.get(name) for name in env_keys}
    for policy in POLICIES:
        result={"status":"UNRUN","policy":policy,
                "quality_reference":"all-key FP32 dense sampled queries",
                "real_video":"UNRUN"}
        row["policies"][policy]=result
        write()
        try:
            flow.clear_session_cache()
            torch.cuda.empty_cache()
            os.environ["APL_HTRC_DIRECTFLOW_GROUP_POLICY"]=policy
            os.environ["APL_HTRC_DIRECTFLOW_GROUP"]=str(
                11 if policy=="legacy" else args.max_group
            )
            os.environ["APL_HTRC_DIRECTFLOW_VB_MODE"]="materialized"
            os.environ["APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB"]=str(args.max_extra_mib)
            # The first-call parity diagnostic goes into a TEMP directory.
            # Only the aggregate matrix JSON persists after this arm.
            with tempfile.TemporaryDirectory(prefix="htrc-group-gate-") as temp:
                os.environ["APL_HTRC_DIRECTFLOW_REPORT"]=str(
                    Path(temp)/"first_call_gate.json"
                )
                torch.cuda.reset_peak_memory_stats()
                result["status"]="RUNNING"
                write()

                def call():
                    # Both the first call and every warmed replay must be
                    # inference-only. H3 Direct-Flow explicitly rejects
                    # autograd-enabled invocation. Keep the scope local:
                    # restore the caller's original grad setting on return.
                    with torch.inference_mode():
                        return flow.graph_attention_bthd(
                            *bank,heads=args.heads,mode=args.mode,tiles=args.tiles,
                            budget_mib=args.budget_mib,reserve_mib=args.reserve_mib,
                            workspace=True,compare_only=True,
                        )

                got=call()
                torch.cuda.synchronize()
                gate=json.loads(Path(os.environ[
                    "APL_HTRC_DIRECTFLOW_REPORT"]).read_text(encoding="utf-8"))
                if gate.get("status")!="RECORDED_FIRST_CALL":
                    raise RuntimeError("Missing research first-call Graph report")
                saved[policy]=got.detach().clone()
                result["quality"]=quality_against_exact(
                    saved[policy],exact,positions
                )
                result["planner"]=gate.get("group_plan")
                result["group_size"]=gate.get("head_group_cap")
                result["group_count"]=math.ceil(args.heads/int(result["group_size"]))
                result["distinct_session_sizes"]=gate.get("graph_cache_group_sizes")
                result["extra_reserved_mib"]=gate.get("extra_reserved_mib")
                result["eager_flat_diagnostic"]=gate.get("head_groups")
                result["timing"]=_timing(call,args.rounds)
                result["peak_allocated_mib"]=float(
                    torch.cuda.max_memory_allocated()/2**20)
                result["peak_reserved_mib"]=float(
                    torch.cuda.max_memory_reserved()/2**20)
                result["status"]=(
                    "QUALITY_AND_TIME_MEASURED" if result["quality"]["valid"]
                    else "NONFINITE_QUALITY_RECORDED"
                )
        except BaseException as exc:
            result["status"]="RUNTIME_FAILED"
            result["error"]=f"{type(exc).__name__}: {exc}"[:1600]
        finally:
            try:
                flow.clear_session_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except BaseException as cleanup:
                result["cleanup_error"]=str(cleanup)[:800]
            write()
    for key,value in original.items():
        if value is None:
            os.environ.pop(key,None)
        else:
            os.environ[key]=value
    if len(saved)==2:
        row["pairwise_difference_percent_to_dense_scale"]=(
            normalized_difference_percent(
                saved["legacy"],saved["reuse_aware"],exact,positions
            )
        )
        row["bit_exact_between_policies"]=bool(torch.equal(
            saved["legacy"],saved["reuse_aware"]
        ))
    else:
        row["pairwise_difference_percent_to_dense_scale"]=None
    row["sampled_queries"]=len(positions)
    row["sampled_query_positions"]=list(positions)
    row["status"]=(
        "MEASURED" if all(
            p["status"] in ("QUALITY_AND_TIME_MEASURED",
                            "NONFINITE_QUALITY_RECORDED")
            for p in row["policies"].values()
        ) else "PARTIAL_RUNTIME_FAILURE"
    )


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--scope",choices=tuple(SHAPES),default="short")
    ap.add_argument("--mode",choices=("balanced","fast"),default="balanced")
    ap.add_argument("--tiles",type=int,choices=(16,32,64),default=16)
    ap.add_argument("--heads",type=int,default=56)
    ap.add_argument("--max-group",type=int,default=32)
    ap.add_argument("--budget-mib",type=int,default=1024)
    ap.add_argument("--reserve-mib",type=int,default=256)
    ap.add_argument("--max-extra-mib",type=int,default=768)
    ap.add_argument("--free-mib-plan",type=float,default=5200.0)
    ap.add_argument("--sample-queries",type=int,default=64)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--seed",type=int,default=42026)
    ap.add_argument("--output",type=Path,default=OUTPUT)
    args=ap.parse_args()
    if (args.heads<1 or args.heads>56 or args.max_group<1 or
        args.max_group>32 or args.rounds<2 or args.sample_queries<1):
        ap.error("Invalid heads, max-group, rounds or sample-queries")
    # Apply Graph-specific caps exactly as the runtime does; this is
    # only a synthetic model-free planning comparison until --execute.
    rows=[]
    for tokens in SHAPES[args.scope]:
        plan=planner.choose_group(
            tokens=tokens,heads=args.heads,tiles=args.tiles,
            mode=args.mode,vb_mode="materialized",
            max_group=args.max_group,budget_mib=args.budget_mib,
            reserve_mib=args.reserve_mib,max_extra_mib=args.max_extra_mib,
            free_mib=args.free_mib_plan,min_free_mib=max(256,args.reserve_mib),
        )
        rows.append({
            "tokens":tokens,
            "band":"short" if tokens<20000 else "medium_long",
            "model_reuse_aware":plan,
            "policies":{arm:{"status":"UNRUN"} for arm in POLICIES},
            "status":"UNRUN",
        })
    report={
        "experiment":"h3_graph_balanced_group_memory_to_speed_ab_v1",
        "status":"PLANNED_ONLY",
        "gpu_status":"UNRUN",
        "mode":args.mode,
        "dispatch":"graph",
        "vb_mode":"materialized",
        "default_runtime_group_policy":"legacy",
        "baseline":"graph_materialized_legacy_group11",
        "candidate":"graph_materialized_reuse_aware_group",
        "reference":"FP32 exact dense all-key sampled queries",
        "no_bitwise_requirement_between_modes":True,
        "quality_is_percent_relative_l2_to_dense":True,
        "short_cases_are_mandatory":True,
        "scope":args.scope,"gpu_heads":args.heads,"window_tiles":args.tiles,
        "warnings":[
            "No stage K/V copies eliminated by this experiment: it only "
            "exploits already existing native KV aliases and better group sizing.",
            "Stage and graph-pool memory model is theoretical; actual "
            "post-capture graph allocator and free VRAM gates are authoritative.",
            "Fewer Graph replays does NOT guarantee faster full attention.",
            "No video, checkpoint, or Sol comparison.",
            "Numerical discrepancies are percent metrics and never abort the matrix."
        ],
        "rows":rows,
    }
    def persist(): _atomic_report(args.output,report)
    persist()
    if args.execute:
        import torch
        if not torch.cuda.is_available():
            report["status"]="CUDA_UNAVAILABLE"
            persist()
            raise RuntimeError("CUDA unavailable: analytical plan only")
        report["gpu"]=torch.cuda.get_device_name(0)
        report["gpu_status"]="RUNNING"
        for row in report["rows"]:
            try:
                run_case(row,args,persist)
            except BaseException as err:
                row["status"]="RUNTIME_FAILED"
                row["error"]=f"{type(err).__name__}: {err}"[:1200]
            persist()
        report["gpu_status"]="MEASURED_WITH_FAILURES" if any(
            x["status"]!="MEASURED" for x in report["rows"]
        ) else "MEASURED"
        report["status"]=report["gpu_status"]
    persist()
    print("one_report="+str(args.output)+" status="+report["status"])


if __name__=="__main__":
    main()
