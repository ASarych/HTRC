"""Research-only non-strict quality/speed A/B for HTRC Flat Graph variants.

Reference = exact FP32 scaled dot-product softmax across *all key tokens*
at evenly spaced sampled query rows, NOT eager/materialized Flat.
Compare percent error and timing separately. Numerical divergence NEVER
aborts the matrix. Runtime/CUDA errors are recorded per mode.

Not a full-query dense benchmark and not a real H3 checkpoint/video.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tools"))
from h3_flat_quality_reference import (
    query_positions, exact_dense_sampled_queries, quality_against_exact,
    normalized_difference_percent,
)


def _timed(call, rounds):
    import torch
    samples=[]
    for _ in range(rounds):
        begin=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        begin.record()
        started=time.perf_counter()
        result=call()
        end.record()
        end.synchronize()
        samples.append({
            "gpu_event_ms":float(begin.elapsed_time(end)),
            "synchronized_wall_ms":float((time.perf_counter()-started)*1000.0),
        })
        del result
    return {
        "gpu_event_ms_median":statistics.median(s["gpu_event_ms"] for s in samples),
        "synchronized_wall_ms_median":statistics.median(
            s["synchronized_wall_ms"] for s in samples),
        "samples":samples,
        "rounds":rounds,
        "status":"TIMED_NOT_QUALITY_GATE",
    }


def run_case(*, tokens, heads, mode, tiles, rounds, budget_mib,
             reserve_mib, variants, sample_queries=64,
             seed=42026, on_update=None):
    """Run baseline + optional variants over two fresh QKV banks.

    The first internal eager Flat comparison is requested as
    compare_only=True. A non-bit-exact result is RECORDED and continues.
    Graph capture/OOM can fail that *mode* but the runner tries the
    next mode after clearing graph caches. Numeric NaNs are reported
    as invalid quality, never treated as a successful score.
    """
    import torch
    from h3 import htrc_flat_directflow_live as flow
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available")
    if (type(tokens) is not int or tokens<16 or
        type(heads) is not int or not 1<=heads<=56 or
        mode not in ("fast","balanced") or
        tiles not in (16,32,64) or rounds<2 or
        sample_queries<1 or budget_mib<=reserve_mib+128 or
        reserve_mib<0):
        raise ValueError("Invalid geometry, mode or memory budget")
    if not variants or any(v not in ("split_stream","fused_vb") for v in variants):
        raise ValueError("Unexpected research variant")
    order=("materialized",)+tuple(variants)
    positions=query_positions(tokens,min(tokens,sample_queries))
    torch.manual_seed(seed)
    shape=(1,heads,tokens,128)
    banks=[
        tuple(torch.randn(shape,device="cuda",dtype=torch.bfloat16)
              for _ in range(3))
        for _ in range(2)
    ]
    # Exact dense references: all keys/values, sampled queries only.
    references=[
        exact_dense_sampled_queries(*bank,positions,query_batch=8)
        for bank in banks
    ]
    outcomes={}
    retained={}
    original=os.environ.copy()
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-quality-") as scratch:
            os.environ["APL_HTRC_DIRECTFLOW_REPORT"]=str(Path(scratch)/"gate.json")
            os.environ["APL_HTRC_DIRECTFLOW_GROUP"]="11"
            os.environ["APL_HTRC_DIRECTFLOW_WARMUPS"]="4"
            for arm in order:
                outcome={
                    "variant":arm,"status":"RUNNING",
                    "reference":"exact_dense_FP32_all_keys_sampled_queries",
                    "quality":[{"status":"UNRUN"},{"status":"UNRUN"}],
                    "timing":{"status":"UNRUN"},
                }
                outcomes[arm]=outcome
                if on_update is not None:
                    on_update(arm,outcome)
                try:
                    flow.clear_session_cache()
                    os.environ["APL_HTRC_DIRECTFLOW_VB_MODE"]=arm
                    torch.cuda.reset_peak_memory_stats()
                    def attention(bank):
                        return flow.graph_attention_bthd(
                            *bank,heads=heads,mode=mode,tiles=tiles,
                            budget_mib=budget_mib,reserve_mib=reserve_mib,
                            workspace=True,compare_only=True
                        )
                    saved=[]
                    for bank_index,bank in enumerate(banks):
                        output=attention(bank)
                        torch.cuda.synchronize()
                        saved.append(output.detach().clone())
                        outcome["quality"][bank_index]=quality_against_exact(
                            saved[-1],references[bank_index],positions
                        )
                        if bank_index==0:
                            gate_path=Path(os.environ["APL_HTRC_DIRECTFLOW_REPORT"])
                            gate=json.loads(gate_path.read_text(encoding="utf-8"))
                            if (gate.get("status")!="RECORDED_FIRST_CALL"
                                or gate.get("strict_parity_required") is not False):
                                raise RuntimeError(
                                    "Expected non-strict first-call diagnostic, "
                                    "not a silent strict or missing graph gate"
                                )
                            outcome["eager_flat_diagnostic"]={
                                "all_groups_bit_exact":gate.get("all_groups_bit_exact"),
                                "head_groups":[{
                                    "head_range":g.get("head_range"),
                                    "rel_l2_to_eager_percent":
                                        g.get("relative_l2_to_eager_flat_percent"),
                                    "max_abs_to_eager":g.get("max_abs_to_eager_flat"),
                                } for g in gate.get("head_groups",[])],
                                "not_the_quality_reference":True,
                            }
                            outcome["extra_graph_reserved_mib"]=gate.get(
                                "extra_reserved_mib")
                    if bool(torch.equal(saved[0],saved[1])):
                        outcome["different_qkv_banks_produced_bit_exact_same_output"]=True
                    else:
                        outcome["different_qkv_banks_produced_bit_exact_same_output"]=False
                    retained[arm]=saved
                    # Timed only after capture, eager diagnostic and dense
                    # reference computation. Includes QKV refresh and merge.
                    outcome["timing"]=_timed(lambda:attention(banks[1]),rounds)
                    outcome["gpu_peak_reserved_mib"]=float(
                        torch.cuda.max_memory_reserved()/2**20)
                    outcome["gpu_peak_allocated_mib"]=float(
                        torch.cuda.max_memory_allocated()/2**20)
                    outcome["status"]=(
                        "QUALITY_MEASURED"
                        if all(q.get("valid",False) for q in outcome["quality"])
                        else "NUMERIC_NONFINITE_RECORDED"
                    )
                    outcome["bit_exact_to_materialized_is_not_required"]=True
                except BaseException as exc:
                    outcome["status"]="RUNTIME_FAILED"
                    outcome["runtime_error"]=f"{type(exc).__name__}: {exc}"[:1600]
                    # Do not declare any numerical pass or fabricate a time.
                finally:
                    try:
                        flow.clear_session_cache()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except BaseException as cleanup_exc:
                        outcome["cleanup_error"]=(
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"[:1000]
                        )
                    if on_update is not None:
                        on_update(arm,outcome)
    finally:
        for key in (
            "APL_HTRC_DIRECTFLOW_REPORT",
            "APL_HTRC_DIRECTFLOW_GROUP",
            "APL_HTRC_DIRECTFLOW_WARMUPS",
            "APL_HTRC_DIRECTFLOW_VB_MODE",
        ):
            if key in original:
                os.environ[key]=original[key]
            else:
                os.environ.pop(key,None)
    # Pairwise disagreements are diagnostics; compare percentages to the
    # SAME dense reference and never suppress the candidate result.
    pairwise={}
    baseline=outcomes["materialized"]
    for arm in variants:
        candidate=outcomes.get(arm,{})
        if arm in retained and "materialized" in retained:
            pairwise[arm]={
                "bit_exact_to_materialized":[
                    bool(torch.equal(retained[arm][i],retained["materialized"][i]))
                    for i in (0,1)
                ],
                "normalized_difference_to_materialized_percent":[
                    normalized_difference_percent(
                        retained[arm][i],retained["materialized"][i],
                        references[i],positions
                    ) for i in (0,1)
                ],
                "interpretation":"diagnostic_only_not_quality_gate",
            }
        else:
            pairwise[arm]={"status":"UNAVAILABLE_DUE_RUNTIME_ERROR"}
        for i in (0,1):
            a=candidate.get("quality",[{},{}])[i]
            b=baseline.get("quality",[{},{}])[i]
            if a.get("valid") and b.get("valid"):
                a["error_delta_vs_materialized_percentage_points"]=round(
                    a["relative_l2_percent"]-b["relative_l2_percent"],6
                )
    rankings=[]
    for arm,result in outcomes.items():
        valid=[q["relative_l2_percent"] for q in result["quality"]
               if q.get("valid",False)]
        if len(valid)==2:
            rankings.append({"variant":arm,
                             "mean_relative_l2_percent":sum(valid)/2})
    rankings.sort(key=lambda x:x["mean_relative_l2_percent"])
    return {
        "status":("MEASURED" if all(
            r["status"] in ("QUALITY_MEASURED","NUMERIC_NONFINITE_RECORDED")
            for r in outcomes.values()) else "PARTIAL_RUNTIME_FAILURE"),
        "numerical_disagreement_aborts":False,
        "fatal_error_is_only_runtime_not_numerical":True,
        "tokens":tokens,"heads":heads,"mode":mode,"window_tiles":tiles,
        "sampled_query_positions":list(positions),
        "reference":{
            "name":"dense_exact_scaled_dot_product",
            "precision":"FP32_logits_softmax_and_V_reduction",
            "keys_per_query":tokens,
            "queries_sampled":len(positions),
            "sequence_queries_fully_evaluated":False,
            "noncausal":True,
            "score":"relative_L2_percent_to_exact_dense",
            "not_eager_flat":True,
        },
        "results":outcomes,"pairwise":pairwise,
        "quality_ranking":rankings,
        "speed_vs_sol":"NOT_MEASURED",
        "real_model_video":"UNRUN",
        "bank_count":2,"rng_seed":seed,
        "gpu":torch.cuda.get_device_name(0),
    }
