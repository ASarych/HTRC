"""One-file GPU A/B: HTRC DirectFlow materialized VB vs fused in-Tile VB.

Same two independent native BF16 Q/K/V banks. Materialized Graph and
fused-VB Graph each run the existing 56-head group exact-native-eager
first-call gate; this tool then checks both banks' full attention outputs
bit-for-bit and measures warmed Graph replay inclusive of K8/QKV refresh.
Synthetic inputs only; NOT a real checkpoint, video or Sol comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from h3 import htrc_flat_directflow_live as flow
from h3 import htrc_flat_live_backend as live


def _timed(call, rounds):
    start=torch.cuda.Event(enable_timing=True)
    stop=torch.cuda.Event(enable_timing=True)
    start.record()
    start_wall=time.perf_counter()
    out=None
    for _ in range(rounds):
        out=call()
    stop.record()
    stop.synchronize()
    wall=(time.perf_counter()-start_wall)*1000/rounds
    return out, {"gpu_event_ms_per_call":float(start.elapsed_time(stop))/rounds,
                 "sync_wall_ms_per_call":wall}


def _difference_summary(reference, candidate, sample_limit=8):
    """Exact BF16 output comparison; statistics are NOT an acceptance tolerance."""
    if (reference.shape != candidate.shape or
        reference.dtype != candidate.dtype or
        reference.ndim != 4 or reference.shape[0] != 1 or
        reference.shape[-1] != 128):
        raise ValueError("Diagnostic expects identical [1,S,H,128] BF16 outputs")
    s, h, d = map(int, reference.shape[1:])
    unequal = reference != candidate
    delta = candidate.float() - reference.float()
    abs_delta = delta.abs()
    count = int(unequal.sum().item())
    total = int(unequal.numel())
    magnitude = abs_delta.flatten()
    k = min(sample_limit, total)
    values, indices = torch.topk(magnitude, k=k, largest=True, sorted=True)
    rf = reference.flatten()
    cf = candidate.flatten()
    samples = []
    for magnitude_f, index_t in zip(values.tolist(), indices.tolist()):
        if magnitude_f == 0:
            break
        i = int(index_t)
        feature = i % d
        head = (i // d) % h
        token = (i // (d*h)) % s
        samples.append({
            "token":token, "head":head, "feature":feature,
            "reference":float(rf[i].item()), "candidate":float(cf[i].item()),
            "abs_difference":float(magnitude_f),
        })
    per_head = []
    for head in range(h):
        u = unequal[0,:,head,:]
        dv = delta[0,:,head,:]
        per_head.append({
            "head":head,
            "mismatch_count":int(u.sum().item()),
            "max_abs":float(dv.abs().max().item()),
        })
    return {
        "bit_exact":count == 0,
        "total_elements":total, "mismatch_count":count,
        "mismatch_percent":100.0 * count / total,
        "max_abs":float(abs_delta.max().item()),
        "relative_l2_percent":float(
            100.0 * delta.norm().item() /
            max(reference.float().norm().item(), 1.0e-30)
        ),
        "per_head":per_head,
        "largest_differences":samples,
        "numerical_tolerance_applied":False,
    }


@torch.inference_mode()
def diagnose_numeric(tokens, heads, mode, tiles, budget_mib, reserve_mib):
    """Diagnostic only: bypass the eager first-call gate WITHOUT changing it.

    Compare-only Graph results are explicitly checked afterward against
    the untouched native eager reference. A mismatch is NEVER PASS_NUMERIC.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for fused-VB numerical diagnostics")
    if (tokens != 15275 or heads != 4 or mode != "balanced" or tiles != 32
        or budget_mib != 1024 or reserve_mib != 256):
        raise ValueError("Diagnostic is pinned to 15275/4/Balanced/Tile32/1024/256")
    torch.manual_seed(42026)
    shape = (1, heads, tokens, 128)
    bank = tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                 for _ in range(3))
    results = {}
    env_keys = ("APL_HTRC_DIRECTFLOW_REPORT",
                "APL_HTRC_DIRECTFLOW_GROUP",
                "APL_HTRC_DIRECTFLOW_WARMUPS",
                "APL_HTRC_DIRECTFLOW_VB_MODE")
    old = {key: os.environ.get(key) for key in env_keys}
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-fused-numeric-") as tmp:
            os.environ["APL_HTRC_DIRECTFLOW_REPORT"] = str(Path(tmp)/"gate.json")
            os.environ["APL_HTRC_DIRECTFLOW_GROUP"] = "11"
            os.environ["APL_HTRC_DIRECTFLOW_WARMUPS"] = "4"
            for variant in ("materialized", "fused_vb"):
                flow.clear_session_cache()
                os.environ["APL_HTRC_DIRECTFLOW_VB_MODE"] = variant
                # compare_only disables the internal eager gate ONLY for
                # this opt-in probe; explicit full eager checks follow.
                value = flow.graph_attention_bthd(
                    *bank, heads=heads, mode=mode, tiles=tiles,
                    budget_mib=budget_mib, reserve_mib=reserve_mib,
                    workspace=True, compare_only=True,
                )
                torch.cuda.synchronize()
                results[variant] = value.detach().clone()
            reference = live.flat_attention_bthd(
                *bank, heads=heads, mode=mode, tiles=tiles,
                budget_mib=budget_mib, reserve_mib=reserve_mib,
                workspace=True, query_staging="native_torch_mu",
                kv_staging="native_direct", dispatch="eager",
            )
            torch.cuda.synchronize()
            comparisons = {
                "materialized_vs_eager":_difference_summary(
                    reference, results["materialized"]),
                "fused_vb_vs_eager":_difference_summary(
                    reference, results["fused_vb"]),
                "fused_vb_vs_materialized":_difference_summary(
                    results["materialized"], results["fused_vb"]),
            }
            if not comparisons["materialized_vs_eager"]["bit_exact"]:
                raise AssertionError(
                    "Diagnostic control failed: materialized Graph differs from eager"
                )
            mismatch = not comparisons["fused_vb_vs_eager"]["bit_exact"]
            return {
                "status":("DIAG_NUMERIC_MISMATCH" if mismatch
                          else "DIAG_MATCH_NOT_ACCEPTED"),
                "completed":True,
                "numerical_gate":("FAIL" if mismatch
                                  else "NOT_AN_ACCEPTANCE_TEST"),
                "strict_first_call_gate_changed":False,
                "compare_only_for_diagnosis":True,
                "performance_measured":False,
                "real_h3_or_video":"UNRUN",
                "promotion_to_real_h3":"PROHIBITED_DIAGNOSTIC_ONLY",
                "tokens":tokens, "heads":heads, "mode":mode,
                "tiles":tiles, "exact_leaves":512,
                "gpu":torch.cuda.get_device_name(0),
                "comparisons":comparisons,
                "notes":[
                    "One synthetic QKV bank; same deterministic first bank as strict benchmark",
                    "Full final-output BF16 comparisons, no silent tolerance",
                    "This localizes output errors, not which internal Triton stage changed",
                    "No performance or independent CUDA Graph memory comparison",
                ],
            }
    finally:
        flow.clear_session_cache()
        for key in env_keys:
            if old[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old[key]


@torch.inference_mode()
def run_ab(tokens, heads, mode, tiles, rounds, budget_mib, reserve_mib, output):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    if (tokens<16 or heads<1 or heads>56 or tiles not in (16,32,64)
        or rounds<2 or budget_mib<=reserve_mib+128):
        raise ValueError("Invalid benchmark shape or budget")
    torch.manual_seed(42026)
    shape=(1,heads,tokens,128)
    banks=[tuple(torch.randn(shape,device="cuda",dtype=torch.bfloat16)
                 for _ in range(3)) for _ in range(2)]
    saved={}
    metrics={}
    gates={}
    # First invocation in each mode checks ALL groups against exactly the
    # same eager HTRC numerical path; then second bank proves QKV refresh.
    old=os.environ.copy()
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-fused-vb-") as tmp:
            os.environ["APL_HTRC_DIRECTFLOW_REPORT"]=str(Path(tmp)/"gate.json")
            os.environ["APL_HTRC_DIRECTFLOW_GROUP"]="11"
            os.environ["APL_HTRC_DIRECTFLOW_WARMUPS"]="4"
            for variant in ("materialized","fused_vb"):
                flow.clear_session_cache()
                os.environ["APL_HTRC_DIRECTFLOW_VB_MODE"]=variant
                def calculate(bank):
                    return flow.graph_attention_bthd(
                        *bank,heads=heads,mode=mode,tiles=tiles,
                        budget_mib=budget_mib,reserve_mib=reserve_mib,
                        workspace=True)
                first=calculate(banks[0])
                torch.cuda.synchronize()
                gate=json.loads((Path(tmp)/"gate.json").read_text(encoding="utf-8"))
                if gate.get("status")!="PASS_FIRST_REAL_CALL" or not gate.get("all_groups_bit_exact"):
                    raise AssertionError(f"Initial eager/Graph parity failed: {variant}")
                # The model-resident status field in the generic gate
                # describes its *source shape* contract; this probe uses
                # non-model random tensors and explicitly overrides that.
                gates[variant]={
                    "status":gate["status"],
                    "first_call_bit_exact":bool(gate["all_groups_bit_exact"]),
                    "head_groups":len(gate["head_groups"]),
                    "extra_reserved_mib":gate["extra_reserved_mib"],
                    "capture_and_parity_ms":gate["capture_and_parity_first_call_wall_ms"],
                }
                saved[(variant,0)]=first.clone()
                second=calculate(banks[1])
                torch.cuda.synchronize()
                saved[(variant,1)]=second.clone()
                # Mutable source bank B is independently compared with
                # the proven native eager path, not merely the other Graph.
                eager_b=live.flat_attention_bthd(
                    *banks[1],heads=heads,mode=mode,tiles=tiles,
                    budget_mib=budget_mib,reserve_mib=reserve_mib,
                    workspace=True,query_staging="native_torch_mu",
                    kv_staging="native_direct",dispatch="eager",
                )
                gates[variant]["bank_b_eager_bit_exact"]=bool(
                    torch.equal(second,eager_b))
                if not gates[variant]["bank_b_eager_bit_exact"]:
                    raise AssertionError(f"Bank B eager parity FAILED: {variant}")
                del eager_b
                if torch.equal(first,second):
                    raise AssertionError("Bank A/B outputs should differ")
                # Exclude first-call capture and eager parity from timing.
                warmed,metrics[variant]=_timed(lambda:calculate(banks[1]),rounds)
                if not torch.equal(warmed,second):
                    raise AssertionError(f"Mutable bank replay changed: {variant}")
                del first,second,warmed
    finally:
        flow.clear_session_cache()
        for key in ("APL_HTRC_DIRECTFLOW_REPORT",
                    "APL_HTRC_DIRECTFLOW_GROUP",
                    "APL_HTRC_DIRECTFLOW_WARMUPS",
                    "APL_HTRC_DIRECTFLOW_VB_MODE"):
            if key in old:
                os.environ[key]=old[key]
            else:
                os.environ.pop(key,None)
    exact={
        f"bank_{bank}_materialized_vs_fused_vb":bool(torch.equal(
            saved[("materialized",bank)],saved[("fused_vb",bank)]
        )) for bank in (0,1)
    }
    if not all(exact.values()):
        raise AssertionError("Whole-output Graph VB mode bit-exact mismatch: "+str(exact))
    return {
        "status":"PASS_NUMERIC",
        "completed":True,
        "synthetic_nonzero_native_bf16_banks":2,
        "all_distinct_heads":True,
        "heads":heads,"tokens":tokens,"mode":mode,"window_tiles":tiles,
        "rounds":rounds,"budget_mib":budget_mib,"reserve_mib":reserve_mib,
        "numeric":exact,
        "eager_graph_first_call_gates":gates,
        "gpu":torch.cuda.get_device_name(0),
        "warm_graph_timings":metrics,
        "wall_ratio_materialized_over_fused_vb":(
            metrics["materialized"]["sync_wall_ms_per_call"]/
            metrics["fused_vb"]["sync_wall_ms_per_call"]
        ),
        "new_path_quality_coefficient_used":False,
        "near_leaves":256 if mode=="fast" else 512,
        "far_stream_producer_memory":"none; VB computed inside Tile CTA",
        "scope":"synthetic full-attention A/B; no H3 checkpoint, no Sol comparison",
        "real_video":"UNRUN",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens",type=int,default=15275)
    ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--mode",choices=("fast","balanced"),default="fast")
    ap.add_argument("--tiles",type=int,choices=(16,32,64),default=16)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--budget-mib",type=int,default=1024)
    ap.add_argument("--reserve-mib",type=int,default=256)
    ap.add_argument("--diagnose-numeric",action="store_true",
                    help="strict-failure analysis only; never PASS_NUMERIC")
    ap.add_argument("--output",type=Path,default=Path(
        r"D:\ComfyOutput\H3\math_probe\flat_fused_vb_ab.json"))
    args=ap.parse_args()
    result={
        "experiment":"h3_flat_fused_vb_vs_materialized_graph_ab_v1",
        "status":"FAIL_CLOSED","completed":False,
    }
    try:
        if args.diagnose_numeric:
            result.update(diagnose_numeric(
                args.tokens,args.heads,args.mode,args.tiles,
                args.budget_mib,args.reserve_mib))
        else:
            result.update(run_ab(
                args.tokens,args.heads,args.mode,args.tiles,args.rounds,
                args.budget_mib,args.reserve_mib,args.output))
    except BaseException as exc:
        result["error"]=f"{type(exc).__name__}: {exc}"[:1500]
        raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        print(f"one_report={args.output} status={result['status']}")


if __name__=="__main__":
    main()
