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
import math
import statistics
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


def _abba_speed_verdict(materialized_ms, fused_ms):
    """Pure, CPU-testable speed signal. Never authorizes production."""
    if len(materialized_ms) != 2 or len(fused_ms) != 2:
        raise ValueError("ABBA requires two independent block medians per variant")
    if any(not math.isfinite(v) or v <= 0
           for v in materialized_ms + fused_ms):
        raise ValueError("ABBA GPU event medians must be finite and positive")
    materialized = float(statistics.median(materialized_ms))
    fused = float(statistics.median(fused_ms))
    ratio = materialized / fused
    all_faster = max(fused_ms) < min(materialized_ms)
    return ratio, all_faster, bool(ratio >= 1.05 and all_faster)


@torch.inference_mode()
def run_abba(tokens, heads, mode, tiles, rounds, budget_mib, reserve_mib):
    """Four strict, independent Graph captures on identical Q/K/V banks.

    Only a research stability gate: not a real-H3 or VRAM comparison.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    if (tokens, heads, mode, tiles, budget_mib, reserve_mib) != (
        15275, 4, "balanced", 32, 1024, 256
    ) or rounds < 5 or rounds > 12:
        raise ValueError("ABBA requires pinned 15275/4/balanced/32/1024/256, rounds 5..12")
    order = ("materialized", "fused_vb", "fused_vb", "materialized")
    torch.manual_seed(42026)
    shape = (1, heads, tokens, 128)
    banks = [
        tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16)
              for _ in range(3))
        for _ in range(2)
    ]
    baseline_outputs = {}  # CPU copies, not persistent GPU allocations
    blocks = []
    env_keys = ("APL_HTRC_DIRECTFLOW_REPORT",
                "APL_HTRC_DIRECTFLOW_GROUP",
                "APL_HTRC_DIRECTFLOW_WARMUPS",
                "APL_HTRC_DIRECTFLOW_VB_MODE")
    old = {name: os.environ.get(name) for name in env_keys}
    device = banks[0][0].device
    try:
        with tempfile.TemporaryDirectory(prefix="htrc-fused-abba-") as temp:
            os.environ["APL_HTRC_DIRECTFLOW_REPORT"] = str(Path(temp)/"gate.json")
            os.environ["APL_HTRC_DIRECTFLOW_GROUP"] = "11"
            os.environ["APL_HTRC_DIRECTFLOW_WARMUPS"] = "4"
            for block_index, variant in enumerate(order, start=1):
                flow.clear_session_cache()
                os.environ["APL_HTRC_DIRECTFLOW_VB_MODE"] = variant

                def calculate(bank):
                    return flow.graph_attention_bthd(
                        *bank, heads=heads, mode=mode, tiles=tiles,
                        budget_mib=budget_mib, reserve_mib=reserve_mib,
                        workspace=True,
                    )

                first = calculate(banks[0])
                torch.cuda.synchronize(device)
                gate = json.loads((Path(temp)/"gate.json").read_text(encoding="utf-8"))
                if (gate.get("status") != "PASS_FIRST_REAL_CALL" or
                    gate.get("all_groups_bit_exact") is not True):
                    raise AssertionError(
                        f"{variant} block{block_index}: strict eager Graph gate FAIL"
                    )
                first_cpu = first.detach().cpu()
                second = calculate(banks[1])
                torch.cuda.synchronize(device)
                second_cpu = second.detach().cpu()
                if torch.equal(first_cpu, second_cpu):
                    raise AssertionError("QKV bank A/B outputs must differ")
                eager_b = live.flat_attention_bthd(
                    *banks[1], heads=heads, mode=mode, tiles=tiles,
                    budget_mib=budget_mib, reserve_mib=reserve_mib,
                    workspace=True, query_staging="native_torch_mu",
                    kv_staging="native_direct", dispatch="eager",
                )
                if not torch.equal(second, eager_b):
                    raise AssertionError(
                        f"{variant} block{block_index}: bank B eager parity failed"
                    )
                del eager_b
                for bank_index, value in enumerate((first_cpu, second_cpu)):
                    key = (variant, bank_index)
                    if key not in baseline_outputs:
                        baseline_outputs[key] = value.clone()
                    elif not torch.equal(baseline_outputs[key], value):
                        raise AssertionError(
                            f"{variant} block{block_index}: repeat capture output changed"
                        )
                # An independent materialized/fused comparison is made
                # after all four blocks, on both complete CPU output banks.
                del first, first_cpu, second_cpu

                # Five extra FULL attention replays after capture and eager
                # parity, before the first timed GPU event.
                warmed = None
                for _ in range(5):
                    warmed = calculate(banks[1])
                torch.cuda.synchronize(device)
                if not torch.equal(warmed, second):
                    raise AssertionError(
                        f"{variant} block{block_index}: post-capture replay changed"
                    )
                del warmed
                gpu_samples = []
                wall_samples = []
                last = None
                for _ in range(rounds):
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    wall_started = time.perf_counter()
                    last = calculate(banks[1])
                    stop.record()
                    stop.synchronize()
                    gpu_samples.append(float(start.elapsed_time(stop)))
                    wall_samples.append((time.perf_counter() - wall_started) * 1000.0)
                if not torch.equal(last, second):
                    raise AssertionError(
                        f"{variant} block{block_index}: timed replay changed"
                    )
                blocks.append({
                    "capture_block": block_index,
                    "variant": variant,
                    "first_call_gate": gate["status"],
                    "bank_a_bit_exact_to_eager": True,
                    "bank_b_bit_exact_to_eager": True,
                    "bank_b_replay_bit_exact": True,
                    "post_capture_warmup_replays": 5,
                    "gpu_event_samples_ms": gpu_samples,
                    "sync_wall_samples_ms": wall_samples,
                    "gpu_event_median_ms": float(statistics.median(gpu_samples)),
                    "sync_wall_median_ms": float(statistics.median(wall_samples)),
                    "first_call_extra_reserved_mib_non_isolated": gate.get("extra_reserved_mib"),
                })
                del last, second
                flow.clear_session_cache()
        parity = {}
        for bank_index in (0,1):
            parity[f"bank_{bank_index}_materialized_vs_fused_vb"] = bool(
                torch.equal(
                    baseline_outputs[("materialized",bank_index)],
                    baseline_outputs[("fused_vb",bank_index)],
                )
            )
        if not all(parity.values()):
            raise AssertionError("ABBA full-output cross-variant bit-exact failure")
        by_variant = {
            name: [row["gpu_event_median_ms"] for row in blocks
                   if row["variant"] == name]
            for name in ("materialized", "fused_vb")
        }
        old_ms = float(statistics.median(by_variant["materialized"]))
        fused_ms = float(statistics.median(by_variant["fused_vb"]))
        ratio, faster_every_block, signal = _abba_speed_verdict(
            by_variant["materialized"], by_variant["fused_vb"]
        )
        return {
            "experiment": "h3_fused_vb_materialized_independent_capture_abba_v1",
            "status": "PASS_NUMERIC",
            "completed": True,
            "paired_abba": True,
            "capture_order": list(order),
            "tokens": tokens, "heads": heads, "mode": mode,
            "window_tiles": tiles, "near_leaves": 512,
            "rounds_per_block": rounds,
            "extra_post_capture_warmups_per_block": 5,
            "qkv_banks": 2,
            "scope": "synthetic complete Graph attention; no model/VAE/video",
            "numeric": parity,
            "per_block": blocks,
            "block_gpu_event_medians_ms": by_variant,
            "median_of_block_gpu_event_medians_ms": {
                "materialized": old_ms,
                "fused_vb": fused_ms,
            },
            "gpu_speed_ratio_materialized_over_fused_vb": ratio,
            "consistent_faster_across_capture_blocks": faster_every_block,
            "candidate_speed_signal": signal,
            "speed_signal_criterion": (
                ">=5% faster median of block medians AND both fused block "
                "medians strictly below both materialized block medians"
            ),
            "candidate_promotion": "NOT_AUTHORIZED_SYNTHETIC_ONLY",
            "gpu": torch.cuda.get_device_name(0),
            "memory_comparison": "NOT_ISOLATED: one-process allocator and Graph pools",
            "real_h3_or_video": "UNRUN",
        }
    finally:
        flow.clear_session_cache()
        for key in env_keys:
            if old[key] is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old[key]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens",type=int,default=15275)
    ap.add_argument("--heads",type=int,default=4)
    ap.add_argument("--mode",choices=("fast","balanced"),default="fast")
    ap.add_argument("--tiles",type=int,choices=(16,32,64),default=16)
    ap.add_argument("--rounds",type=int,default=3)
    ap.add_argument("--budget-mib",type=int,default=1024)
    ap.add_argument("--reserve-mib",type=int,default=256)
    ap.add_argument("--paired-abba",action="store_true",
                    help="four independent strict captures, separate GPU events")
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
        if args.diagnose_numeric and args.paired_abba:
            raise ValueError("Choose either diagnostic or strict ABBA")
        if args.paired_abba:
            result.update(run_abba(
                args.tokens,args.heads,args.mode,args.tiles,args.rounds,
                args.budget_mib,args.reserve_mib))
        elif args.diagnose_numeric:
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
