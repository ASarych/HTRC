"""Read-only, source-grounded HTRC Flat long-context traffic/planner model.

This is NOT a GPU benchmark, an attention time predictor, or a model VRAM
admission decision. All per-head staging formulas follow the research
version of h3/htrc_flat_head_batch.py and h3/htrc_flat_model_shadow.py.
The model is deliberately independent from ComfyUI, Torch and Triton.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MIB = 1024**2
GIB = 1024**3
# Fixed implementation: 64 query rows/centre, 16 key tokens/leaf,
# BF16 K/V, INT8 G, BF16 VB, FP32 LZ/SG/SEL.
ROWS = 64
LEAF = 16
DIM = 128
EXACT = {"fast": 256, "balanced": 512}
STATE_BYTES = DIM + 2*DIM + 4 + 4          # G, VB, LZ, SG: 392
SCORE_BYTES = 4                           # SEL
EXCL_BYTES = 1                            # exclusion mask


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def estimate_group(*, tokens: int, heads: int, tiles: int, mode: str,
                   budget_mib: int, reserve_mib: int, max_group: int) -> dict:
    """Replicate both HTRC conservative group-planner clamps.

    This is a capacity upper bound from a configured workspace budget.
    The actual model-resident free VRAM, graph pool, and other allocations
    can reduce the feasible cap or reject the call.
    """
    nkeys = tokens // LEAF * LEAF
    leaves = nkeys // LEAF
    kx = min(EXACT[mode], leaves)
    centres = tiles                        # center_group=1 only
    splits = 4
    rows = tiles * ROWS

    # estimate_tile_head_bytes(... partial_copies=2), then *1.5
    q_mu = rows*DIM*4 + centres*DIM*4
    near = nkeys*(DIM + DIM*2 + 4)
    far = centres*leaves*(DIM + DIM*2 + 4 + 4 + EXCL_BYTES)
    index = centres*kx*4
    partial = 2*tiles*splits*ROWS*(DIM+2)*4
    raw = q_mu + near + far + index + partial
    per_estimate = math.ceil(raw*1.5)

    # _head_plan's extra resident KBF/VBF/K8/KS + workspace estimate.
    per_cache = nkeys*(DIM*2 + DIM*2 + DIM + 4)
    per_workspace = (
        centres*leaves*(DIM + DIM*2 + 4 + 4 + 4 + EXCL_BYTES)
        + centres*kx*4
        + tiles*splits*ROWS*(4+4+DIM*4)
        + tiles*ROWS*DIM*4
    )
    per_head = per_estimate + math.ceil(per_cache*1.3) + math.ceil(per_workspace*1.3)
    allowance = (budget_mib-reserve_mib-128)*MIB
    # Flat requires the budget to exceed reserve + fixed 128 MiB.
    if allowance <= 0 or per_head > allowance:
        return {"feasible_under_budget": False,
                "group_cap": 0, "group_count": None,
                "window_count": ceildiv(tokens,rows), "graph_replays": None,
                "estimate_mib_per_head": round(per_head/MIB, 3),
                "raw_mib_per_head": round(raw/MIB, 3)}
    groups = min(heads,max_group,allowance//per_estimate,allowance//per_head)
    groups = int(groups)
    group_count = ceildiv(heads, groups)
    return {
        "feasible_under_budget": True,
        "group_cap": groups,
        "group_count": group_count,
        "window_count": ceildiv(tokens,rows),
        "graph_replays": group_count*ceildiv(tokens,rows),
        "estimate_mib_per_head": round(per_head/MIB, 3),
        "raw_mib_per_head": round(raw/MIB, 3),
    }


def estimate_case(*, tokens: int, heads: int = 56, mode: str = "fast",
                  budget_mib: int = 1024, reserve_mib: int = 256,
                  max_group: int = 11, candidates: tuple[int, ...] = (16,32,64),
                  mem_bandwidth_gbps: float = 896.0) -> dict:
    if not isinstance(tokens,int) or tokens < 16:
        raise ValueError("tokens must be an integer >= 16")
    if not isinstance(heads,int) or heads < 1:
        raise ValueError("heads must be >= 1")
    if mode not in EXACT:
        raise ValueError("Only fixed fast/balanced modes supported")
    if not (0 < max_group <= heads and reserve_mib >= 0 and
            budget_mib > reserve_mib + 128):
        raise ValueError("Invalid conservative group budget")
    if not candidates or any(
        not isinstance(t,int) or t not in (16,32,64) for t in candidates
    ):
        raise ValueError("Candidate tiles must be among 16,32,64")
    if not (0 < mem_bandwidth_gbps < math.inf):
        raise ValueError("Bandwidth must be finite and positive")

    centres = ceildiv(tokens,ROWS)
    leaves = tokens // LEAF
    pairs = heads*centres*leaves
    exact = min(EXACT[mode],leaves)*LEAF
    # The following are total logical bytes processed in ONE full attention
    # invocation across 56 heads and all query windows, not simultaneous VRAM.
    state_write = pairs*STATE_BYTES
    score_write = pairs*SCORE_BYTES
    excl_write = pairs*EXCL_BYTES
    rowmax_read = pairs*(DIM + 4 + 4)     # G, LZ, SG
    far_read = pairs*STATE_BYTES        # G, VB, LZ, SG once at group1
    # The hypothetical score-only/recompute-far redesign emits SEL/EXCL,
    # then recomputes stats in the far kernel; no GPU implementation yet.
    # It trades lower DRAM traffic for additional K/V dot products.
    q_or_kv = heads*tokens*DIM*2
    baseline_traffic = state_write+score_write+excl_write+rowmax_read+far_read
    candidate_stage_write = score_write+excl_write
    runs = [
        {"tiles":t,**estimate_group(tokens=tokens,heads=heads,tiles=t,
                                   mode=mode,budget_mib=budget_mib,
                                   reserve_mib=reserve_mib,max_group=max_group)}
        for t in candidates
    ]
    valid = [r for r in runs if r["feasible_under_budget"]]
    best = min(valid, key=lambda r:(r["graph_replays"],r["tiles"])) if valid else None

    return {
        "tokens":tokens,"heads":heads,"head_dim":DIM,"mode":mode,
        "exact_leaves":EXACT[mode],"full_key_leaves":leaves,
        "true_query_centres":centres,
        "selected_key_fraction":round(exact/tokens,6),
        "per_head_bf16_q_or_kv_gib":round(q_or_kv/GIB,4),
        "full_bf16_qkv_and_output_gib":round(4*q_or_kv/GIB,4),
        "centre_leaf_pairs":pairs,
        "state_write_gib_per_attention":round(state_write/GIB,4),
        "rowmax_read_gib_per_attention":round(rowmax_read/GIB,4),
        "far_read_gib_per_attention":round(far_read/GIB,4),
        "baseline_ideal_stage_traffic_gib_per_attention":round(baseline_traffic/GIB,4),
        "two_pass_score_only_write_gib_per_attention":round(candidate_stage_write/GIB,4),
        "stage_write_reduction_theoretical":round(
            1-candidate_stage_write/(state_write+score_write+excl_write),6
        ),
        "min_state_write_seconds_at_bandwidth":round(
            state_write/(mem_bandwidth_gbps*10**9),6
        ),
        "candidate_must_recompute_key_stats":True,
        "graph_tile_candidates":runs,
        "least_replay_tiles_under_configured_budget":(
            best["tiles"] if best else None
        ),
        "capacity_not_a_real_vram_admission":True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens",default="15275,18723,40759,82000,223605")
    ap.add_argument("--heads",type=int,default=56)
    ap.add_argument("--mode",choices=tuple(EXACT),default="fast")
    ap.add_argument("--budgets",default="1024,2048")
    ap.add_argument("--reserve-mib",type=int,default=256)
    ap.add_argument("--max-group",type=int,default=11)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    rows=[]
    for n in [int(v.strip()) for v in args.tokens.split(",")]:
        for budget in [int(v.strip()) for v in args.budgets.split(",")]:
            rows.append(estimate_case(tokens=n,heads=args.heads,mode=args.mode,
                                      budget_mib=budget,reserve_mib=args.reserve_mib,
                                      max_group=args.max_group))
    report={
        "schema":"h3-flat-long-context-source-model-v1",
        "status":"ANALYTIC_ONLY_NO_GPU_RUN",
        "source_branch_base":"experiment/flat-directflow-real-h3-20261112",
        "fixed_math":True,"quality_coefficient_used":False,
        "candidate_tiles":[16,32,64],
        "model_scope":"source formulas only; neither Sol crossover nor video speed",
        "assumptions":[
            "One 64-row query centre per key leaf, head dimension 128.",
            "BF16 QKV, INT8 G, BF16 VB, FP32 LZ/SG/SEL, exact tail retained.",
            "The score-only streaming candidate repeats leaf stats after top-k.",
            "Conservative fixed-budget cap can understate or overstate actual VRAM availability.",
            "Graph replay count is not GPU kernel count or a timing prediction.",
            "DRAM bytes are logical reads/writes, not memory profiler measurements.",
        ],
        "cases":rows
    }
    raw=json.dumps(report,indent=2,ensure_ascii=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(raw,encoding="utf-8")
        print(str(args.output))
    else:
        print(raw,end="")


if __name__=="__main__":
    main()
