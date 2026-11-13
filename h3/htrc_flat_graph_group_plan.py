"""Research-only memory-to-speed Graph grouping model for native BF16 H3.

Estimate *persistently staged* Direct-Flow Graph memory, not the older
double-counted packed K/V + fresh per-window allocations. The model does
NOT predict CUDA Graph private pools, real VRAM peaks, latency or quality.
The runtime retains hard post-capture allocator/free-memory gates and
fails closed. No changes to normal 11-head legacy scheduling.
"""
from __future__ import annotations

import math

MIB = 1024**2
DIM = 128
LEAF = 16
ROWS = 64
SPLITS = 4
NEAR = {"fast": 256, "balanced": 512}
VB_MODES = ("materialized", "split_stream", "fused_vb")


def _ceildiv(a: int, b: int) -> int:
    return (a+b-1)//b


def _window_tiles(tokens: int, tiles: int) -> tuple[tuple[int,int], ...]:
    """Distinct Graph geometries: same tiles but different valid rows
    still require different CUDA Graph captures and scratch pools.
    """
    length=tiles*ROWS
    full, tail=divmod(tokens,length)
    result=[]
    if full:
        result.append((tiles,length))
    if tail:
        result.append((_ceildiv(tail,ROWS),tail))
    return tuple(result)


def graph_stage_head_bytes(*, tokens: int, tiles: int, mode: str,
                           vb_mode: str = "materialized") -> dict:
    """Live Graph allocations for ONE logical head/unique graph geometry.

    Native KV: Graph stage K/V (BF16) and cached K8/KS; native KBF/VBF
    alias the Graph stage buffers and are NEVER counted a second time.
    Q and workspaces exist once for each distinct full/tail geometry,
    not once per query window or repeated head group.
    """
    if type(tokens) is not int or tokens<16:
        raise ValueError("tokens must be an integer >=16")
    if type(tiles) is not int or tiles not in (16,32,64):
        raise ValueError("Graph tiles must be 16/32/64")
    if mode not in NEAR or vb_mode not in VB_MODES:
        raise ValueError("Unsupported fixed Flat mode/VB staging")
    aligned=(tokens//LEAF)*LEAF
    leaves=aligned//LEAF
    exact=min(NEAR[mode],leaves)
    stage_kv=2*tokens*DIM*2             # Graph stage BF16 K and V
    cache_k8_ks=aligned*(DIM+4)         # Native KBF/VBF are views
    tail=2*(tokens-aligned)*DIM*2      # Explicit exact K/V tail buffers
    windows=[]
    for active,valid in _window_tiles(tokens,tiles):
        centres=active                 # centre group1, fixed row 64
        q=active*ROWS*DIM*2
        g=centres*leaves*DIM           # int8 G
        vb=(centres*leaves*DIM*2 if vb_mode=="materialized" else 2)
        lz_sg_sel_excl=centres*leaves*(4+4+4+1)
        idx=centres*exact*4
        partial=active*SPLITS*ROWS*(DIM+2)*4
        out=active*ROWS*DIM*4
        split_scratch=0
        if vb_mode=="split_stream":
            far_bn=64
            span=max(far_bn,_ceildiv(_ceildiv(leaves,SPLITS),far_bn)*far_bn)
            split_scratch=centres*span*DIM*2
        total=(q+g+vb+lz_sg_sel_excl+idx+partial+out+split_scratch)
        windows.append({"tiles":active,"valid_rows":valid,
                        "bytes_per_head":total,
                        "q_bytes":q,"vb_bytes":vb,
                        "vb_split_bytes":split_scratch})
    state=sum(w["bytes_per_head"] for w in windows)
    per_head=stage_kv+cache_k8_ks+tail+state
    return {
        "tokens":tokens,"mode":mode,"vb_mode":vb_mode,
        "exact_leaves":NEAR[mode],
        "stage_bf16_kv_bytes_per_head":stage_kv,
        "native_kv_cache_k8_ks_bytes_per_head":cache_k8_ks,
        "duplicated_kbf_vbf_bytes_per_head":0,
        "tail_bytes_per_head":tail,
        "windows":windows,"unique_graph_geometries":len(windows),
        "estimated_persisted_bytes_per_head":per_head,
        "estimated_persisted_mib_per_head":round(per_head/MIB,3),
        "private_cuda_graph_pool_unmeasured":True,
    }



def research_forced_group_size(value, *, compare_only: bool, policy: str,
                               max_group: int, heads: int) -> int | None:
    """Opt-in research override; never bypass runtime CUDA VRAM gates.

    This may bypass the conservative *planner estimate* to test divisors
    such as 28 on real hardware. It is forbidden for production calls,
    where strict eager parity and planner protections must remain intact.
    """
    if value is None:
        return None
    if not compare_only or policy != "reuse_aware":
        raise ValueError(
            "Forced Graph group is allowed only for reuse_aware "
            "compare_only=True research calls"
        )
    if type(value) is not str or not value.isdecimal():
        raise ValueError("Forced Graph group must be a positive decimal integer")
    forced = int(value)
    if not 1 <= forced <= min(max_group, heads):
        raise ValueError(
            f"Forced Graph group {forced} exceeds selected cap/heads "
            f"{max_group}/{heads}"
        )
    return forced


def choose_group(*, tokens: int, heads: int, tiles: int, mode: str,
                 vb_mode: str, max_group: int, budget_mib: int,
                 reserve_mib: int, max_extra_mib: int, free_mib: float,
                 min_free_mib: int, safety_factor: float = 1.5,
                 fixed_margin_mib: int = 64) -> dict:
    """Pick fewer graph replays without exceeding conservative staged budget.

    A non-divisor g creates TWO resident Graph session sizes (g and
    heads%g), each with its own full/tail Graph pools. Their memory is
    summed; equal-sized groups reuse ONE session. Prefer fewer replays,
    then fewer distinct pools, then lower estimated memory.
    """
    if (type(heads) is not int or heads<1 or
        type(max_group) is not int or not 1<=max_group<=56 or
        type(budget_mib) is not int or
        type(reserve_mib) is not int or reserve_mib<0 or
        type(max_extra_mib) is not int or max_extra_mib<1 or
        type(min_free_mib) is not int or min_free_mib<0 or
        not math.isfinite(float(free_mib)) or free_mib<0 or
        not math.isfinite(safety_factor) or safety_factor<1.0 or
        type(fixed_margin_mib) is not int or fixed_margin_mib<0):
        raise ValueError("Invalid group or memory configuration")
    footprint=graph_stage_head_bytes(tokens=tokens,tiles=tiles,
                                     mode=mode,vb_mode=vb_mode)
    # Preserve the original 128 MiB scratch margin and external
    # resident-model reserve, then independently honor Graph-specific
    # total extra-reservation cap and the minimum free-VRAM gate.
    limits={
        "budget_minus_reserve_and_128_mib":budget_mib-reserve_mib-128,
        "graph_extra_cap_mib":max_extra_mib,
        "current_free_minus_floor_mib":free_mib-min_free_mib,
    }
    allowance=min(limits.values())
    candidates=[]
    for size in range(1,min(heads,max_group)+1):
        remainder=heads%size
        unique_sizes=(size, remainder) if remainder else (size,)
        resident_heads=sum(unique_sizes)
        estimated=(fixed_margin_mib + safety_factor *
                   footprint["estimated_persisted_bytes_per_head"]*
                   resident_heads/MIB)
        if estimated > allowance:
            continue
        groups=_ceildiv(heads,size)
        windows=_ceildiv(tokens,tiles*ROWS)
        candidates.append({
            "group_size":size,
            "group_count":groups,
            "unique_graph_session_sizes":list(unique_sizes),
            "estimated_total_extra_mib":round(estimated,3),
            "estimated_replays_per_attention":windows*groups,
        })
    if not candidates:
        raise MemoryError(
            "No Graph group fits stage+scratch estimate within "
            f"{allowance:.1f} MiB; leave legacy planner unchanged"
        )
    # Do not assume larger is always faster; this is the explicit
    # *minimum replay count* hypothesis for an opt-in GPU A/B.
    chosen=min(candidates,key=lambda c:(
        c["estimated_replays_per_attention"],
        len(c["unique_graph_session_sizes"]),
        c["estimated_total_extra_mib"],
        c["group_size"],
    ))
    return {
        "policy":"reuse_aware",
        "group_size":chosen["group_size"],
        "group_count":chosen["group_count"],
        "estimated_replays_per_attention":
            chosen["estimated_replays_per_attention"],
        "unique_graph_session_sizes":chosen["unique_graph_session_sizes"],
        "estimated_total_extra_mib":chosen["estimated_total_extra_mib"],
        "estimate_is_not_actual_cuda_graph_pool":True,
        "safety_factor":safety_factor,
        "fixed_margin_mib":fixed_margin_mib,
        "available_extra_mib":round(allowance,3),
        "limits_mib":{k:round(v,3) for k,v in limits.items()},
        "head_footprint":footprint,
        "feasible_candidates":candidates,
        "speedup_status":"UNRUN",
        "quality_status":"UNRUN",
    }
