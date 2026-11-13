"""HTRC Flat benchmark scope and hypothesis registry (stdlib only).

Classification is a research hypothesis, NOT established speedup.
Every executable long-context scope includes the two short quality/latency cases.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeCase:
    name: str
    tokens: int
    band: str
    gpu_eligible: bool = True


CASES = (
    ShapeCase("short_h3_15k", 15275, "short"),
    ShapeCase("short_h3_18k", 18723, "short"),
    ShapeCase("medium_h3_40k", 40759, "medium"),
    ShapeCase("long_h3_82k", 82000, "long"),
    ShapeCase("fullhd_h3_223k", 223605, "capacity", False),
)
SCOPES = {
    "short": ("short_h3_15k", "short_h3_18k"),
    "scaling": ("short_h3_15k", "short_h3_18k",
                "medium_h3_40k", "long_h3_82k"),
    "capacity": ("short_h3_15k", "short_h3_18k",
                 "medium_h3_40k", "long_h3_82k", "fullhd_h3_223k"),
}

# Key is strategy/implementation id; class predicts where benefits might
# appear and is never evidence of a measured speed gain.
APPROACHES = {
    "graph_dispatch": {
        "class": "universal_candidate",
        "target": "host launches and scheduling across all sequence lengths",
        "baseline": "eager Flat",
        "validation": "short and long latency + percent error vs exact attention",
    },
    "native_qkv_staging": {
        "class": "universal_candidate",
        "target": "unnecessary QKV copies, quantization staging",
        "baseline": "same math with old staging",
        "validation": "short and all other lengths; error % vs dense reference",
    },
    "stats_score_only": {
        "class": "universal_candidate",
        "target": "remove first-pass Vbar dot/store",
        "baseline": "materialized stats",
        "validation": "stats-only micro A/B first; full-attention quality separately",
    },
    "graph_window_tiles": {
        "class": "length_amplified_candidate",
        "target": "group cap and graph replay count depend on N and budget",
        "baseline": "tiles16 with identical QKV and same mode",
        "validation": "short/medium/long: quality % vs dense, timing and memory",
    },
    "split_stream": {
        "class": "length_amplified_candidate",
        "target": "bounded Vbar scratch; N-dependent group size",
        "baseline": "graph + materialized",
        "validation": "two-bank percent error to dense: short, medium, long",
    },
    "fused_vb": {
        "class": "length_amplified_candidate",
        "target": "avoid global Vbar but potentially 16x extra K/V traffic",
        "baseline": "graph + materialized",
        "validation": "short dense-reference error % and speed; then medium/long",
    },
    "multi_centre_kv_reuse": {
        "class": "length_amplified_candidate",
        "target": "reuse native K/V across many query centres",
        "baseline": "graph + materialized",
        "validation": "short quality/latency regression + medium/long crossover",
    },
    "qkv_streaming": {
        "class": "capacity_enabler",
        "target": "resident QKV+output need 11.94 GiB at 223605 tokens",
        "baseline": "unmodified H3 memory preflight",
        "validation": "short error %, long peak VRAM and Full HD admission",
    },
    "workspace_grouping": {
        "class": "capacity_enabler",
        "target": "prevent small model-resident free VRAM from shrinking groups",
        "baseline": "old conservative planner",
        "validation": "short quality %, short latency, long VRAM/group count",
    },
}

GPU_VARIANTS = ("split_stream", "fused_vb")


def select_cases(scope: str) -> tuple[ShapeCase, ...]:
    if scope not in SCOPES:
        raise ValueError(f"Invalid scope: {scope}")
    wanted = SCOPES[scope]
    selected = tuple(c for c in CASES if c.name in wanted)
    # Long-only GPU experiments would hide short-sequence regressions.
    assert len(selected) >= 2 and all(c.band == "short" for c in selected[:2])
    assert len({c.name for c in selected}) == len(selected)
    return selected


def plan(variants: tuple[str, ...], scope: str,
         *, heads: int, tiles: int, mode: str,
         rounds: int, budget_mib: int, reserve_mib: int) -> dict:
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("Select one or two distinct variants")
    if any(v not in GPU_VARIANTS for v in variants):
        raise ValueError("Only implemented experimental GPU variants supported")
    if (heads < 1 or heads > 56 or tiles not in (16, 32, 64)
        or mode not in ("fast", "balanced") or rounds < 2
        or budget_mib <= reserve_mib + 128 or reserve_mib < 0):
        raise ValueError("Invalid shape or conservative GPU budget")
    selected = select_cases(scope)
    return {
        "schema": "h3-flat-length-gate-v1",
        "purpose": "short+long quality percent vs exact dense + speed, one JSON",
        "reference": "exact_FP32_softmax_over_all_keys_sampled_queries",
        "disagreement_policy": "record_and_continue",
        "bit_exact_between_flat_modes_required": False,
        "classification_is_hypothesis_not_evidence": True,
        "scope": scope,
        "baseline": "graph_materialized",
        "variants": list(variants),
        "approach_classes": {v: APPROACHES[v]["class"] for v in variants},
        "heads": heads, "window_tiles": tiles, "mode": mode,
        "rounds_per_case": rounds,
        "budget_mib": budget_mib, "reserve_mib": reserve_mib,
        "cases": [
            {"id": c.name, "tokens": c.tokens, "band": c.band,
             "gpu_eligible": c.gpu_eligible,
             "required_short_regression": c.band == "short"}
            for c in selected
        ],
        "no_fullhd_gpu_execution": True,
        "capacity_admission_is_analytic_only": True,
        "no_sol_comparison_in_this_runner": True,
    }
