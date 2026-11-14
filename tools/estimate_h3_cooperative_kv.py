"""CPU-only source-traffic and CTA ownership admission for HTRC Flat.

Not CUDA timing, physical DRAM bytes, measured occupancy, or memory admission.
See research/H3_FLAT_KV_COOPERATIVE_ADMISSION_2026-11-14.md.
"""
from __future__ import annotations

import argparse
import json
import math

DIM = 128
LEAF = 16
Q_ROWS = 64
STATS_BT = 16
EXACT = {"fast": 256, "balanced": 512}
K_OR_V_LEAF_BYTES = LEAF * DIM * 2  # BF16
VB_BYTES_PER_PAIR = DIM * 2
W_FP32_BYTES_PER_PAIR = LEAF * 4
FP32_ACC_BYTES_PER_CENTRE = Q_ROWS * DIM * 4
REFERENCE_LOGICAL_BYTES_PER_PAIR = (
    2 * K_OR_V_LEAF_BYTES / STATS_BT + 2 * VB_BYTES_PER_PAIR
)


def estimate(*, tokens: int, heads: int = 4, tiles: int = 32,
             mode: str = "balanced", centres_per_cta: int = 8,
             query_row_groups: int = 1) -> dict:
    if type(tokens) is not int or tokens < 16:
        raise ValueError("tokens must be an integer >=16")
    if type(heads) is not int or not 1 <= heads <= 56:
        raise ValueError("heads must be 1..56")
    if tiles not in (16, 32, 64) or mode not in EXACT:
        raise ValueError("Pinned Flat only: tiles16/32/64 fast/balanced")
    if centres_per_cta not in (1, 2, 4, 8, 16) or tiles % centres_per_cta:
        raise ValueError("Centre group must divide one full Tile window")
    if query_row_groups not in (1, 2, 4, 8) or Q_ROWS % query_row_groups:
        raise ValueError("Query row grouping must exactly divide 64")

    leaves = tokens // LEAF
    selected = min(leaves, EXACT[mode])
    far = leaves - selected
    far_fraction = far / leaves
    b = centres_per_cta
    r = query_row_groups
    # Independent near-set assumption is illustrative, NOT measured.
    union_far_fraction_independent = 1 - (selected / leaves) ** b
    upper_union = 1.0 if far else 0.0
    def direct(union: float) -> float:
        # SCORE_ONLY stats reads K once per 16 centres; far recomputation
        # rereads both K and V for each cooperating CTA + query-row group.
        return (K_OR_V_LEAF_BYTES / STATS_BT
                + union * 2 * K_OR_V_LEAF_BYTES * r / b)
    def w_staged(union: float) -> float:
        # Store lossless FP32 16-way weights in Stage A; each row group
        # rereads weights and V in Stage B (no recomputed MU.K softmax).
        return (K_OR_V_LEAF_BYTES / STATS_BT
                + W_FP32_BYTES_PER_PAIR * (1 + r)
                + union * K_OR_V_LEAF_BYTES * r / b)

    accum_bytes = b * (Q_ROWS // r) * DIM * 4
    pairs = heads * tiles * leaves   # one FULL graph window; not whole video
    return {
        "status": "ANALYTIC_ONLY_NO_GPU",
        "tokens": tokens, "heads": heads, "window_tiles": tiles,
        "mode": mode, "full_key_leaves": leaves,
        "exact_near_leaves": selected, "far_leaves": far,
        "far_fraction": round(far_fraction, 9),
        "centres_per_cta": b, "query_row_groups": r,
        "rows_per_centre_per_cta": Q_ROWS // r,
        "split_ctas_per_head_window": (tiles // b) * 4 * r,
        "logical_accumulator_kib_per_cta": round(accum_bytes / 1024, 3),
        "logical_accumulator_f32_elements_per_cta": accum_bytes // 4,
        "reference_logical_bytes_per_centre_leaf": REFERENCE_LOGICAL_BYTES_PER_PAIR,
        "source_model_scope": (
            "ONLY amortized BF16 K/V reads and BF16 VB write/read; "
            "unchanged G/LZ/SG/EXCL/IDX/Q/near/tail omitted"
        ),
        "union_far_fraction_independent_assumption": round(union_far_fraction_independent, 9),
        "direct_recompute_bytes_per_pair": {
            "optimistic_same_far_sets": round(direct(far_fraction), 3),
            "illustrative_independent_far_sets": round(direct(union_far_fraction_independent), 3),
            "worst_union_all_leaves": round(direct(upper_union), 3),
        },
        "fp32_weight_staging_bytes_per_pair": {
            "optimistic_same_far_sets": round(w_staged(far_fraction), 3),
            "illustrative_independent_far_sets": round(w_staged(union_far_fraction_independent), 3),
            "worst_union_all_leaves": round(w_staged(upper_union), 3),
        },
        "reference_vb_window_mib": round(pairs * VB_BYTES_PER_PAIR / (1024**2), 3),
        "fp32_weights_window_mib": round(pairs * W_FP32_BYTES_PER_PAIR / (1024**2), 3),
        "independent_traffic_below_reference": {
            "direct_recompute": direct(union_far_fraction_independent) < REFERENCE_LOGICAL_BYTES_PER_PAIR,
            "fp32_weight_staging": w_staged(union_far_fraction_independent) < REFERENCE_LOGICAL_BYTES_PER_PAIR,
        },
        "single_cta_64_kib_accumulator_review_cap_met": accum_bytes <= 64 * 1024,
        "note": (
            "64 KiB is a DESIGN REVIEW FILTER, not a measured RTX 5070 Ti hardware limit; "
            "FP32 accumulator footprint is logical, not predicted registers or shared memory. "
            "Selection correlation, L2 reuse, occupancy, launched CTA count, "
            "extra tensor-core work, and GPU latency are unmeasured."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=82000)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--tiles", type=int, default=32)
    parser.add_argument("--mode", choices=tuple(EXACT), default="balanced")
    parser.add_argument("--centres-per-cta", type=int, default=8)
    parser.add_argument("--query-row-groups", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(estimate(tokens=args.tokens, heads=args.heads,
                              tiles=args.tiles, mode=args.mode,
                              centres_per_cta=args.centres_per_cta,
                              query_row_groups=args.query_row_groups),
                     indent=2))


if __name__ == "__main__":
    main()
