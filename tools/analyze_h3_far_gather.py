"""Torch-free reference geometry and strict CPU semantic cases for far gathers."""
from __future__ import annotations

from analyze_h3_far_pack import reference_pack


def gather_geometry(reference: dict) -> dict:
    s = reference["selection"]
    centers = s["verified_tile_centers"]
    nb, kx = s["leaves_per_center"], s["selected_near_leaves_per_center"]
    splits, lps = s["far_splits"], s["aligned_leaves_per_split"]
    if splits * lps < nb:
        raise AssertionError("Split coverage does not reach all leaf positions")
    dense = centers * nb
    packed = reference["packed_indices"]
    if packed != centers * (nb - kx):
        raise AssertionError("Packed prefix count does not match eligible far leaves")
    if dense < packed:
        raise AssertionError("Far leaf count exceeded dense coverage")
    return {
        "tile_centers": centers,
        "dense_valid_source_leaves": dense,
        "packed_valid_source_leaves": packed,
        "source_leaf_load_ratio_packed_over_dense": packed / dense,
        "packed_far_dot_block_lower_bound": reference["packed_blocks"],
        "existing_far_dot_blocks": s["current_far_dot_blocks"],
        "note": "Leaf-load counts are not measured DRAM transactions or attention FLOPs",
    }


def gather_scalar_reference(values, reference: dict, *, dense: bool):
    """Return nested lists of valid prefixes; never inspect any packed suffix.

    values: [head][tile][leaf] arbitrary scalar values. This is an
    algorithmic oracle only, not a GPU/attention implementation.
    """
    out = []
    s = reference["selection"]
    nb, splits, lps = s["leaves_per_center"], s["far_splits"], s["aligned_leaves_per_split"]
    if len(values) != len(reference["indices"]):
        raise ValueError("Source head count drift")
    for h, tiles in enumerate(reference["indices"]):
        if len(values[h]) != len(tiles):
            raise ValueError("Source tile count drift")
        ho = []
        for t, packed_splits in enumerate(tiles):
            if len(values[h][t]) != nb:
                raise ValueError("Source leaf shape drift")
            ho.append([])
            for sp in range(splits):
                ids = list(range(sp*lps, min(nb, (sp+1)*lps))) if dense else packed_splits[sp]
                ho[-1].append([values[h][t][i] for i in ids])
        out.append(ho)
    return out
