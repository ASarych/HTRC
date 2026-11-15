"""CPU-only, fail-closed analysis of HTRC Tile far-leaf exclusion maps.

No torch dependency. The numbers measure mask geometry, NOT GPU speed,
DRAM transactions, tensor-core utilization or a validated compacted kernel.
"""
from __future__ import annotations

import hashlib


def aligned_split(n: int, splits: int, bn: int) -> int:
    """Match h3.htrc_flat_triton._split for the far-field loop."""
    if not all(type(x) is int and x > 0 for x in (n, splits, bn)):
        raise ValueError("Invalid far split parameters")
    per = (n + splits - 1) // splits
    return max(bn, ((per + bn - 1) // bn) * bn)


def _bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 15:
        return "1-15"
    if n <= 31:
        return "16-31"
    if n <= 47:
        return "32-47"
    if n <= 63:
        return "48-63"
    return "64"


def summarize_exclusions(exclusions, selected, *, splits: int = 4,
                         bn: int = 64, expected_near: int | None = None) -> dict:
    """Audit all [head][tile][leaf] masks against topk IDs and count far blocks.

    A fully excluded 64-leaf block might be skippable without compaction,
    but this does NOT implement or time the branch. Packed blocks per split
    are only an optimistic block-count lower bound, not a speed estimate.
    """
    if not exclusions or len(exclusions) != len(selected):
        raise ValueError("Missing heads or inconsistent EXCL/IDX head count")
    if not all(type(x) is int and x > 0 for x in (splits, bn)):
        raise ValueError("Invalid splits/block")
    heads = len(exclusions)
    tiles = len(exclusions[0])
    if not tiles or any(len(h) != tiles for h in exclusions) or any(
        len(h) != tiles for h in selected
    ):
        raise ValueError("Missing tiles or inconsistent per-head tile count")
    nb = len(exclusions[0][0])
    kx = len(selected[0][0])
    if nb < 1 or kx < 1 or kx > nb:
        raise ValueError("Invalid far/near leaf shape")
    if expected_near is not None and kx != expected_near:
        raise ValueError("Unexpected exact near leaf budget")
    lps = aligned_split(nb, splits, bn)
    hist = {k: 0 for k in ("0", "1-15", "16-31", "32-47", "48-63", "64")}
    blocks = 0
    empty_blocks = 0
    packed_lower_bound = 0
    eligible_leaves = 0
    padded_dot_lanes = 0
    per_head = []
    mask_bytes = bytearray()
    for h in range(heads):
        head_blocks = head_empty = head_packed = head_eligible = 0
        for t in range(tiles):
            mask = exclusions[h][t]
            ids = selected[h][t]
            if len(mask) != nb or len(ids) != kx:
                raise ValueError("EXCL/IDX row shape drift")
            if any(type(x) is not int or x not in (0, 1) for x in mask):
                raise ValueError("EXCL must be a binary integer tensor")
            if any(type(x) is not int or x < 0 or x >= nb for x in ids):
                raise ValueError("IDX includes an invalid leaf index")
            if len(set(ids)) != kx:
                raise ValueError("Topk IDX must be distinct within each tile")
            near = set(ids)
            if {j for j, v in enumerate(mask) if v == 1} != near:
                raise ValueError("EXCL must exactly equal selected IDX leaves")
            mask_bytes.extend(mask)
            far = nb - kx
            head_eligible += far
            for sp in range(splits):
                lo = sp * lps
                hi = min(nb, lo + lps)
                if lo >= hi:
                    continue
                split_far = sum(1 for j in range(lo, hi) if mask[j] == 0)
                head_packed += (split_far + bn - 1) // bn
                for start in range(lo, hi, bn):
                    end = min(hi, start + bn)
                    count = sum(1 for j in range(start, end) if mask[j] == 0)
                    hist[_bucket(count)] += 1
                    head_blocks += 1
                    if count == 0:
                        head_empty += 1
        blocks += head_blocks
        empty_blocks += head_empty
        packed_lower_bound += head_packed
        eligible_leaves += head_eligible
        padded_dot_lanes += head_blocks * bn
        per_head.append({"head": h, "current_blocks": head_blocks,
                         "fully_excluded_blocks": head_empty,
                         "optimistic_packed_blocks_per_split": head_packed,
                         "eligible_far_leaves": head_eligible})
    if sum(hist.values()) != blocks:
        raise AssertionError("Block histogram does not reconcile")
    return {
        "heads": heads,
        "tiles_per_head": tiles,
        "verified_tile_centers": heads * tiles,
        "leaves_per_center": nb,
        "selected_near_leaves_per_center": kx,
        "eligible_far_leaves_per_center": nb - kx,
        "far_splits": splits,
        "far_bn": bn,
        "aligned_leaves_per_split": lps,
        "current_far_dot_blocks": blocks,
        "fully_excluded_blocks": empty_blocks,
        "nonempty_far_dot_blocks": blocks - empty_blocks,
        "optimistic_packed_blocks_per_split_lower_bound": packed_lower_bound,
        "theoretical_block_reduction_if_packed": blocks - packed_lower_bound,
        "eligible_far_leaves_all_centers": eligible_leaves,
        "current_dot_lanes_including_padding": padded_dot_lanes,
        "excluded_fraction_of_real_far_leaves": kx / nb,
        "block_far_eligible_histogram": hist,
        "per_head": per_head,
        "selection_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "measurement_semantics": (
            "Selected EXCL/IDX geometry read from actual prepared Graph tensors; "
            "not measured DRAM traffic, number of issued tensor-core instructions, "
            "an executable compaction, or an attention speedup"
        ),
    }
