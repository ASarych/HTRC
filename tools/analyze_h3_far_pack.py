"""CPU reference for split-preserving far-leaf index packing (no torch).

Every logical (head, tile, original far split) retains increasing leaf order.
The compacted index suffix is *undefined*: its explicit count is the only
validity bound. A compacted attention kernel is NOT implemented here.
"""
from __future__ import annotations

from analyze_h3_far_exclusion import aligned_split, summarize_exclusions


def reference_pack(exclusions, selected, *, splits=4, bn=64, expected_near=None):
    audit = summarize_exclusions(exclusions, selected, splits=splits, bn=bn,
                                 expected_near=expected_near)
    nb = audit["leaves_per_center"]
    lps = aligned_split(nb, splits, bn)
    indices = []
    total_blocks = total_indices = 0
    for head_masks in exclusions:
        head_rows = []
        for mask in head_masks:
            row = []
            for split in range(splits):
                lo, hi = split * lps, min(nb, (split + 1) * lps)
                far = [i for i in range(lo, hi) if mask[i] == 0]
                if len(far) > lps:
                    raise AssertionError("Packed far split exceeds capacity")
                row.append(far)
                total_indices += len(far)
                total_blocks += (len(far) + bn - 1) // bn
            head_rows.append(row)
        indices.append(head_rows)
    if (total_indices != audit["eligible_far_leaves_all_centers"] or
        total_blocks != audit["optimistic_packed_blocks_per_split_lower_bound"]):
        raise AssertionError("CPU packing disagrees with mask geometry")
    return {"indices": indices, "split_capacity": lps,
            "packed_blocks": total_blocks, "packed_indices": total_indices,
            "selection": audit}


def verify_pack_outputs(packed, counts, reference):
    """Check exact compacted prefixes; never read/require the uninitialized tail."""
    expected = reference["indices"]
    lps = reference["split_capacity"]
    if len(packed) != len(expected) or len(counts) != len(expected):
        raise AssertionError("Packed head count drift")
    total = blocks = 0
    for h, rows in enumerate(expected):
        if len(packed[h]) != len(rows) or len(counts[h]) != len(rows):
            raise AssertionError("Packed tile count drift")
        for t, splits in enumerate(rows):
            if len(packed[h][t]) != len(splits) or len(counts[h][t]) != len(splits):
                raise AssertionError("Packed far split count drift")
            for sp, expected_indices in enumerate(splits):
                n = counts[h][t][sp]
                if type(n) is not int or not (0 <= n <= lps):
                    raise AssertionError("Packed far count invalid")
                if n != len(expected_indices):
                    raise AssertionError(f"Packed count mismatch at h{h}/t{t}/sp{sp}")
                actual = packed[h][t][sp]
                if len(actual) != lps or actual[:n] != expected_indices:
                    raise AssertionError(f"Packed order/content mismatch at h{h}/t{t}/sp{sp}")
                total += n
                blocks += (n + reference["selection"]["far_bn"] - 1) // reference["selection"]["far_bn"]
    if total != reference["packed_indices"] or blocks != reference["packed_blocks"]:
        raise AssertionError("GPU index count/block geometry disagrees with reference")
    return {"valid_prefixes_bit_exact": True, "split_membership_and_order_exact": True,
            "padded_suffix_not_read": True, "packed_valid_indices": total,
            "packed_block_lower_bound": blocks}
