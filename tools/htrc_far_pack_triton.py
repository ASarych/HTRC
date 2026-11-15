"""Research ONLY: ordered EXCL-to-packed-far-leaf-index GPU kernel.

Not imported by CPU tests or HTRC's production attention launcher.
The packed suffix beyond COUNTS is deliberately undefined and must be masked.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def pack_ordered_far_index(EXCL, PACKED, COUNTS,
                           NB: tl.constexpr, SPLITS: tl.constexpr,
                           LPS: tl.constexpr):
    """One CTA per original (tile, split); preserve increasing far leaf order."""
    tile = tl.program_id(0)  # flattened head * 32 + local_tile
    split = tl.program_id(1)
    r = tl.arange(0, LPS)
    leaf = split * LPS + r
    valid = leaf < NB
    excluded = tl.load(EXCL + tile * NB + leaf, mask=valid, other=1)
    keep = valid & (excluded == 0)
    indicator = keep.to(tl.int32)
    dest = tl.cumsum(indicator, 0) - 1
    # Keep-only scatter: [0,count) is overwritten on every replay;
    # suffix [count,LPS) is undefined and must NEVER be read.
    tl.store(PACKED + (tile * SPLITS + split) * LPS + dest, leaf,
             mask=keep)
    tl.store(COUNTS + tile * SPLITS + split, tl.sum(indicator, 0))
