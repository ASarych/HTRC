"""Research-only gather traffic probes: NOT attention, no GPU runtime import elsewhere.

Every output is split-major with capacity LPS, and only masked valid entries
are defined. The unused suffix must never be accessed by a consumer.
"""
import triton
import triton.language as tl


@triton.jit
def far_dense_copy(G, VB, LZ, SG, EXCL, OG, OV, OL, OS, OE,
                   NB: tl.constexpr, SPLITS: tl.constexpr, LPS: tl.constexpr,
                   D: tl.constexpr, BN: tl.constexpr):
    """Control: coalesced reads of all original far leaves (near-selected included)."""
    tile = tl.program_id(0)
    sp = tl.program_id(1)
    d = tl.arange(0, D)
    for offset in range(0, LPS, BN):
        slot = offset + tl.arange(0, BN)
        leaf = sp * LPS + slot
        valid = leaf < NB
        base = (tile * NB + leaf)
        obase = ((tile * SPLITS + sp) * LPS + slot)
        gg = tl.load(G + base[:, None] * D + d[None, :], mask=valid[:, None], other=0)
        vv = tl.load(VB + base[:, None] * D + d[None, :], mask=valid[:, None], other=0)
        lz = tl.load(LZ + base, mask=valid, other=0)
        sg = tl.load(SG + base, mask=valid, other=0)
        ex = tl.load(EXCL + base, mask=valid, other=1)
        tl.store(OG + obase[:, None] * D + d[None, :], gg, mask=valid[:, None])
        tl.store(OV + obase[:, None] * D + d[None, :], vv, mask=valid[:, None])
        tl.store(OL + obase, lz, mask=valid)
        tl.store(OS + obase, sg, mask=valid)
        tl.store(OE + obase, ex, mask=valid)


@triton.jit
def far_packed_copy(G, VB, LZ, SG, EXCL, PACKED, COUNTS,
                    OG, OV, OL, OS, OE,
                    NB: tl.constexpr, SPLITS: tl.constexpr, LPS: tl.constexpr,
                    D: tl.constexpr, BN: tl.constexpr):
    """Candidate: indirect ordered leaf reads via a previously validated PACKED prefix."""
    tile = tl.program_id(0)
    sp = tl.program_id(1)
    d = tl.arange(0, D)
    pos = tile * SPLITS + sp
    count = tl.load(COUNTS + pos)
    # Count comes from a separately validated selection mask; avoid uninitialized
    # suffix reads even if the count is zero. The wrapper enforces 0<=count<=LPS.
    for offset in range(0, count, BN):
        slot = offset + tl.arange(0, BN)
        valid = slot < count
        leaf = tl.load(PACKED + pos * LPS + slot, mask=valid, other=0)
        base = (tile * NB + leaf)
        obase = (pos * LPS + slot)
        gg = tl.load(G + base[:, None] * D + d[None, :], mask=valid[:, None], other=0)
        vv = tl.load(VB + base[:, None] * D + d[None, :], mask=valid[:, None], other=0)
        lz = tl.load(LZ + base, mask=valid, other=0)
        sg = tl.load(SG + base, mask=valid, other=0)
        ex = tl.load(EXCL + base, mask=valid, other=1)
        tl.store(OG + obase[:, None] * D + d[None, :], gg, mask=valid[:, None])
        tl.store(OV + obase[:, None] * D + d[None, :], vv, mask=valid[:, None])
        tl.store(OL + obase, lz, mask=valid)
        tl.store(OS + obase, sg, mask=valid)
        tl.store(OE + obase, ex, mask=valid)
