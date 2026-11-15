"""Research-only far PASS=1 clone with original BN64 groups and masked G/VB reads.

No production import. Zero masks compile to a dense research clone for an
independent bit-for-bit gate against the unchanged baseline _tile_kernel.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def far_masked_load_partial(Q, MU, G, LZ, VB, EXCL, SG, PM, PL, PACC,
                            NB: tl.constexpr, LPS: tl.constexpr,
                            TILES: tl.constexpr, T: tl.constexpr,
                            D: tl.constexpr, BN: tl.constexpr,
                            MASK_G: tl.constexpr, MASK_VB: tl.constexpr,
                            Q_DIRECT: tl.constexpr,
                            Q_SH: tl.constexpr, Q_ST: tl.constexpr,
                            Q_SR: tl.constexpr, Q_SD: tl.constexpr):
    """Original far BN64 online-softmax/PV order; only G/VB read masks differ."""
    t = tl.program_id(0)
    sp = tl.program_id(1)
    r = tl.arange(0, T)
    d = tl.arange(0, D)
    if Q_DIRECT:
        head = t // TILES
        tile = t % TILES
        q = tl.load(Q + head * Q_SH + tile * Q_ST +
                    r[:, None] * Q_SR + d[None, :] * Q_SD).to(tl.float32)
    else:
        q = tl.load(Q + (t * T + r)[:, None] * D + d[None, :])
    mu = tl.load(MU + t * D + d)
    dq = q - mu[None, :]
    # Same INT8 row quantization and far score formula as original PASS=1.
    sdq = tl.maximum(tl.max(tl.abs(dq), axis=1) * (1.0 / 127.0), 1.0e-20)
    x = dq / sdq[:, None]
    dq8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)
    m_i = tl.full((T,), float("-inf"), tl.float32)
    l_i = tl.zeros((T,), tl.float32)
    acc = tl.zeros((T, D), tl.float32)
    lo = sp * LPS
    hi = tl.minimum(NB, lo + LPS)
    for n0 in range(lo, hi, BN):
        c = n0 + tl.arange(0, BN)
        cm = c < hi
        lz = tl.load(LZ + t * NB + c, mask=cm, other=float("-inf"))
        ex = tl.load(EXCL + t * NB + c, mask=cm, other=1)
        eligible = (ex == 0) & cm
        if MASK_G:
            gmask = eligible
        else:
            gmask = cm
        g = tl.load(G + (t * NB + c)[:, None] * D + d[None, :],
                    mask=gmask[:, None], other=0.0)
        sgc = tl.load(SG + t * NB + c, mask=cm, other=0.0)
        sc = tl.dot(dq8, tl.trans(g)).to(tl.float32) * (
            sdq[:, None] * sgc[None, :]) + lz[None, :]
        sc = tl.where(eligible[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_s = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_s[:, None])
        alpha = tl.exp(m_i - m_s)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if MASK_VB:
            vmask = eligible
        else:
            vmask = cm
        vb = tl.load(VB + (t * NB + c)[:, None] * D + d[None, :],
                     mask=vmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
        m_i = m_new
    base = t * tl.num_programs(1) + sp
    tl.store(PM + base * T + r, m_i)
    tl.store(PL + base * T + r, l_i)
    tl.store(PACC + (base * T + r)[:, None] * D + d[None, :], acc)
