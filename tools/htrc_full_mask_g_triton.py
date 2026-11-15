"""Research-only full Tile32 far+near clone, original BN64 groups, mask G only.

This is not imported by h3 or production. First compare MASK_G=0 (mirror)
against the unchanged PASS=0 kernel. A MASK_G=1 candidate is eligible only
when mirror and both distinct QKV banks yield raw bit-exact PM/PL/PACC.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def full_mask_g_partial(Q, MU, G, LZ, VB, EXCL, IDX, K, V, KS, SG,
                        PM, PL, PACC,
                        NB: tl.constexpr, KX: tl.constexpr,
                        LPS: tl.constexpr, EPS: tl.constexpr,
                        TILES: tl.constexpr, KEYS: tl.constexpr,
                        T: tl.constexpr, D: tl.constexpr,
                        BN: tl.constexpr, LEAF: tl.constexpr,
                        MASK_G: tl.constexpr,
                        Q_SH: tl.constexpr, Q_ST: tl.constexpr,
                        Q_SR: tl.constexpr, Q_SD: tl.constexpr,
                        V_DIRECT: tl.constexpr, V_SH: tl.constexpr,
                        V_SS: tl.constexpr, V_SD: tl.constexpr,
                        SCALE: tl.constexpr):
    """One CTA per original (head*tile,split), BOTH far and exact-near.

    Preserve every original BN64 group, far→near order and F32 partial.
    This intentionally does not change exact token selection or the merge.
    """
    t = tl.program_id(0)
    sp = tl.program_id(1)
    ns = tl.num_programs(1)
    head = t // TILES
    key_base = head * KEYS
    r = tl.arange(0, T)
    d = tl.arange(0, D)
    q = tl.load(Q + head * Q_SH + (t % TILES) * Q_ST +
                r[:, None] * Q_SR + d[None, :] * Q_SD).to(tl.float32)
    mu = tl.load(MU + t * D + d)
    dq = q - mu[None, :]
    # Original default DOT=1, I8=1, LATE_NEAR_Q=0.
    qb = q.to(tl.bfloat16)
    dqb = dq.to(tl.bfloat16)
    sdq = tl.maximum(tl.max(tl.abs(dq), axis=1) * (1.0 / 127.0), 1.0e-20)
    x = dq / sdq[:, None]
    dq8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)
    sq = tl.maximum(tl.max(tl.abs(q), axis=1) * (1.0 / 127.0), 1.0e-20)
    x = q / sq[:, None]
    q8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)

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
        # G-only candidate: VB loads are unchanged, and so are all PV dots.
        vb = tl.load(VB + (t * NB + c)[:, None] * D + d[None, :],
                     mask=cm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
        m_i = m_new

    # Original near path, same K8 quantization and exact token ordering.
    e_n = KX * LEAF
    e_lo = sp * EPS
    e_hi = tl.minimum(e_n, e_lo + EPS)
    for e0 in range(e_lo, e_hi, BN):
        j = e0 + tl.arange(0, BN)
        jm = j < e_hi
        leaf = tl.load(IDX + t * KX + j // LEAF, mask=jm, other=0)
        tok = leaf * LEAF + j % LEAF
        kk = tl.load(K + (key_base + tok)[:, None] * D + d[None, :],
                     mask=jm[:, None], other=0.0)
        if V_DIRECT:
            vv = tl.load(V + head * V_SH + tok[:, None] * V_SS +
                         d[None, :] * V_SD, mask=jm[:, None], other=0.0)
        else:
            vv = tl.load(V + (key_base + tok)[:, None] * D + d[None, :],
                         mask=jm[:, None], other=0.0)
        ksc = tl.load(KS + key_base + tok, mask=jm, other=0.0)
        sc = tl.dot(q8, tl.trans(kk)).to(tl.float32) * (
            sq[:, None] * ksc[None, :] * SCALE)
        sc = tl.where(jm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_s = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_s[:, None])
        alpha = tl.exp(m_i - m_s)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
        m_i = m_new

    base = t * ns + sp
    tl.store(PM + base * T + r, m_i)
    tl.store(PL + base * T + r, l_i)
    tl.store(PACC + (base * T + r)[:, None] * D + d[None, :], acc)
