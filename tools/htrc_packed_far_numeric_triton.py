"""Research ONLY: original-split ordered packed far QK/PV/online-softmax.

Not imported by production HTRC. Numerics are NOT assumed identical to the
baseline because packing changes BN64 dot/online-softmax reduction groups.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def packed_far_partial(Q, MU, G, LZ, VB, SG, PACKED, COUNTS,
                       PM, PL, PACC,
                       NB: tl.constexpr, SPLITS: tl.constexpr,
                       LPS: tl.constexpr, TILES: tl.constexpr,
                       T: tl.constexpr, D: tl.constexpr,
                       BN: tl.constexpr,
                       Q_DIRECT: tl.constexpr,
                       Q_SH: tl.constexpr, Q_ST: tl.constexpr,
                       Q_SR: tl.constexpr, Q_SD: tl.constexpr):
    """One CTA per original (head*tile,split), far PASS=1 ONLY.

    All valid packed original leaf indices remain in increasing order within
    the original split. Different 64-column group boundaries make this an
    experimental numerical hypothesis, not a drop-in equivalent kernel.
    """
    t = tl.program_id(0)
    sp = tl.program_id(1)
    r = tl.arange(0, T)
    d = tl.arange(0, D)
    if Q_DIRECT:
        head = t // TILES
        tile = t % TILES
        q = tl.load(Q + head*Q_SH + tile*Q_ST +
                    r[:, None]*Q_SR + d[None, :]*Q_SD).to(tl.float32)
    else:
        q = tl.load(Q + (t*T+r)[:,None]*D+d[None,:])
    mu = tl.load(MU + t*D+d)
    dq = q - mu[None,:]
    # Same input row INT8 Q quantization as materialized original PASS=1.
    sdq = tl.maximum(tl.max(tl.abs(dq), axis=1)*(1.0/127.0),1.0e-20)
    x = dq / sdq[:,None]
    dq8 = tl.where(x>=0,x+0.5,x-0.5).to(tl.int8)
    m_i = tl.full((T,),float('-inf'),tl.float32)
    l_i = tl.zeros((T,),tl.float32)
    acc = tl.zeros((T,D),tl.float32)
    pos = t*SPLITS+sp
    count = tl.load(COUNTS+pos)
    # Never read PACKED suffix; it is deliberately uninitialized.
    for offset in range(0,count,BN):
        slot = offset+tl.arange(0,BN)
        valid = slot<count
        leaf = tl.load(PACKED+pos*LPS+slot,mask=valid,other=0)
        base = t*NB+leaf
        lz = tl.load(LZ+base,mask=valid,other=float('-inf'))
        g = tl.load(G+base[:,None]*D+d[None,:],
                    mask=valid[:,None],other=0)
        sgc = tl.load(SG+base,mask=valid,other=0.0)
        sc = tl.dot(dq8,tl.trans(g)).to(tl.float32)*(
            sdq[:,None]*sgc[None,:])+lz[None,:]
        sc = tl.where(valid[None,:],sc,float('-inf'))
        m_new = tl.maximum(m_i,tl.max(sc,axis=1))
        m_s = tl.where(m_new == float('-inf'),0.0,m_new)
        p = tl.exp(sc-m_s[:,None])
        alpha = tl.exp(m_i-m_s)
        l_i = l_i*alpha + tl.sum(p,axis=1)
        vb = tl.load(VB+base[:,None]*D+d[None,:],
                     mask=valid[:,None],other=0.0)
        acc = acc*alpha[:,None] + tl.dot(p.to(tl.bfloat16),vb)
        m_i = m_new
    outbase = t*SPLITS+sp
    tl.store(PM+outbase*T+r,m_i)
    tl.store(PL+outbase*T+r,l_i)
    tl.store(PACC+(outbase*T+r)[:,None]*D+d[None,:],acc)
