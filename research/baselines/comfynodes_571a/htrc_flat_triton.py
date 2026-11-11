"""Triton kernels of the flat operator (h3/htrc_flat.py, backend="triton").

_stats_kernel: per (BT tile centres x one leaf) the exact leaf cumulants at the centres
    lz = logsumexp_s(scale mu.k_s), g = scale sum_s w_s k_s, vbar = sum_s w_s v_s
_rowmax_kernel: SEL[t, l] = lz + max over (a stride of) the tile rows of (q - mu).g_l
_tile_kernel: program (tile, split): online softmax over
    far field  logit = lz_l + (q - mu).g_l, value vbar_l  (leaves not excluded)
    near field logit = scale q.k_t, value v_t             (tokens of the tile's exact leaves)
    partial (max, sum, acc) per split, merged on the host side (torch).
DOT=1 runs the far/near dots with bf16 operands (fp32 accumulate): needs bf16 g/vbar and bf16
copies of k/v for the near field.
"""
from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl


_DLL_STATE = {"done": False, "tries": 0}


def _dll():
    """Windows Triton DLL search path, once.  This used to scan ALL of sys.modules (ComfyUI has
    thousands) on every kernel entry, ~4-5 calls per head: ~1 ms each, i.e. host time comparable to the
    GPU time of a head (sampler profile: >50% of the main thread's samples sat in this loop)."""
    if _DLL_STATE["done"]:
        return
    _DLL_STATE["tries"] += 1
    for name, mod in list(sys.modules.items()):
        if name.endswith("rrc3_kernels") and hasattr(mod, "_ensure_windows_triton_dll_search"):
            mod._ensure_windows_triton_dll_search()
            _DLL_STATE["done"] = True
            return
    if _DLL_STATE["tries"] >= 3:          # provider module never showed up: stop scanning
        _DLL_STATE["done"] = True


@triton.jit
def _stats_kernel(MU, K, V, G, LZ, VB, SG, SV, tc, nb, scale,
                  D: tl.constexpr, BT: tl.constexpr, LEAF: tl.constexpr, BL: tl.constexpr,
                  PREC: tl.constexpr, FP8: tl.constexpr = 0, I8: tl.constexpr = 0):
    pt = tl.program_id(0)
    pl = tl.program_id(1)
    t = pt * BT + tl.arange(0, BT)
    tm = t < tc
    d = tl.arange(0, D)
    s = tl.arange(0, LEAF)
    mu = tl.load(MU + t[:, None] * D + d[None, :], mask=tm[:, None], other=0.0)
    for j in tl.static_range(0, BL):
        l = pl * BL + j
        if l < nb:
            kj = tl.load(K + (l * LEAF + s)[:, None] * D + d[None, :]).to(tl.float32)
            vj = tl.load(V + (l * LEAF + s)[:, None] * D + d[None, :]).to(tl.float32)
            a = tl.dot(mu, tl.trans(kj), input_precision=PREC) * scale      # [BT, LEAF]
            m = tl.max(a, axis=1)
            w = tl.exp(a - m[:, None])
            z = tl.sum(w, axis=1)
            w = w / z[:, None]
            g = tl.dot(w, kj, input_precision=PREC) * scale                   # [BT, D]
            vb = tl.dot(w, vj, input_precision=PREC)
            off = (t * nb + l)[:, None] * D + d[None, :]
            if FP8:
                # row-scaled e4m3: stored = x / s, s = max|x| / 448 (state tolerates 4-8 bit, STATE round 42/44)
                sg = tl.maximum(tl.max(tl.abs(g), axis=1) * (1.0 / 448.0), 1.0e-30)
                sv = tl.maximum(tl.max(tl.abs(vb), axis=1) * (1.0 / 448.0), 1.0e-30)
                tl.store(G + off, (g / sg[:, None]).to(G.dtype.element_ty), mask=tm[:, None])
                tl.store(VB + off, (vb / sv[:, None]).to(VB.dtype.element_ty), mask=tm[:, None])
                tl.store(SG + t * nb + l, sg, mask=tm)
                tl.store(SV + t * nb + l, sv, mask=tm)
            elif I8:
                # int8 far-field slope g (row scale sg); vbar stays bf16 (it is the PV operand)
                am = tl.maximum(tl.max(tl.abs(g), axis=1), 1.0e-30)
                sg = am * (1.0 / 127.0)
                x = g * (127.0 / am)[:, None]
                x = tl.where(x >= 0, x + 0.5, x - 0.5)
                tl.store(G + off, x.to(tl.int8), mask=tm[:, None])
                tl.store(VB + off, vb.to(VB.dtype.element_ty), mask=tm[:, None])
                tl.store(SG + t * nb + l, sg, mask=tm)
            else:
                tl.store(G + off, g.to(G.dtype.element_ty), mask=tm[:, None])
                tl.store(VB + off, vb.to(VB.dtype.element_ty), mask=tm[:, None])
            tl.store(LZ + t * nb + l, m + tl.log(z), mask=tm)


@triton.jit
def _tile_kernel(Q, MU, G, LZ, VB, EXCL, IDX, K, V, PM, PL, PACC, SG, SV, TK, TV, KS, nb, kx, lps, eps, nt, scale,
                 T: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, LEAF: tl.constexpr,
                 PREC: tl.constexpr, DOT: tl.constexpr, GR: tl.constexpr = 1, FP8: tl.constexpr = 0,
                 TAIL: tl.constexpr = 0, RB: tl.constexpr = 1, I8: tl.constexpr = 0):
    """Split-K over leaves: program (tile t, split sp) covers far leaves [sp*lps, +lps) and
    exact tokens [sp*eps, +eps); writes partial (max, sum, acc) merged on the host."""
    t = tl.program_id(0)
    gt = t // GR          # far-field state / near leaves are shared by GR consecutive tiles
    sp = tl.program_id(1)
    ns = tl.num_programs(1)
    RT: tl.constexpr = T // RB          # rows per program: RB programs share one tile (same centre/near set), more warps resident
    r = tl.program_id(2) * RT + tl.arange(0, RT)
    d = tl.arange(0, D)
    q = tl.load(Q + (t * T + r)[:, None] * D + d[None, :])
    mu = tl.load(MU + gt * D + d)
    dq = q - mu[None, :]
    if DOT == 1:
        qb = q.to(tl.bfloat16)
        dqb = dq.to(tl.bfloat16)
        if FP8:
            dqh = dq.to(tl.float16)       # e4m3 -> fp16 is exact and a single packed cvt (-> bf16 went via fp32)
        if I8:
            # int8 x int8 -> int32 tensor-core dots for the QK parts; PV stays bf16
            sdq = tl.maximum(tl.max(tl.abs(dq), axis=1) * (1.0 / 127.0), 1.0e-20)
            x = dq / sdq[:, None]
            dq8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)
            sq = tl.maximum(tl.max(tl.abs(q), axis=1) * (1.0 / 127.0), 1.0e-20)
            x = q / sq[:, None]
            q8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)
    m_i = tl.full((RT,), float("-inf"), tl.float32)
    l_i = tl.zeros((RT,), tl.float32)
    acc = tl.zeros((RT, D), tl.float32)
    lo = sp * lps
    hi = tl.minimum(nb, lo + lps)
    for n0 in range(lo, hi, BN):
        c = n0 + tl.arange(0, BN)
        cm = c < hi
        lz = tl.load(LZ + gt * nb + c, mask=cm, other=float("-inf"))
        ex = tl.load(EXCL + gt * nb + c, mask=cm, other=1)
        if DOT == 1:
            g = tl.load(G + (gt * nb + c)[:, None] * D + d[None, :], mask=cm[:, None], other=0.0)
            if I8:
                sgc = tl.load(SG + gt * nb + c, mask=cm, other=0.0)
                sc = tl.dot(dq8, tl.trans(g)).to(tl.float32) * (sdq[:, None] * sgc[None, :]) + lz[None, :]
            elif FP8:
                g = g.to(tl.float16)
                sgc = tl.load(SG + gt * nb + c, mask=cm, other=0.0)
                sc = tl.dot(dqh, tl.trans(g)) * sgc[None, :] + lz[None, :]
            else:
                sc = tl.dot(dqb, tl.trans(g)) + lz[None, :]
        else:
            g = tl.load(G + (gt * nb + c)[:, None] * D + d[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            sc = tl.dot(dq, tl.trans(g), input_precision=PREC) + lz[None, :]
        sc = tl.where(((ex == 0) & cm)[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_s = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_s[:, None])
        alpha = tl.exp(m_i - m_s)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vb = tl.load(VB + (gt * nb + c)[:, None] * D + d[None, :], mask=cm[:, None], other=0.0)
        if DOT == 1:
            if FP8:
                svc = tl.load(SV + gt * nb + c, mask=cm, other=0.0)
                acc = acc * alpha[:, None] + tl.dot((p * svc[None, :]).to(tl.float16), vb.to(tl.float16))
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, vb.to(tl.float32), input_precision=PREC)
        m_i = m_new
    e_n = kx * LEAF
    e_lo = sp * eps
    e_hi = tl.minimum(e_n, e_lo + eps)
    for e0 in range(e_lo, e_hi, BN):
        j = e0 + tl.arange(0, BN)
        jm = j < e_hi
        leaf = tl.load(IDX + gt * kx + j // LEAF, mask=jm, other=0)
        tok = leaf * LEAF + j % LEAF
        kk = tl.load(K + tok[:, None] * D + d[None, :], mask=jm[:, None], other=0.0)
        vv = tl.load(V + tok[:, None] * D + d[None, :], mask=jm[:, None], other=0.0)
        if DOT == 1:
            if I8:
                ksc = tl.load(KS + tok, mask=jm, other=0.0)
                sc = tl.dot(q8, tl.trans(kk)).to(tl.float32) * (sq[:, None] * ksc[None, :] * scale)
            else:
                sc = tl.dot(qb, tl.trans(kk)) * scale
        else:
            sc = tl.dot(q, tl.trans(kk.to(tl.float32)), input_precision=PREC) * scale
        sc = tl.where(jm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_s = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_s[:, None])
        alpha = tl.exp(m_i - m_s)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if DOT == 1:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, vv.to(tl.float32), input_precision=PREC)
        m_i = m_new
    if TAIL:
        # keys past the last full leaf (< 16, exact for every query): handled once, by the last split
        jt = tl.arange(0, 16)
        jtm = (jt < nt) & (sp == ns - 1)
        kt = tl.load(TK + jt[:, None] * D + d[None, :], mask=jtm[:, None], other=0.0)
        vt = tl.load(TV + jt[:, None] * D + d[None, :], mask=jtm[:, None], other=0.0)
        if DOT == 1:
            sc = tl.dot(qb, tl.trans(kt)) * scale
        else:
            sc = tl.dot(q, tl.trans(kt.to(tl.float32)), input_precision=PREC) * scale
        sc = tl.where(jtm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_s = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_s[:, None])
        alpha = tl.exp(m_i - m_s)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if DOT == 1:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
        else:
            acc = acc * alpha[:, None] + tl.dot(p, vt.to(tl.float32), input_precision=PREC)
        m_i = m_new
    base = t * ns + sp
    tl.store(PM + base * T + r, m_i)
    tl.store(PL + base * T + r, l_i)
    tl.store(PACC + (base * T + r)[:, None] * D + d[None, :], acc)


@triton.jit
def _rowmax_kernel(Q, MU, G, LZ, SEL, SG, nb, lps,
                   T: tl.constexpr, TS: tl.constexpr, STRIDE: tl.constexpr, D: tl.constexpr,
                   BN: tl.constexpr, PREC: tl.constexpr, DOT: tl.constexpr, FP8: tl.constexpr = 0, I8: tl.constexpr = 0):
    """SEL[t, l] = lz + max over TS rows (every STRIDE-th) of (q - mu).g_l; split over leaves."""
    t = tl.program_id(0)
    sp = tl.program_id(1)
    r = tl.arange(0, TS) * STRIDE
    d = tl.arange(0, D)
    q = tl.load(Q + (t * T + r)[:, None] * D + d[None, :])
    mu = tl.load(MU + t * D + d)
    dq = q - mu[None, :]
    if DOT == 1:
        dqb = dq.to(tl.bfloat16)
        if FP8:
            dqh = dq.to(tl.float16)
        if I8:
            sdq = tl.maximum(tl.max(tl.abs(dq), axis=1) * (1.0 / 127.0), 1.0e-20)
            x = dq / sdq[:, None]
            dq8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)
    lo = sp * lps
    hi = tl.minimum(nb, lo + lps)
    for n0 in range(lo, hi, BN):
        c = n0 + tl.arange(0, BN)
        cm = c < hi
        lz = tl.load(LZ + t * nb + c, mask=cm, other=float("-inf"))
        if DOT == 1:
            g = tl.load(G + (t * nb + c)[:, None] * D + d[None, :], mask=cm[:, None], other=0.0)
            if I8:
                sgc = tl.load(SG + t * nb + c, mask=cm, other=0.0)
                sc = tl.dot(dq8, tl.trans(g)).to(tl.float32) * (sdq[:, None] * sgc[None, :])
            elif FP8:
                g = g.to(tl.float16)
                sgc = tl.load(SG + t * nb + c, mask=cm, other=0.0)
                sc = tl.dot(dqh, tl.trans(g)) * sgc[None, :]
            else:
                sc = tl.dot(dqb, tl.trans(g))
        else:
            g = tl.load(G + (t * nb + c)[:, None] * D + d[None, :], mask=cm[:, None], other=0.0).to(tl.float32)
            sc = tl.dot(dq, tl.trans(g), input_precision=PREC)
        tl.store(SEL + t * nb + c, tl.max(sc, axis=0) + lz, mask=cm)


@triton.jit
def _merge_kernel(PM, PL, PACC, OUT, NS: tl.constexpr, T: tl.constexpr, D: tl.constexpr):
    """One program per tile: merge NS split partials (max, sum, acc) -> normalised output rows."""
    t = tl.program_id(0)
    r = tl.arange(0, T)
    d = tl.arange(0, D)
    mm = tl.full((T,), float("-inf"), tl.float32)
    for si in tl.static_range(0, NS):
        mm = tl.maximum(mm, tl.load(PM + (t * NS + si) * T + r))
    mm = tl.where(mm == float("-inf"), 0.0, mm)
    den = tl.zeros((T,), tl.float32)
    num = tl.zeros((T, D), tl.float32)
    for si in tl.static_range(0, NS):
        w = tl.exp(tl.load(PM + (t * NS + si) * T + r) - mm)        # empty split: m=-inf -> 0
        den += tl.load(PL + (t * NS + si) * T + r) * w
        num += tl.load(PACC + ((t * NS + si) * T + r)[:, None] * D + d[None, :]) * w[:, None]
    tl.store(OUT + (t * T + r)[:, None] * D + d[None, :], num / tl.maximum(den, 1.0e-30)[:, None])


@triton.jit
def _dequant_kernel(X8, S, OUT, rows, stride_x, stride_s, D: tl.constexpr, BR: tl.constexpr):
    pr = tl.program_id(0)
    r = pr * BR + tl.arange(0, BR)
    m = r < rows
    d = tl.arange(0, D)
    x = tl.load(X8 + r[:, None] * stride_x + d[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    sc = tl.load(S + r * stride_s, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r[:, None] * D + d[None, :], (x * sc[:, None]).to(OUT.dtype.element_ty), mask=m[:, None])


def dequant_rows(x8, scale, out, br=64):
    """out[r, :] = x8[r, :] * scale[r] (one pass, fp8 -> bf16); x8 [rows, D] and scale [rows] may be
    strided head views of the per-token cache, out [rows, D] contiguous."""
    _dll()
    rows, d = int(x8.shape[0]), int(x8.shape[1])
    _dequant_kernel[(triton.cdiv(rows, br),)](x8, scale, out, rows, int(x8.stride(0)), int(scale.stride(0)),
                                              D=d, BR=br, num_warps=4)
    return out


def _split(n, ns, bn):
    per = -(-n // ns)
    return max(bn, -(-per // bn) * bn)


def _buf(ws, name, shape, dtype, device):
    """Workspace buffer reused across calls (ws dict), sliced to the requested leading size.
    Under the cudaMallocAsync allocator every fresh torch.empty of tens of MiB is returned to the
    driver on free, which made a head ~10x slower than with the caching allocator."""
    if ws is None:
        return torch.empty(shape, device=device, dtype=dtype)
    b = ws.get(name)
    if b is None or b.dtype != dtype or b.shape[1:] != tuple(shape[1:]) or b.shape[0] < shape[0]:
        b = torch.empty(shape, device=device, dtype=dtype)
        ws[name] = b
    return b[:shape[0]]


def rowmax_select(q_c, mu_c, g, lz, prec="tf32", bn=32, warps=4, splits=8, stride=1, dot=0, ws=None, sg=None):
    _dll()
    tc, tile, d = (int(x) for x in q_c.shape)
    nb = int(lz.shape[1])
    sel = _buf(ws, "sel", (tc, nb), torch.float32, q_c.device)
    lps = _split(nb, splits, bn)
    _rowmax_kernel[(tc, triton.cdiv(nb, lps))](
        q_c.contiguous(), mu_c.contiguous(), g, lz, sel, sg if sg is not None else lz, nb, lps,
        T=tile, TS=tile // stride, STRIDE=stride, D=d, BN=bn, PREC=prec, DOT=dot,
        FP8=1 if (sg is not None and g.dtype != torch.int8) else 0, I8=1 if g.dtype == torch.int8 else 0,
        num_warps=warps, num_stages=1)
    return sel


def leaf_stats(mu, kf, vf, nb, scale, leaf=16, prec="tf32", bt=16, bl=4, warps=4, store=torch.float32, ws=None):
    """mu [tc,D] fp32, kf/vf [nb*leaf, D] -> g, lz, vbar, sg, sv: g/vbar [tc,nb,D] (dtype store),
    lz [tc,nb] fp32; store float8_e4m3fn adds row scales sg/sv [tc,nb] fp32 (else None)."""
    _dll()
    tc, d = int(mu.shape[0]), int(mu.shape[1])
    g = _buf(ws, "g", (tc, nb, d), store, mu.device)   # bf16 halves the traffic
    i8 = store == torch.int8
    vb = _buf(ws, "vb", (tc, nb, d), torch.bfloat16 if i8 else store, mu.device)
    lz = _buf(ws, "lz", (tc, nb), torch.float32, mu.device)
    grid = (triton.cdiv(tc, bt), triton.cdiv(nb, bl))
    fp8 = store == torch.float8_e4m3fn
    sg = _buf(ws, "sg", (tc, nb), torch.float32, mu.device) if (fp8 or i8) else None
    sv = _buf(ws, "sv", (tc, nb), torch.float32, mu.device) if fp8 else None
    _stats_kernel[grid](mu.contiguous(), kf, vf, g, lz, vb, sg if sg is not None else lz, sv if fp8 else lz,
                        tc, nb, float(scale), D=d, BT=bt, LEAF=leaf, BL=bl, PREC=prec,
                        FP8=1 if fp8 else 0, I8=1 if i8 else 0, num_warps=warps)
    return g, lz, vb, sg, sv


def tile_attention(q_c, mu_c, g, lz, vb, excl, idx, k_ex, v_ex, scale, leaf=16, prec="tf32",
                   bn=32, warps=8, stages=1, splits=8, dot=0, tail=None, ws=None, group=1, sg=None, sv=None, out=None, rb=1, ks=None):
    """q_c [tc,T,D], mu_c [tc,D], g/vb [tc,nb,D], lz [tc,nb], excl int8 [tc,nb], idx [tc,K];
    k_ex/v_ex: key/value tokens for the near field (fp32, or bf16 when dot=1).
    tail: optional (tk [n,D], tv [n,D]) fp32 keys beyond the last full leaf, attended exactly by
    every query as one extra softmax partial (host-side, graph-safe)."""
    _dll()
    tc, tile, d = (int(x) for x in q_c.shape)       # tc tiles; state arrays hold tc // group rows
    nb = int(lz.shape[1])
    kx = int(idx.shape[1]) if idx is not None else 0
    if idx is None:
        idx = torch.zeros((int(lz.shape[0]), 1), device=q_c.device, dtype=torch.int32)
    ns = max(1, int(splits))
    lps = _split(nb, ns, bn)
    eps = _split(kx * leaf, ns, bn)
    pm = _buf(ws, "pm", (tc, ns, tile), torch.float32, q_c.device)
    pl = _buf(ws, "pl", (tc, ns, tile), torch.float32, q_c.device)
    pacc = _buf(ws, "pacc", (tc, ns, tile, d), torch.float32, q_c.device)
    ntail = int(tail[0].shape[0]) if tail is not None else 0
    use_tail_kernel = 0 < ntail <= 16
    tk = tv = None
    if use_tail_kernel:
        tdt = torch.bfloat16 if dot == 1 else torch.float32
        tk, tv = tail[0].to(tdt).contiguous(), tail[1].to(tdt).contiguous()
    _tile_kernel[(tc, ns, int(rb))](q_c.contiguous(), mu_c.contiguous(), g, lz, vb, excl,
                           idx.to(torch.int32).contiguous(), k_ex, v_ex, pm, pl, pacc,
                           sg if sg is not None else lz, sv if sv is not None else lz,
                           tk if use_tail_kernel else lz, tv if use_tail_kernel else lz,
                           ks if ks is not None else lz, nb, kx, lps, eps, ntail if use_tail_kernel else 0,
                           float(scale), T=tile, D=d, BN=bn, LEAF=leaf, PREC=prec, DOT=dot, GR=int(group),
                           FP8=1 if (sg is not None and g.dtype != torch.int8) else 0, TAIL=1 if use_tail_kernel else 0, RB=int(rb),
                           I8=1 if g.dtype == torch.int8 else 0,
                           num_warps=warps, num_stages=stages)
    if ntail > 16:                                            # large tails: host-side partial (rare)
        tk, tv = tail
        lg = torch.einsum("ctd,nd->ctn", q_c.float(), tk.float()) * float(scale)
        tm = lg.amax(-1)
        te = torch.exp(lg - tm[..., None])
        pm = torch.cat([pm, tm[:, None]], 1)
        pl = torch.cat([pl, te.sum(-1)[:, None]], 1)
        pacc = torch.cat([pacc, torch.einsum("ctn,nd->ctd", te, tv.float())[:, None]], 1)
    res = out if out is not None else torch.empty((tc, tile, d), device=q_c.device, dtype=torch.float32)
    _merge_kernel[(tc,)](pm, pl, pacc, res, NS=int(pm.shape[1]), T=tile, D=d, num_warps=4)
    return res
