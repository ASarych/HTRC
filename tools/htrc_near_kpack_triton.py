"""Strictly research-only K8+scale near packing on pinned 4x32 Tile.

Original Tile arithmetic is copied verbatim from HTRC. Exactly four
source-token transformations are authorized; the CPU contract reconstructs
this code from the canonical original and rejects any other drift.

For one 64-row query centre, pack 512 chosen K8 leaves + FP32 scales
in EXACT original unsorted IDX order. Keep original BF16 V as a native
strided tensor and original IDX for its address. No V packing or extra
near softmax permutation. Pseudo-head == (real head * 32 + centre).
Do NOT import this module from production or the default Graph path.
"""
import triton
import triton.language as tl
from h3.htrc_flat_triton import _fused_vbar_one_leaf


@triton.jit
def pack_near_key_and_scale(K, KS, IDX, PACKED_K, PACKED_KS,
                            SRC_KEYS: tl.constexpr, KX: tl.constexpr,
                            NEAR_TOKENS: tl.constexpr,
                            CENTRES_PER_HEAD: tl.constexpr,
                            D: tl.constexpr, BLOCK: tl.constexpr):
    centre = tl.program_id(0)
    block = tl.program_id(1)
    j = block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    active = j < NEAR_TOKENS
    leaf = tl.load(IDX + centre * KX + j // 16,
                   mask=active, other=0)
    source_token = leaf * 16 + j % 16
    real_head = centre // CENTRES_PER_HEAD
    source_row = real_head * SRC_KEYS + source_token
    destination_row = centre * NEAR_TOKENS + j
    k = tl.load(K + source_row[:, None] * D + d[None, :],
                mask=active[:, None], other=0)
    scale = tl.load(KS + source_row, mask=active, other=0)
    tl.store(PACKED_K + destination_row[:, None] * D + d[None, :],
             k, mask=active[:, None])
    tl.store(PACKED_KS + destination_row, scale, mask=active)


# Below is the COMPLETE original _tile_kernel with ONLY near K/KS address
# substitution, head-to-real-head V base, and research function rename.
# Keep all far groups, original native V, 4 splits and FP32 accumulation.

@triton.jit
def near_kpacked_tile_kernel(Q, MU, G, LZ, VB, EXCL, IDX, K, V, PM, PL, PACC, SG, SV, TK, TV, KS, nb, kx, lps, eps, nt, scale,
                 T: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, BN_NEAR: tl.constexpr, LEAF: tl.constexpr,
                 PREC: tl.constexpr, DOT: tl.constexpr, GR: tl.constexpr = 1, FP8: tl.constexpr = 0,
                 TAIL: tl.constexpr = 0, RB: tl.constexpr = 1, I8: tl.constexpr = 0,
                 PASS: tl.constexpr = 0, LATE_NEAR_Q: tl.constexpr = 0,
                 FAR_PREFETCH_LZ: tl.constexpr = 0,
                 FAR_PV_ACC: tl.constexpr = 0, NEAR_PV_ACC: tl.constexpr = 0,
                 BATCH_TILES_PER_HEAD: tl.constexpr = 0,
                 BATCH_KEY_ROWS: tl.constexpr = 0,
                 Q_DIRECT: tl.constexpr = 0, Q_SH: tl.constexpr = 0,
                 Q_ST: tl.constexpr = 0, Q_SR: tl.constexpr = 0,
                 Q_SD: tl.constexpr = 0, V_DIRECT: tl.constexpr = 0,
                 V_SH: tl.constexpr = 0, V_SS: tl.constexpr = 0,
                 V_SD: tl.constexpr = 0,
                 VB_SPLIT_STREAM: tl.constexpr = 0,
                 VB_START: tl.constexpr = 0,
                 VB_SPAN: tl.constexpr = 0,
                 SPLIT_ID: tl.constexpr = -1,
                 SPLIT_COUNT: tl.constexpr = 0,
                 FUSED_VB: tl.constexpr = 0, KBF = None,
                 KB_SH: tl.constexpr = 0,
                 KB_SS: tl.constexpr = 0,
                 KB_SD: tl.constexpr = 0, REAL_TILES_PER_HEAD: tl.constexpr = 32):
    """Split-K over leaves: program (tile t, split sp) covers far leaves [sp*lps, +lps) and
    exact tokens [sp*eps, +eps); writes partial (max, sum, acc) merged on the host."""
    t = tl.program_id(0)
    # New opt-in grid: t = head * tiles_per_head + local_tile.
    # Per-head K/V/KS pointers use local key indices; Q/G/states/IDX/
    # split output are flattened on the tile dimension. Original
    # specialization compiles away this branch at BATCH_TILES_PER_HEAD=0.
    if BATCH_TILES_PER_HEAD > 0:
        head = t // BATCH_TILES_PER_HEAD
        key_base = head * BATCH_KEY_ROWS
    else:
        key_base = 0
    gt = t // GR          # far-field state / near leaves are shared by GR consecutive tiles
    if VB_SPLIT_STREAM:
        sp = SPLIT_ID
        ns = SPLIT_COUNT
    else:
        sp = tl.program_id(1)
        ns = tl.num_programs(1)
    RT: tl.constexpr = T // RB          # rows per program: RB programs share one tile (same centre/near set), more warps resident
    r = tl.program_id(2) * RT + tl.arange(0, RT)
    d = tl.arange(0, D)
    if Q_DIRECT:
        qhead = t // BATCH_TILES_PER_HEAD
        qtile = t % BATCH_TILES_PER_HEAD
        q = tl.load(Q + qhead*Q_SH + qtile*Q_ST +
                    r[:, None]*Q_SR+d[None,:]*Q_SD).to(tl.float32)
    else:
        q = tl.load(Q + (t * T + r)[:, None] * D + d[None, :])
    mu = tl.load(MU + gt * D + d)
    dq = q - mu[None, :]
    if DOT == 1:
        if not LATE_NEAR_Q:
            qb = q.to(tl.bfloat16)
        dqb = dq.to(tl.bfloat16)
        if FP8:
            dqh = dq.to(tl.float16)       # e4m3 -> fp16 is exact and a single packed cvt (-> bf16 went via fp32)
        if I8:
            # int8 x int8 -> int32 tensor-core dots for the QK parts; PV stays bf16
            sdq = tl.maximum(tl.max(tl.abs(dq), axis=1) * (1.0 / 127.0), 1.0e-20)
            x = dq / sdq[:, None]
            dq8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)
            if not LATE_NEAR_Q:
                sq = tl.maximum(tl.max(tl.abs(q), axis=1) * (1.0 / 127.0), 1.0e-20)
                x = q / sq[:, None]
                q8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)
    if PASS == 2:
        # Resume far-field online softmax without repeating its QK/PV loops.
        prior = t * ns + sp
        m_i = tl.load(PM + prior * T + r)
        l_i = tl.load(PL + prior * T + r)
        acc = tl.load(PACC + (prior * T + r)[:, None] * D + d[None, :])
    else:
        m_i = tl.full((RT,), float("-inf"), tl.float32)
        l_i = tl.zeros((RT,), tl.float32)
        acc = tl.zeros((RT, D), tl.float32)
    if PASS != 2:
        lo = sp * lps
        hi = tl.minimum(nb, lo + lps)
        if FAR_PREFETCH_LZ:
            # Prime next iteration's log-partition load before the QK/PV work.
            pc = lo + tl.arange(0, BN)
            pmask = pc < hi
            pref_lz = tl.load(LZ + gt * nb + pc, mask=pmask, other=float("-inf"))
        for n0 in range(lo, hi, BN):
            c = n0 + tl.arange(0, BN)
            cm = c < hi
            if FAR_PREFETCH_LZ:
                lz = pref_lz
                # Issue the following block's LZ load before this block's
                # tensor-core dots. The last iteration is fully masked.
                pc = c + BN
                pmask = pc < hi
                pref_lz = tl.load(LZ + gt * nb + pc, mask=pmask, other=float("-inf"))
            else:
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
            if FUSED_VB:
                # One centre per tile: recompute Vbar for each full leaf
                # directly in registers; do NOT load from VB or write VB.
                # The existing Tile far online-softmax/PV math follows.
                vb = tl.full((BN,D),0.0,tl.bfloat16)
                row = tl.arange(0,BN)
                for lane in range(0,BN):
                    leaf = n0 + lane
                    if leaf < nb:
                        one = _fused_vbar_one_leaf(
                            mu, KBF, V, head, leaf, scale,
                            D=D, LEAF=LEAF, PREC=PREC,
                            KB_SH=KB_SH,KB_SS=KB_SS,KB_SD=KB_SD,
                            V_SH=V_SH,V_SS=V_SS,V_SD=V_SD,
                        )
                        vb = tl.where(row[:,None]==lane,one[None,:],vb)
            elif VB_SPLIT_STREAM:
                vb = tl.load(VB + (gt * VB_SPAN + c - VB_START)[:, None] * D + d[None, :],
                             mask=cm[:, None], other=0.0)
            else:
                vb = tl.load(VB + (gt * nb + c)[:, None] * D + d[None, :],
                             mask=cm[:, None], other=0.0)
            if DOT == 1:
                if FP8:
                    svc = tl.load(SV + gt * nb + c, mask=cm, other=0.0)
                    if FAR_PV_ACC:
                        acc = tl.dot((p * svc[None, :]).to(tl.float16), vb.to(tl.float16),
                                     acc=acc * alpha[:, None])
                    else:
                        acc = acc * alpha[:, None] + tl.dot((p * svc[None, :]).to(tl.float16), vb.to(tl.float16))
                else:
                    if FAR_PV_ACC:
                        acc = tl.dot(p.to(tl.bfloat16), vb, acc=acc * alpha[:, None])
                    else:
                        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
            else:
                if FAR_PV_ACC:
                    acc = tl.dot(p, vb.to(tl.float32), input_precision=PREC,
                                 acc=acc * alpha[:, None])
                else:
                    acc = acc * alpha[:, None] + tl.dot(p, vb.to(tl.float32), input_precision=PREC)
            m_i = m_new
    if PASS != 1:
        if DOT == 1 and LATE_NEAR_Q:
            # Late preparation: Q operands for exact near-field are not
            # live across the far-field loop (compiled within one kernel).
            qb = q.to(tl.bfloat16)
            if I8:
                sq = tl.maximum(tl.max(tl.abs(q), axis=1) * (1.0 / 127.0), 1.0e-20)
                x = q / sq[:, None]
                q8 = tl.where(x >= 0, x + 0.5, x - 0.5).to(tl.int8)
        e_n = kx * LEAF
        e_lo = sp * eps
        e_hi = tl.minimum(e_n, e_lo + eps)
        for e0 in range(e_lo, e_hi, BN_NEAR):
            j = e0 + tl.arange(0, BN_NEAR)
            jm = j < e_hi
            leaf = tl.load(IDX + gt * kx + j // LEAF, mask=jm, other=0)
            tok = leaf * LEAF + j % LEAF
            kk = tl.load(K + (key_base + j)[:, None] * D + d[None, :], mask=jm[:, None], other=0.0)
            if V_DIRECT:
                vv=tl.load(V+(t // REAL_TILES_PER_HEAD)*V_SH+tok[:,None]*V_SS+d[None,:]*V_SD,
                           mask=jm[:,None],other=0.0)
            else:
                vv = tl.load(V + (key_base + tok)[:, None] * D + d[None, :], mask=jm[:, None], other=0.0)
            if DOT == 1:
                if I8:
                    ksc = tl.load(KS + key_base + j, mask=jm, other=0.0)
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
                if NEAR_PV_ACC:
                    acc = tl.dot(p.to(tl.bfloat16), vv, acc=acc * alpha[:, None])
                else:
                    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
            else:
                if NEAR_PV_ACC:
                    acc = tl.dot(p, vv.to(tl.float32), input_precision=PREC,
                                 acc=acc * alpha[:, None])
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


