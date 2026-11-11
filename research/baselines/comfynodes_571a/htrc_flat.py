"""Flat (tree-free) HTRC-family attention operator.

Same idea as the tree executor -- expand each 16-token leaf's log-partition around a
query centre and evaluate only the near field exactly -- but with no adaptive tree walk:

  queries are cut into tiles of `tile` rows; each tile has its own expansion point
  mu_t = mean of its rows.  For every (tile, leaf l) the exact leaf cumulants at mu_t:

      a_s   = scale * mu_t . k_s                    (s = 16 tokens of leaf l)
      lz    = logsumexp_s a_s                        log-partition at the centre
      g     = scale * sum_s w_s k_s                  gradient in q      (w = softmax a)
      vbar  = sum_s w_s v_s                          value at the centre
      tr    = scale^2 (sum_s w_s |k_s|^2 - |sum_s w_s k_s|^2)   Hessian trace

  far field, query q of tile t:  logit(q, l) = lz + (q - mu_t) . g  [+ 0.5 tr |q-mu|^2 / d]
                                  value(q, l) = vbar
  near field: the `exact_leaves` leaves with the largest far logit (max over the tile's
  rows) are replaced by their 16 exact tokens; keys past the last full leaf are always
  exact.  Both parts are merged with a shared max (online-softmax algebra).

Everything is fixed-shape batched matmuls per chunk of tiles: no host synchronisation,
no data-dependent sizes.  Reference semantics for a later fused kernel.
"""
from __future__ import annotations

import math
import time

import torch


class _Tick:
    def __init__(self, prof):
        self.prof = prof
        if prof is not None:
            torch.cuda.synchronize()
            self.t = time.perf_counter()

    def __call__(self, name):
        if self.prof is not None:
            torch.cuda.synchronize()
            t = time.perf_counter()
            self.prof[name] = self.prof.get(name, 0.0) + 1e3 * (t - self.t)
            self.t = t


def kmeans_order(q, clusters=32, sample=4096, iters=8):
    """Permutation grouping queries by a small k-means (Lloyd on an even sample)."""
    n = int(q.shape[0])
    x = q.float()
    idx = torch.linspace(0, n - 1, min(sample, n), device=q.device).round().long()
    s = x.index_select(0, idx)
    c = s[torch.linspace(0, s.shape[0] - 1, clusters, device=q.device).round().long()].clone()
    for _ in range(iters):
        a = torch.cdist(s, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, s)
        cnt = torch.zeros((clusters,), device=q.device, dtype=s.dtype).index_add_(
            0, a, torch.ones_like(a, dtype=s.dtype))                  # no host sync (bincount syncs)
        c = torch.where((cnt > 0)[:, None], sums / cnt.clamp_min(1.0)[:, None], c)
    assign = torch.cdist(x, c).argmin(1)
    return torch.argsort(assign, stable=True)


_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)        # e2m1 magnitudes (NVFP4 element grid)
_FP4_THR = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def fake_quant(x, fmt, group=0):
    """Round-trip x [..., D] through a low-bit format with symmetric per-group scales
    (group=0: one scale per row). fmt: int8, int4, fp8 (e4m3), fp4 (e2m1; default group 16).
    Simulation only: measures the accuracy cost before any real low-bit kernel exists."""
    shape = x.shape
    d = int(shape[-1])
    if fmt == "fp4" and not group:
        group = 16
    gsz = d if not group else int(group)
    if d % gsz:
        raise ValueError("quant group must divide the feature size")
    y = x.float().reshape(*shape[:-1], d // gsz, gsz)
    amax = y.abs().amax(-1, keepdim=True).clamp_min(1e-20)
    if fmt in ("int8", "int4"):
        qmax = 127.0 if fmt == "int8" else 7.0
        sc = amax / qmax
        y = (y / sc).round().clamp_(-qmax, qmax) * sc
    elif fmt == "fp8":
        sc = amax / 448.0
        y = (y / sc).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn).float() * sc
    elif fmt == "fp4":
        sc = amax / 6.0
        a = (y / sc).abs()
        grid = torch.tensor(_FP4, device=x.device, dtype=torch.float32)
        thr = torch.tensor(_FP4_THR, device=x.device, dtype=torch.float32)
        y = torch.sign(y) * grid[torch.bucketize(a, thr)] * sc
    else:
        raise ValueError(f"unknown quant format {fmt}")
    return y.reshape(shape)


@torch.inference_mode()
def flat_attention(q, k, v, *, tile=64, exact_leaves=64, order="seq", curvature="lin",
                   tile_chunk=32, precision="fp32", leaf=16, scale=None, prof=None,
                   select="max", backend="torch", store="fp32", splits=8, dot="tf32",
                   sel_stride=1, kern=None, quant=None, sink_leaves=0, tail=None, ws=None):
    """q [N,D], k/v [M,D] (any float dtype) -> out [N,D] fp32.

    select: "max" = near leaves by the largest far logit over the tile rows (needs the
    [T, leaves] logits); "bound" = by lz + max_rows|q - mu| * |g| (upper bound of that
    logit, no logits needed; measured much worse).  backend "triton" (h3/htrc_flat_triton.py):
    fused stats, row-max selection and far+near tile kernels; curvature="lin", no partial leaf.
    dot="bf16": far/near dots on bf16 operands (fp32 accumulate; needs store="bf16" and bf16-exact
    k/v); sel_stride>1: near-leaf selection from every sel_stride-th tile row only; kern: dict of
    launch parameters (tile_bn, tile_warps, tile_stages, stats_bt, stats_bl, stats_warps, sel_bn,
    sel_warps).
    quant (triton backend, simulation): dict kv / state / q -> "int8"|"int4"|"fp8"|"fp4" and
    group (0 = per row): k,v per token at the input (quantised cache), the per-(tile, leaf)
    far-field state g/vbar, and the query rows.
    """
    if backend == "triton":
        if curvature != "lin":
            raise ValueError("triton backend needs curvature='lin'")
        return _flat_triton(q, k, v, tile, exact_leaves, order, tile_chunk, leaf, scale, prof,
                            "tf32" if precision == "tf32" else "ieee", select,
                            {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn, "int8": torch.int8}.get(store, torch.float32), splits,
                            dot, sel_stride, kern or {}, quant or {}, int(sink_leaves), tail, ws)
    if sink_leaves or tail is not None:
        raise ValueError("sink_leaves / tail are supported by the triton backend only")
    if select not in ("max", "bound"):
        raise ValueError("select must be max or bound")
    if order not in ("seq", "kmeans"):
        raise ValueError("order must be seq or kmeans")
    if curvature not in ("lin", "iso"):
        raise ValueError("curvature must be lin or iso")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    try:
        return _flat(q, k, v, tile, exact_leaves, order, curvature, tile_chunk, leaf, scale, prof,
                     select)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def _tiles(q, tile, order):
    """(qt [nt,T,D] fp32, mu [nt,D] tile centres over real rows, perm or None, n)."""
    n, d = int(q.shape[0]), int(q.shape[1])
    qf = q.float()
    perm = kmeans_order(q) if order == "kmeans" else None
    qs = qf.index_select(0, perm) if perm is not None else qf
    nt = -(-n // tile)
    npad = nt * tile
    if npad > n:
        qs = torch.cat([qs, qs[-1:].expand(npad - n, -1)], 0)
    qt = qs.view(nt, tile, d).contiguous()
    mu = qt.mean(1)
    if npad > n:                                       # last tile: centre over its real rows only
        mu[-1] = qt[-1, : n - (nt - 1) * tile].mean(0)
    return qt, mu.contiguous(), perm, n


def _unperm(out, n, perm):
    res = out[:n]
    if perm is not None:
        full = torch.empty_like(res)
        full.index_copy_(0, perm, res)
        res = full
    return res


def _load_triton():
    import importlib.util
    import sys
    from pathlib import Path
    name = "_apl_htrc_flat_triton"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("htrc_flat_triton.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _flat_triton(q, k, v, tile, exact_leaves, order, tile_chunk, leaf, scale, prof, prec, select,
                 store, splits, dot, sel_stride, kern, quant, sink_leaves=0, tail=None, ws=None):
    tri = _load_triton()
    tick = _Tick(prof)
    d = int(q.shape[1])
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)
    nb = int(k.shape[0]) // leaf
    if nb * leaf != int(k.shape[0]):
        raise ValueError("triton backend: key count must be a multiple of the leaf size")
    qg = int(quant.get("group", 0))
    if quant.get("kv"):
        k = fake_quant(k, quant["kv"], qg)
        v = fake_quant(v, quant["kv"], qg)
    if quant.get("q"):
        q = fake_quant(q, quant["q"], qg)
    if k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16 and dot == "bf16":
        kf, vf = k.contiguous(), v.contiguous()          # the stats kernel upcasts on load: no fp32 copies
    elif ws is None:
        kf = k.float().contiguous()
        vf = v.float().contiguous()
    else:
        kf = tri._buf(ws, "kf", tuple(k.shape), torch.float32, k.device)
        vf = tri._buf(ws, "vf", tuple(v.shape), torch.float32, v.device)
        kf.copy_(k)
        vf.copy_(v)
    use_bf = dot == "bf16"
    if use_bf and store not in (torch.bfloat16, torch.float8_e4m3fn, torch.int8):
        raise ValueError("dot='bf16' needs store='bf16', 'fp8' or 'int8'")
    if store in (torch.float8_e4m3fn, torch.int8) and (not use_bf or select == "bound" or quant.get("state")):
        raise ValueError("store='fp8' needs dot='bf16', select='max' and no state fake-quant")
    k_ex = (kf if kf.dtype == torch.bfloat16 else k.to(torch.bfloat16).contiguous()) if use_bf else kf
    v_ex = (vf if vf.dtype == torch.bfloat16 else v.to(torch.bfloat16).contiguous()) if use_bf else vf
    ks = None
    if store == torch.int8:
        # near-field keys as int8 with a per-token scale (int8 x int8 -> int32 dot in the tile kernel)
        kfl = k_ex.float()
        ks = (kfl.abs().amax(-1) * (1.0 / 127.0)).clamp_min_(1e-20).contiguous()
        k_ex = (kfl / ks[:, None]).round_().clamp_(-127, 127).to(torch.int8).contiguous()
        del kfl
    dflag = 1 if use_bf else 0
    qt, mu, perm, n = _tiles(q, tile, order)
    nt = int(qt.shape[0])
    grp = max(1, int(kern.get("group", 1)))           # G consecutive tiles share one expansion centre
    if grp > 1:
        ntp = -(-nt // grp) * grp
        if ntp > nt:                                  # pad with repeats of the last tile
            qt = torch.cat([qt, qt[-1:].expand(ntp - nt, -1, -1)], 0)
        mu = qt.view(ntp // grp, grp * tile, d).mean(1).contiguous()    # [groups, D]
        nt_pad = ntp
    else:
        nt_pad = nt
    sink_leaves = max(0, min(int(sink_leaves), nb))     # leading leaves (conditioning KV) kept exact
    kx_n = min(int(exact_leaves) + sink_leaves, nb)
    out = tri._buf(ws, "out", (nt_pad, tile, d), torch.float32, q.device)
    tick("prep")
    gstep = max(1, int(tile_chunk) // grp)            # groups per chunk
    ng = nt_pad // grp
    for g0 in range(0, ng, gstep):
        g1 = min(ng, g0 + gstep)
        gc = g1 - g0
        t0, t1 = g0 * grp, g1 * grp
        mu_c, q_c = mu[g0:g1], qt[t0:t1]
        g, lz, vb, sg, sv = tri.leaf_stats(mu_c, kf, vf, nb, scale, leaf=leaf, prec=prec, store=store,
                                   bt=kern.get("stats_bt", 16), bl=kern.get("stats_bl", 4),
                                   warps=kern.get("stats_warps", 4), ws=ws)
        if quant.get("state"):
            g.copy_(fake_quant(g, quant["state"], qg))
            vb.copy_(fake_quant(vb, quant["state"], qg))
        tick("stats")
        excl = tri._buf(ws, "excl", (gc, nb), torch.int8, q.device)
        excl.zero_()
        idx = None
        if kx_n > 0:
            qg_rows = q_c.reshape(gc, grp * tile, d)
            if select == "bound":
                r = (qg_rows - mu_c[:, None, :]).norm(dim=-1).amax(1)
                sel = lz + r[:, None] * g.float().norm(dim=-1)
            else:
                sel = tri.rowmax_select(qg_rows, mu_c, g, lz, prec=prec, splits=splits,
                                        stride=sel_stride, dot=dflag, bn=kern.get("sel_bn", 32),
                                        warps=kern.get("sel_warps", 4), ws=ws, sg=sg)
            if sink_leaves:
                sel[:, :sink_leaves] = float("inf")
            idx = torch.topk(sel, kx_n, dim=-1, sorted=False).indices
            excl.scatter_(1, idx, 1)
        tick("select")
        tri.tile_attention(q_c, mu_c, g, lz, vb, excl, idx, k_ex, v_ex, scale,
                                        leaf=leaf, prec=prec, splits=splits, dot=dflag,
                                        bn=kern.get("tile_bn", 32), warps=kern.get("tile_warps", 4),
                                        stages=kern.get("tile_stages", 1), rb=kern.get("tile_rb", 1), tail=tail, ws=ws,
                                        group=grp, sg=sg, sv=sv, out=out[t0:t1], ks=ks)
        del g, lz, vb, excl
        tick("tile")
    res = _unperm(out[:nt].reshape(nt * tile, d), n, perm)
    tick("finish")
    return res


def _flat(q, k, v, tile, exact_leaves, order, curvature, tile_chunk, leaf, scale, prof, select):
    tick = _Tick(prof)
    dev = q.device
    n, d = int(q.shape[0]), int(q.shape[1])
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)
    m = int(k.shape[0])
    nb = m // leaf
    pre = nb * leaf
    if nb <= 0:
        raise ValueError("flat attention needs at least one full leaf")
    kf = k[:pre].float().contiguous()
    vf = v[:pre].float().contiguous()
    k3, v3 = kf.view(nb, leaf, d), vf.view(nb, leaf, d)
    tail = m - pre
    kt = k[pre:].float() if tail else None
    vt = v[pre:].float() if tail else None
    kn2 = (k3 * k3).sum(-1) if curvature == "iso" else None      # [nb, leaf]

    qf = q.float()
    perm = kmeans_order(q) if order == "kmeans" else None
    qs = qf.index_select(0, perm) if perm is not None else qf
    nt = -(-n // tile)
    npad = nt * tile
    valid = torch.ones((npad,), device=dev, dtype=torch.float32)
    if npad > n:
        qs = torch.cat([qs, qs[-1:].expand(npad - n, -1)], 0)
        valid[n:] = 0.0
    qt = qs.view(nt, tile, d)
    vw = valid.view(nt, tile)
    mu = (qt * vw[..., None]).sum(1) / vw.sum(1, keepdim=True)   # tile centres, real rows only
    kx_n = min(int(exact_leaves), nb)
    out = torch.empty((npad, d), device=dev, dtype=torch.float32)
    ar = torch.arange(leaf, device=dev)
    tick("prep")

    for t0 in range(0, nt, tile_chunk):
        t1 = min(nt, t0 + tile_chunk)
        tc = t1 - t0
        mu_c, q_c = mu[t0:t1], qt[t0:t1]                          # [tc,d], [tc,T,d]
        # ---- exact leaf cumulants at the tile centres
        a = (mu_c @ kf.T).mul_(scale).view(tc, nb, leaf)
        lz = torch.logsumexp(a, -1)                               # [tc,nb]
        w = torch.softmax(a, -1)
        del a
        wb = w.permute(1, 0, 2)                                   # [nb,tc,leaf]
        g = torch.bmm(wb, k3).mul_(scale)                         # [nb,tc,d]
        vb = torch.bmm(wb, v3)                                    # [nb,tc,d]
        tick("stats")
        # ---- far field logits
        dq = q_c - mu_c[:, None, :]                               # [tc,T,d]
        lg = torch.bmm(dq, g.permute(1, 2, 0)).add_(lz[:, None, :])   # [tc,T,nb]
        if curvature == "iso":
            tr = (w * kn2[None]).sum(-1).mul_(scale * scale) - g.square().sum(-1).T  # [tc,nb]
            lg.add_(0.5 / d * tr.clamp_min(0.0)[:, None, :] * dq.square().sum(-1)[:, :, None])
            del tr
        gn = g.norm(dim=-1).T if select == "bound" else None     # [tc,nb]
        del w, wb, g
        tick("far_logits")
        # ---- near field selection (fixed K per tile) and exclusion from the far field
        if kx_n > 0:
            if select == "bound":
                r = dq.norm(dim=-1).amax(1)
                sel = lz + r[:, None] * gn
            else:
                sel = lg.amax(1)
            idx = torch.topk(sel, kx_n, dim=-1, sorted=False).indices     # [tc,K]
            lg.scatter_(2, idx[:, None, :].expand(-1, tile, -1), -math.inf)
        tick("select")
        mf = lg.amax(-1)                                          # -inf if every leaf is exact
        mfs = torch.where(torch.isfinite(mf), mf, torch.zeros_like(mf))
        p = lg.sub_(mfs[..., None]).exp_()
        zf = p.sum(-1)
        nf = torch.bmm(p, vb.transpose(0, 1))                     # [tc,T,d]
        del p, lg, vb
        tick("far_apply")
        # ---- exact near field
        parts_k, parts_v = [], []
        if kx_n > 0:
            tok = (idx[..., None] * leaf + ar).view(tc, kx_n * leaf)
            parts_k.append(kf.index_select(0, tok.view(-1)).view(tc, kx_n * leaf, d))
            parts_v.append(vf.index_select(0, tok.view(-1)).view(tc, kx_n * leaf, d))
        if tail:
            parts_k.append(kt[None].expand(tc, -1, -1))
            parts_v.append(vt[None].expand(tc, -1, -1))
        if parts_k:
            kx = torch.cat(parts_k, 1) if len(parts_k) > 1 else parts_k[0]
            vx = torch.cat(parts_v, 1) if len(parts_v) > 1 else parts_v[0]
            le = torch.bmm(q_c, kx.transpose(1, 2)).mul_(scale)   # [tc,T,E]
            me = le.amax(-1)
            pe = le.sub_(me[..., None]).exp_()
            ze = pe.sum(-1)
            ne = torch.bmm(pe, vx)
            del kx, vx, le, pe
            mm = torch.maximum(mf, me)
            af, ae = torch.exp(mf - mm), torch.exp(me - mm)
            num = nf * af[..., None] + ne * ae[..., None]
            den = zf * af + ze * ae
        else:
            num, den = nf, zf
        # (with an empty far field mf = -inf -> af = 0; zf = nf = 0 there anyway)
        out[t0 * tile:t1 * tile] = (num / den.clamp_min(1e-30)[..., None]).view(tc * tile, d)
        tick("exact_merge")

    res = out[:n]
    if perm is not None:
        full = torch.empty_like(res)
        full.index_copy_(0, perm, res)
        res = full
    tick("finish")
    return res


PROD = dict(tile=64, exact_leaves=256, tile_chunk=128, backend="triton", precision="tf32",
            store="int8", dot="bf16", splits=4, kern=dict(tile_warps=4, tile_bn=64, tile_stages=2))
# measured on 72 full-Q maps (RTX 5070 Ti): median/p90/max error 9.9/26.5/36.7 %, ~6.7 ms/head
# (5.3/15.7/24.0 % at exact_leaves=512, ~8.5 ms); ~310 MiB at tile_chunk=128.


def flat_attention_prod(q, k, v, **over):
    """Production configuration of the flat operator (bf16 tensor-core dots, bf16 leaf state)."""
    return flat_attention(q, k, v, **{**PROD, **over})
