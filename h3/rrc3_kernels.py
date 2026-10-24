"""Small Triton kernels for the standalone H3 RRC3 backend.

This module has no dependency on sol_flex / attention_flex_native.  It only
kernelizes operations owned by sol_rrc3_live.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


_WINDOWS_DLL_HANDLES = []


def _ensure_windows_triton_dll_search() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if _WINDOWS_DLL_HANDLES:
        return

    candidates = []
    for raw in (sys.prefix, sys.base_prefix):
        if raw:
            candidates.append(Path(raw))
    try:
        candidates.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass
    try:
        candidates.append(
            Path(triton.__file__).resolve().parent
            / "backends"
            / "nvidia"
            / "lib"
        )
    except Exception:
        pass
    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")

    seen = set()
    for path in candidates:
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        key = resolved.lower()
        if key in seen or not Path(resolved).is_dir():
            continue
        seen.add(key)
        try:
            _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(resolved))
        except OSError:
            continue


@triton.jit
def _online_merge_kernel(
    global_m,
    global_z,
    global_n,
    chunk_m,
    chunk_z,
    chunk_n,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask = d < D

    gm = tl.load(global_m + row)
    gz = tl.load(global_z + row)
    cm = tl.load(chunk_m + row)
    cz = tl.load(chunk_z + row)

    nm = tl.maximum(gm, cm)
    old_scale = tl.exp(gm - nm)
    chunk_scale = tl.exp(cm - nm)

    gn = tl.load(global_n + row * D + d, mask=mask, other=0.0)
    cn = tl.load(chunk_n + row * D + d, mask=mask, other=0.0)

    tl.store(global_m + row, nm)
    tl.store(global_z + row, gz * old_scale + cz * chunk_scale)
    tl.store(
        global_n + row * D + d,
        gn * old_scale + cn * chunk_scale,
        mask=mask,
    )


def online_merge_(
    global_m: torch.Tensor,
    global_z: torch.Tensor,
    global_n: torch.Tensor,
    chunk_m: torch.Tensor,
    chunk_z: torch.Tensor,
    chunk_n: torch.Tensor,
) -> None:
    """In-place stable online merge for [H,Q] masses and [H,Q,D] numerators."""
    if global_m.ndim != 2 or global_z.shape != global_m.shape:
        raise ValueError("RRC3 online merge expects matching [H,Q] m/z tensors")
    if global_n.ndim != 3 or global_n.shape[:2] != global_m.shape:
        raise ValueError("RRC3 online merge expects [H,Q,D] numerator")
    if chunk_m.shape != global_m.shape or chunk_z.shape != global_m.shape:
        raise ValueError("RRC3 online merge chunk m/z shape mismatch")
    if chunk_n.shape != global_n.shape:
        raise ValueError("RRC3 online merge chunk numerator shape mismatch")

    tensors = (global_m, global_z, global_n, chunk_m, chunk_z, chunk_n)
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("RRC3 online merge requires fp32 accumulators")
    if any(t.device != global_m.device for t in tensors):
        raise ValueError("RRC3 online merge tensors must share one device")

    # Keep a CPU/reference path for focused tests.  The live H3 path is CUDA.
    if global_m.device.type != "cuda":
        nm = torch.maximum(global_m, chunk_m)
        old_scale = torch.exp(global_m - nm)
        chunk_scale = torch.exp(chunk_m - nm)
        global_z.copy_(global_z * old_scale + chunk_z * chunk_scale)
        global_n.copy_(
            global_n * old_scale[:, :, None]
            + chunk_n * chunk_scale[:, :, None]
        )
        global_m.copy_(nm)
        return

    _ensure_windows_triton_dll_search()
    h, q = (int(v) for v in global_m.shape)
    d = int(global_n.shape[-1])
    if d <= 0 or d > 256:
        raise ValueError(f"RRC3 online merge supports 1..256 channels, got {d}")

    gm = global_m.contiguous()
    gz = global_z.contiguous()
    gn = global_n.contiguous()
    cm = chunk_m.contiguous()
    cz = chunk_z.contiguous()
    cn = chunk_n.contiguous()

    # Runtime accumulators are created contiguous. Refuse silent detached copies
    # for mutation targets if that invariant is ever changed.
    if gm.data_ptr() != global_m.data_ptr():
        raise ValueError("RRC3 global_m must be contiguous")
    if gz.data_ptr() != global_z.data_ptr():
        raise ValueError("RRC3 global_z must be contiguous")
    if gn.data_ptr() != global_n.data_ptr():
        raise ValueError("RRC3 global_n must be contiguous")

    block_d = triton.next_power_of_2(d)
    _online_merge_kernel[(h * q,)](
        gm,
        gz,
        gn,
        cm,
        cz,
        cn,
        D=d,
        BLOCK_D=block_d,
        num_warps=4,
    )









@triton.jit
def _htrc_mgs_kernel(
    X,
    out,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PASSES: tl.constexpr,
    EPS: tl.constexpr,
):
    """Per-target modified Gram-Schmidt with optional reorthogonalization.

    One Triton program owns one [D,K] target matrix.  PASSES=2 performs the
    standard DGKS-style reorthogonalization sweep over the already accepted
    columns, retaining the low-memory one-target program geometry while
    addressing the loss of orthogonality observed in the first one-pass gate.
    """
    mid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    dmask = d < D

    for j in tl.static_range(0, K):
        off = (mid * D + d) * K + j
        v = tl.load(X + off, mask=dmask, other=0.0).to(tl.float32)

        for _pass in tl.static_range(0, PASSES):
            for i in tl.static_range(0, K):
                if i < j:
                    qoff = (mid * D + d) * K + i
                    qi = tl.load(
                        out + qoff, mask=dmask, other=0.0
                    ).to(tl.float32)
                    dot = tl.sum(v * qi, axis=0)
                    v -= dot * qi

        n2 = tl.sum(v * v, axis=0)
        inv = tl.rsqrt(tl.maximum(n2, EPS))
        qj = v * inv
        tl.store(out + off, qj, mask=dmask)


def htrc_mgs(
    X: torch.Tensor,
    *,
    passes: int = 1,
    eps: float = 1.0e-20,
) -> torch.Tensor:
    """Native small-matrix MGS / MGS2 orthogonalization for HTRC."""
    if X.ndim != 3:
        raise ValueError("HTRC MGS expects X=[M,D,K]")
    m, d, k = (int(v) for v in X.shape)
    if d <= 0 or d > 256:
        raise ValueError(f"HTRC MGS supports D=1..256, got {d}")
    if k <= 0 or k > 16:
        raise ValueError(f"HTRC MGS supports K=1..16, got {k}")
    if int(passes) not in (1, 2):
        raise ValueError("HTRC MGS passes must be 1 or 2")

    if X.device.type != "cuda":
        q, _ = torch.linalg.qr(X.float(), mode="reduced")
        return q

    _ensure_windows_triton_dll_search()
    xc = X.float().contiguous()
    out = torch.empty_like(xc)
    block_d = triton.next_power_of_2(d)
    _htrc_mgs_kernel[(m,)](
        xc,
        out,
        D=d,
        K=k,
        BLOCK_D=block_d,
        PASSES=int(passes),
        EPS=float(eps),
        num_warps=4,
    )
    return out



@triton.jit
def _htrc_gram_kernel(
    X,
    gram,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute one small KxK Gram matrix per HTRC target."""
    mid = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)[:, None]
    kk = tl.arange(0, K)[None, :]
    mask = d < D
    off = (mid * D + d) * K + kk
    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    g = tl.dot(tl.trans(x), x, input_precision="ieee")
    gi = tl.arange(0, K)[:, None]
    gj = tl.arange(0, K)[None, :]
    goff = (mid * K + gi) * K + gj
    tl.store(gram + goff, g)


@triton.jit
def _htrc_polar_apply_kernel(
    X,
    gram,
    out,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Apply one Newton-Schulz polar retraction Q(3I-Q^TQ)/2."""
    mid = tl.program_id(0)
    db = tl.program_id(1)
    d = db * BLOCK_D + tl.arange(0, BLOCK_D)[:, None]
    kk = tl.arange(0, K)[None, :]
    xoff = (mid * D + d) * K + kk
    x = tl.load(X + xoff, mask=d < D, other=0.0).to(tl.float32)

    gi = tl.arange(0, K)[:, None]
    gj = tl.arange(0, K)[None, :]
    goff = (mid * K + gi) * K + gj
    g = tl.load(gram + goff).to(tl.float32)
    eye = (gi == gj).to(tl.float32)
    corr = 1.5 * eye - 0.5 * g
    y = tl.dot(x, corr, input_precision="ieee")
    tl.store(out + xoff, y, mask=d < D)


def htrc_polar_retract(
    X: torch.Tensor,
    *,
    steps: int = 1,
) -> torch.Tensor:
    """Native Newton-Schulz polar retraction for small HTRC bases.

    Each step applies Q <- Q(3I-Q^TQ)/2.  This avoids QR/Cholesky and is
    intended for an already-near-orthogonal basis such as the native MGS2
    result.  K is deliberately limited to the target solver's K<=16 regime.
    """
    if X.ndim != 3:
        raise ValueError("HTRC polar retraction expects X=[M,D,K]")
    m, d, k = (int(v) for v in X.shape)
    if d <= 0 or d > 256:
        raise ValueError(f"HTRC polar retraction supports D=1..256, got {d}")
    if k <= 0 or k > 16:
        raise ValueError(f"HTRC polar retraction supports K=1..16, got {k}")
    if int(steps) not in (1, 2):
        raise ValueError("HTRC polar retraction steps must be 1 or 2")

    if X.device.type != "cuda":
        q = X.float()
        for _ in range(int(steps)):
            gram = torch.matmul(q.transpose(1, 2), q)
            eye = torch.eye(k, device=q.device, dtype=torch.float32)[None]
            q = torch.matmul(q, 1.5 * eye - 0.5 * gram)
        return q.contiguous()

    _ensure_windows_triton_dll_search()
    q = X.float().contiguous()
    block_d_gram = triton.next_power_of_2(d)
    block_d_apply = 32 if d >= 32 else triton.next_power_of_2(d)
    for _ in range(int(steps)):
        gram = torch.empty((m, k, k), device=q.device, dtype=torch.float32)
        _htrc_gram_kernel[(m,)](
            q,
            gram,
            D=d,
            K=k,
            BLOCK_D=block_d_gram,
            num_warps=4,
        )
        out = torch.empty_like(q)
        _htrc_polar_apply_kernel[(m, triton.cdiv(d, block_d_apply))](
            q,
            gram,
            out,
            D=d,
            K=k,
            BLOCK_D=block_d_apply,
            num_warps=4,
        )
        q = out
    return q


@triton.jit
def _htrc_target_h_apply_tiled_kernel(
    X,
    weights,
    parent_grad,
    cids,
    nids,
    node_leaf_idx,
    node_leaf_mask,
    leaf_grad,
    leaf_eval,
    leaf_modes,
    out,
    B,
    L,
    D: tl.constexpr,
    K: tl.constexpr,
    K_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Reuse one descendant-leaf load across several subspace columns."""
    mid = tl.program_id(0)
    ktile = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    kk = ktile * K_TILE + tl.arange(0, K_TILE)
    dmask = d < D
    kmask = kk < K

    cid = tl.load(cids + mid).to(tl.int32)
    nid = tl.load(nids + mid).to(tl.int32)

    xoff = (mid * D + d[:, None]) * K + kk[None, :]
    xv = tl.load(
        X + xoff,
        mask=dmask[:, None] & kmask[None, :],
        other=0.0,
    ).to(tl.float32)
    pg = tl.load(
        parent_grad + mid * D + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_D, K_TILE), dtype=tl.float32)

    for li in tl.range(0, L):
        valid_leaf = tl.load(
            node_leaf_mask + nid * L + li
        ).to(tl.int1)
        bid = tl.load(
            node_leaf_idx + nid * L + li,
            mask=valid_leaf,
            other=0,
        ).to(tl.int32)
        w = tl.load(
            weights + mid * L + li,
            mask=valid_leaf,
            other=0.0,
        ).to(tl.float32)

        leaf_base = cid * B + bid
        gv = tl.load(
            leaf_grad + leaf_base * D + d,
            mask=dmask & valid_leaf,
            other=0.0,
        ).to(tl.float32)
        dg = gv - pg
        gd_dot = tl.sum(dg[:, None] * xv, axis=0)
        acc += w * dg[:, None] * gd_dot[None, :]

        for r in tl.static_range(0, 8):
            uv = tl.load(
                leaf_modes + (leaf_base * D + d) * 8 + r,
                mask=dmask & valid_leaf,
                other=0.0,
            ).to(tl.float32)
            proj = tl.sum(uv[:, None] * xv, axis=0)
            lv = tl.load(
                leaf_eval + leaf_base * 8 + r,
                mask=valid_leaf,
                other=0.0,
            ).to(tl.float32)
            acc += w * uv[:, None] * lv * proj[None, :]

    tl.store(
        out + xoff,
        acc,
        mask=dmask[:, None] & kmask[None, :],
    )




@triton.jit
def _htrc_target_reduce_range_kernel(
    cids,
    nids,
    leaf_order,
    node_leaf_start,
    node_leaf_count,
    leaf_logz,
    leaf_grad,
    leaf_value,
    leaf_trace,
    out_logz,
    out_grad,
    out_value,
    out_trace,
    B,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Exact target sufficient statistics from contiguous descendant ranges."""
    mid = tl.program_id(0)
    cid = tl.load(cids + mid).to(tl.int32)
    nid = tl.load(nids + mid).to(tl.int32)
    start = tl.load(node_leaf_start + nid).to(tl.int32)
    count = tl.load(node_leaf_count + nid).to(tl.int32)

    m = -float("inf")
    for li in tl.range(0, count):
        bid = tl.load(leaf_order + start + li).to(tl.int32)
        lz = tl.load(leaf_logz + cid * B + bid).to(tl.float32)
        m = tl.maximum(m, lz)

    d = tl.arange(0, BLOCK_D)
    dmask = d < D
    z = 0.0
    tr_sum = 0.0
    g2_sum = 0.0
    gsum = tl.zeros((BLOCK_D,), dtype=tl.float32)
    vsum = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for li in tl.range(0, count):
        bid = tl.load(leaf_order + start + li).to(tl.int32)
        base = cid * B + bid
        lz = tl.load(leaf_logz + base).to(tl.float32)
        ew = tl.exp(lz - m)
        z += ew
        gv = tl.load(
            leaf_grad + base * D + d, mask=dmask, other=0.0
        ).to(tl.float32)
        vv = tl.load(
            leaf_value + base * D + d, mask=dmask, other=0.0
        ).to(tl.float32)
        gsum += ew * gv
        vsum += ew * vv
        g2_sum += ew * tl.sum(gv * gv, axis=0)
        tr_sum += ew * tl.load(leaf_trace + base).to(tl.float32)

    invz = 1.0 / tl.maximum(z, 1.0e-20)
    pg = gsum * invz
    pv = vsum * invz
    tr = tr_sum * invz + g2_sum * invz - tl.sum(pg * pg, axis=0)

    tl.store(out_logz + mid, m + tl.log(tl.maximum(z, 1.0e-20)))
    tl.store(out_grad + mid * D + d, pg, mask=dmask)
    tl.store(out_value + mid * D + d, pv, mask=dmask)
    tl.store(out_trace + mid, tr)


def htrc_target_reduce_range(
    cids: torch.Tensor,
    nids: torch.Tensor,
    leaf_order: torch.Tensor,
    node_leaf_start: torch.Tensor,
    node_leaf_count: torch.Tensor,
    leaf_logz: torch.Tensor,
    leaf_grad: torch.Tensor,
    leaf_value: torch.Tensor,
    leaf_trace: torch.Tensor,
):
    """Exact parent logZ/grad/value/trace without padded descendant tensors."""
    if cids.ndim != 1 or nids.shape != cids.shape:
        raise ValueError("HTRC range reduce expects cids/nids=[M]")
    m = int(cids.numel())
    ccount, b = (int(x) for x in leaf_logz.shape)
    if leaf_grad.ndim != 3 or leaf_value.shape != leaf_grad.shape:
        raise ValueError("HTRC range reduce leaf grad/value shape mismatch")
    if int(leaf_grad.shape[0]) != ccount or int(leaf_grad.shape[1]) != b:
        raise ValueError("HTRC range reduce leaf state shape mismatch")
    d = int(leaf_grad.shape[-1])
    if leaf_trace.shape != (ccount, b):
        raise ValueError("HTRC range reduce leaf trace shape mismatch")
    dev = leaf_logz.device
    cc = cids.to(device=dev, dtype=torch.int32).contiguous()
    nn = nids.to(device=dev, dtype=torch.int32).contiguous()
    order = leaf_order.to(device=dev, dtype=torch.int32).contiguous()
    start = node_leaf_start.to(device=dev, dtype=torch.int32).contiguous()
    count = node_leaf_count.to(device=dev, dtype=torch.int32).contiguous()
    lz = leaf_logz.float().contiguous()
    lg = leaf_grad.contiguous()
    lv = leaf_value.contiguous()
    lt = leaf_trace.float().contiguous()
    out_lz = torch.empty((m,), device=dev, dtype=torch.float32)
    out_g = torch.empty((m, d), device=dev, dtype=torch.float32)
    out_v = torch.empty((m, d), device=dev, dtype=torch.float32)
    out_tr = torch.empty((m,), device=dev, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    _htrc_target_reduce_range_kernel[(m,)](
        cc, nn, order, start, count, lz, lg, lv, lt,
        out_lz, out_g, out_v, out_tr,
        B=b, D=d, BLOCK_D=block_d, num_warps=4,
    )
    return out_lz, out_g, out_v, out_tr


@triton.jit
def _htrc_target_h_apply_range_kernel(
    X,
    parent_logz,
    parent_grad,
    cids,
    nids,
    leaf_order,
    node_leaf_start,
    node_leaf_count,
    leaf_logz,
    leaf_grad,
    leaf_eval,
    leaf_modes,
    out,
    B,
    D: tl.constexpr,
    K: tl.constexpr,
    K_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Exact H*X from contiguous descendant ranges with weights on the fly."""
    mid = tl.program_id(0)
    ktile = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    kk = ktile * K_TILE + tl.arange(0, K_TILE)
    dmask = d < D
    kmask = kk < K

    cid = tl.load(cids + mid).to(tl.int32)
    nid = tl.load(nids + mid).to(tl.int32)
    start = tl.load(node_leaf_start + nid).to(tl.int32)
    count = tl.load(node_leaf_count + nid).to(tl.int32)
    plz = tl.load(parent_logz + mid).to(tl.float32)

    xoff = (mid * D + d[:, None]) * K + kk[None, :]
    xv = tl.load(
        X + xoff,
        mask=dmask[:, None] & kmask[None, :],
        other=0.0,
    ).to(tl.float32)
    pg = tl.load(
        parent_grad + mid * D + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((BLOCK_D, K_TILE), dtype=tl.float32)

    for li in tl.range(0, count):
        bid = tl.load(leaf_order + start + li).to(tl.int32)
        leaf_base = cid * B + bid
        lz = tl.load(leaf_logz + leaf_base).to(tl.float32)
        w = tl.exp(lz - plz)

        gv = tl.load(
            leaf_grad + leaf_base * D + d,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        dg = gv - pg
        gd_dot = tl.sum(dg[:, None] * xv, axis=0)
        acc += w * dg[:, None] * gd_dot[None, :]

        for r in tl.static_range(0, 8):
            uv = tl.load(
                leaf_modes + (leaf_base * D + d) * 8 + r,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            proj = tl.sum(uv[:, None] * xv, axis=0)
            lam = tl.load(leaf_eval + leaf_base * 8 + r).to(tl.float32)
            acc += w * uv[:, None] * lam * proj[None, :]

    tl.store(
        out + xoff,
        acc,
        mask=dmask[:, None] & kmask[None, :],
    )


def htrc_target_h_apply_range(
    X: torch.Tensor,
    parent_logz: torch.Tensor,
    parent_grad: torch.Tensor,
    cids: torch.Tensor,
    nids: torch.Tensor,
    leaf_order: torch.Tensor,
    node_leaf_start: torch.Tensor,
    node_leaf_count: torch.Tensor,
    leaf_logz: torch.Tensor,
    leaf_grad: torch.Tensor,
    leaf_eval: torch.Tensor,
    leaf_modes: torch.Tensor,
    *,
    k_tile: int = 4,
):
    """Exact tiled H*X without padded ids/masks/weights materialization."""
    if X.ndim != 3:
        raise ValueError("HTRC range H*X expects X=[M,D,K]")
    m, d, k = (int(x) for x in X.shape)
    if parent_logz.shape != (m,) or parent_grad.shape != (m, d):
        raise ValueError("HTRC range H*X parent state shape mismatch")
    if cids.shape != (m,) or nids.shape != (m,):
        raise ValueError("HTRC range H*X cids/nids shape mismatch")
    ccount, b, ld = (int(x) for x in leaf_grad.shape)
    if ld != d or leaf_logz.shape != (ccount, b):
        raise ValueError("HTRC range H*X leaf state shape mismatch")
    if leaf_eval.shape != (ccount, b, 8):
        raise ValueError("HTRC range H*X leaf eval shape mismatch")
    if leaf_modes.shape != (ccount, b, d, 8):
        raise ValueError("HTRC range H*X leaf modes shape mismatch")
    if k_tile not in (2, 4):
        raise ValueError("HTRC range H*X k_tile must be 2 or 4")
    dev = X.device
    out = torch.empty((m, d, k), device=dev, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    ktiles = triton.cdiv(k, int(k_tile))
    _htrc_target_h_apply_range_kernel[(m, ktiles)](
        X.float().contiguous(),
        parent_logz.float().contiguous(),
        parent_grad.float().contiguous(),
        cids.to(device=dev, dtype=torch.int32).contiguous(),
        nids.to(device=dev, dtype=torch.int32).contiguous(),
        leaf_order.to(device=dev, dtype=torch.int32).contiguous(),
        node_leaf_start.to(device=dev, dtype=torch.int32).contiguous(),
        node_leaf_count.to(device=dev, dtype=torch.int32).contiguous(),
        leaf_logz.float().contiguous(),
        leaf_grad.contiguous(),
        leaf_eval.float().contiguous(),
        leaf_modes.contiguous(),
        out,
        B=b, D=d, K=k, K_TILE=int(k_tile), BLOCK_D=block_d,
        num_warps=4,
    )
    return out



@triton.jit
def _htrc_target_h_apply_small8_range_kernel(
    X,
    parent_logz,
    parent_grad,
    cids,
    nids,
    leaf_order,
    node_leaf_start,
    node_leaf_count,
    leaf_logz,
    leaf_grad,
    leaf_small,
    leaf_q8,
    out,
    B,
    D: tl.constexpr,
    K: tl.constexpr,
    K_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Exact range H*X from shared Q8 + centroid-specific small8."""
    mid = tl.program_id(0)
    ktile = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    kk = ktile * K_TILE + tl.arange(0, K_TILE)
    dmask = d < D
    kmask = kk < K

    cid = tl.load(cids + mid).to(tl.int32)
    nid = tl.load(nids + mid).to(tl.int32)
    start = tl.load(node_leaf_start + nid).to(tl.int32)
    count = tl.load(node_leaf_count + nid).to(tl.int32)
    plz = tl.load(parent_logz + mid).to(tl.float32)

    xoff = (mid * D + d[:, None]) * K + kk[None, :]
    xv = tl.load(
        X + xoff,
        mask=dmask[:, None] & kmask[None, :],
        other=0.0,
    ).to(tl.float32)
    pg = tl.load(
        parent_grad + mid * D + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((BLOCK_D, K_TILE), dtype=tl.float32)
    rr = tl.arange(0, 8)

    for li in tl.range(0, count):
        bid = tl.load(leaf_order + start + li).to(tl.int32)
        base = cid * B + bid
        lz = tl.load(leaf_logz + base).to(tl.float32)
        w = tl.exp(lz - plz)

        gv = tl.load(
            leaf_grad + base * D + d,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        dg = gv - pg
        gd_dot = tl.sum(dg[:, None] * xv, axis=0)
        acc += w * dg[:, None] * gd_dot[None, :]

        # p = Q8^T X
        p = tl.zeros((8, K_TILE), dtype=tl.float32)
        for r in tl.static_range(0, 8):
            qv = tl.load(
                leaf_q8 + (bid * D + d) * 8 + r,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            pr = tl.sum(qv[:, None] * xv, axis=0)
            p = tl.where((rr[:, None] == r), pr[None, :], p)

        # z = small8 @ p
        z = tl.zeros((8, K_TILE), dtype=tl.float32)
        for r in tl.static_range(0, 8):
            sv = tl.load(
                leaf_small + (base * 8 + r) * 8 + rr
            ).to(tl.float32)
            zr = tl.sum(sv[:, None] * p, axis=0)
            z = tl.where((rr[:, None] == r), zr[None, :], z)

        # Q8 @ z
        for r in tl.static_range(0, 8):
            qv = tl.load(
                leaf_q8 + (bid * D + d) * 8 + r,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            zr = tl.sum(
                z * (rr[:, None] == r).to(tl.float32),
                axis=0,
            )
            acc += w * qv[:, None] * zr[None, :]

    tl.store(
        out + xoff,
        acc,
        mask=dmask[:, None] & kmask[None, :],
    )


def htrc_target_h_apply_small8_range(
    X: torch.Tensor,
    parent_logz: torch.Tensor,
    parent_grad: torch.Tensor,
    cids: torch.Tensor,
    nids: torch.Tensor,
    leaf_order: torch.Tensor,
    node_leaf_start: torch.Tensor,
    node_leaf_count: torch.Tensor,
    leaf_logz: torch.Tensor,
    leaf_grad: torch.Tensor,
    leaf_small: torch.Tensor,
    leaf_q8: torch.Tensor,
    *,
    k_tile: int = 4,
):
    """Exact range H*X using compact leaf small8 representation."""
    if X.ndim != 3:
        raise ValueError("HTRC small8 range H*X expects X=[M,D,K]")
    m, d, k = (int(x) for x in X.shape)
    if parent_logz.shape != (m,) or parent_grad.shape != (m, d):
        raise ValueError("HTRC small8 range parent state shape mismatch")
    ccount, b, ld = (int(x) for x in leaf_grad.shape)
    if ld != d or leaf_logz.shape != (ccount, b):
        raise ValueError("HTRC small8 range leaf state shape mismatch")
    if leaf_small.shape != (ccount, b, 8, 8):
        raise ValueError("HTRC small8 range leaf_small shape mismatch")
    if leaf_q8.shape != (b, d, 8):
        raise ValueError("HTRC small8 range leaf_q8 shape mismatch")
    if k_tile not in (2, 4):
        raise ValueError("HTRC small8 range k_tile must be 2 or 4")

    dev = X.device
    out = torch.empty((m, d, k), device=dev, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    ktiles = triton.cdiv(k, int(k_tile))
    _htrc_target_h_apply_small8_range_kernel[(m, ktiles)](
        X.float().contiguous(),
        parent_logz.float().contiguous(),
        parent_grad.float().contiguous(),
        cids.to(device=dev, dtype=torch.int32).contiguous(),
        nids.to(device=dev, dtype=torch.int32).contiguous(),
        leaf_order.to(device=dev, dtype=torch.int32).contiguous(),
        node_leaf_start.to(device=dev, dtype=torch.int32).contiguous(),
        node_leaf_count.to(device=dev, dtype=torch.int32).contiguous(),
        leaf_logz.float().contiguous(),
        leaf_grad.contiguous(),
        leaf_small.float().contiguous(),
        leaf_q8.contiguous(),
        out,
        B=b,
        D=d,
        K=k,
        K_TILE=int(k_tile),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@triton.jit
def _htrc_target_h_apply_tiled_counted_kernel(
    X,
    weights,
    parent_grad,
    cids,
    nids,
    node_leaf_idx,
    node_leaf_count,
    leaf_grad,
    leaf_eval,
    leaf_modes,
    out,
    B,
    L,
    D: tl.constexpr,
    K: tl.constexpr,
    K_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """K-tiled exact H*X using the real descendant-leaf count per node."""
    mid = tl.program_id(0)
    ktile = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    kk = ktile * K_TILE + tl.arange(0, K_TILE)
    dmask = d < D
    kmask = kk < K

    cid = tl.load(cids + mid).to(tl.int32)
    nid = tl.load(nids + mid).to(tl.int32)
    leaf_count = tl.load(node_leaf_count + nid).to(tl.int32)

    xoff = (mid * D + d[:, None]) * K + kk[None, :]
    xv = tl.load(
        X + xoff,
        mask=dmask[:, None] & kmask[None, :],
        other=0.0,
    ).to(tl.float32)
    pg = tl.load(
        parent_grad + mid * D + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_D, K_TILE), dtype=tl.float32)

    for li in tl.range(0, leaf_count):
        bid = tl.load(node_leaf_idx + nid * L + li).to(tl.int32)
        w = tl.load(weights + mid * L + li).to(tl.float32)

        leaf_base = cid * B + bid
        gv = tl.load(
            leaf_grad + leaf_base * D + d,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        dg = gv - pg
        gd_dot = tl.sum(dg[:, None] * xv, axis=0)
        acc += w * dg[:, None] * gd_dot[None, :]

        for r in tl.static_range(0, 8):
            uv = tl.load(
                leaf_modes + (leaf_base * D + d) * 8 + r,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            proj = tl.sum(uv[:, None] * xv, axis=0)
            lv = tl.load(leaf_eval + leaf_base * 8 + r).to(tl.float32)
            acc += w * uv[:, None] * lv * proj[None, :]

    tl.store(
        out + xoff,
        acc,
        mask=dmask[:, None] & kmask[None, :],
    )



@triton.jit
def _htrc_target_h_apply_small8_counted_kernel(
    X,
    weights,
    parent_grad,
    cids,
    nids,
    node_leaf_idx,
    node_leaf_count,
    leaf_grad,
    leaf_small,
    leaf_q8,
    out,
    B,
    L,
    D: tl.constexpr,
    K: tl.constexpr,
    K_TILE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Exact counted H*X using leaf H=Q8*small*Q8^T, no leaf eigensystem."""
    mid = tl.program_id(0)
    ktile = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    kk = ktile * K_TILE + tl.arange(0, K_TILE)
    dmask = d < D
    kmask = kk < K

    cid = tl.load(cids + mid).to(tl.int32)
    nid = tl.load(nids + mid).to(tl.int32)
    leaf_count = tl.load(node_leaf_count + nid).to(tl.int32)

    xoff = (mid * D + d[:, None]) * K + kk[None, :]
    xv = tl.load(
        X + xoff,
        mask=dmask[:, None] & kmask[None, :],
        other=0.0,
    ).to(tl.float32)
    pg = tl.load(
        parent_grad + mid * D + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_D, K_TILE), dtype=tl.float32)

    for li in tl.range(0, leaf_count):
        bid = tl.load(node_leaf_idx + nid * L + li).to(tl.int32)
        w = tl.load(weights + mid * L + li).to(tl.float32)

        leaf_base = cid * B + bid
        gv = tl.load(
            leaf_grad + leaf_base * D + d,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        dg = gv - pg
        gd_dot = tl.sum(dg[:, None] * xv, axis=0)
        acc += w * dg[:, None] * gd_dot[None, :]

        p0 = tl.zeros((8, K_TILE), dtype=tl.float32)
        for r in tl.static_range(0, 8):
            qv = tl.load(
                leaf_q8 + (bid * D + d) * 8 + r,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            pr = tl.sum(qv[:, None] * xv, axis=0)
            p0 = tl.where(
                (tl.arange(0, 8)[:, None] == r),
                pr[None, :],
                p0,
            )

        z = tl.zeros((8, K_TILE), dtype=tl.float32)
        rr = tl.arange(0, 8)
        for s in tl.static_range(0, 8):
            sv = tl.load(
                leaf_small + (leaf_base * 8 + s) * 8 + rr
            ).to(tl.float32)
            zs = tl.sum(sv[:, None] * p0, axis=0)
            z = tl.where(
                (tl.arange(0, 8)[:, None] == s),
                zs[None, :],
                z,
            )

        ss = tl.arange(0, 8)
        for s in tl.static_range(0, 8):
            qv = tl.load(
                leaf_q8 + (bid * D + d) * 8 + s,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            zs = tl.sum(
                z * (ss[:, None] == s).to(tl.float32),
                axis=0,
            )
            acc += w * qv[:, None] * zs[None, :]

    tl.store(
        out + xoff,
        acc,
        mask=dmask[:, None] & kmask[None, :],
    )


def htrc_target_h_apply_small8_counted(
    X: torch.Tensor,
    weights: torch.Tensor,
    parent_grad: torch.Tensor,
    cids: torch.Tensor,
    nids: torch.Tensor,
    node_leaf_idx: torch.Tensor,
    node_leaf_count: torch.Tensor,
    leaf_grad: torch.Tensor,
    leaf_small: torch.Tensor,
    leaf_q8: torch.Tensor,
    *,
    k_tile: int = 4,
) -> torch.Tensor:
    """Exact tiled H*X from the leaf 8x8 operator, without leaf eigh."""
    if X.ndim != 3:
        raise ValueError("HTRC small8 counted H*X expects X=[M,D,K]")
    m, d, k = (int(v) for v in X.shape)
    if k_tile not in (2, 4):
        raise ValueError("HTRC small8 counted H*X k_tile must be 2 or 4")
    if weights.ndim != 2 or int(weights.shape[0]) != m:
        raise ValueError("HTRC small8 counted H*X weights must be [M,L]")
    l = int(weights.shape[1])
    if parent_grad.shape != (m, d):
        raise ValueError("HTRC small8 counted H*X parent_grad must be [M,D]")
    if cids.shape != (m,) or nids.shape != (m,):
        raise ValueError("HTRC small8 counted H*X cids/nids must be [M]")
    if node_leaf_idx.ndim != 2 or int(node_leaf_idx.shape[1]) != l:
        raise ValueError("HTRC small8 counted H*X leaf-table width mismatch")
    n = int(node_leaf_idx.shape[0])
    if node_leaf_count.shape != (n,):
        raise ValueError("HTRC small8 counted H*X leaf-count shape mismatch")
    ccount, b, ld = (int(v) for v in leaf_grad.shape)
    if ld != d:
        raise ValueError("HTRC small8 counted H*X D mismatch")
    if leaf_small.shape != (ccount, b, 8, 8):
        raise ValueError("HTRC small8 leaf_small must be [C,B,8,8]")
    if leaf_q8.shape != (b, d, 8):
        raise ValueError("HTRC small8 leaf_q8 must be [B,D,8]")
    if X.device.type != "cuda":
        raise RuntimeError("HTRC small8 counted H*X is a CUDA research kernel")

    dev = X.device
    _ensure_windows_triton_dll_search()
    xc = X.float().contiguous()
    wc = weights.float().contiguous()
    pgc = parent_grad.float().contiguous()
    cc = cids.to(device=dev, dtype=torch.int32).contiguous()
    nc = nids.to(device=dev, dtype=torch.int32).contiguous()
    lic = node_leaf_idx.to(device=dev, dtype=torch.int32).contiguous()
    lcc = node_leaf_count.to(device=dev, dtype=torch.int32).contiguous()
    lgc = leaf_grad.contiguous()
    lsc = leaf_small.float().contiguous()
    lqc = leaf_q8.float().contiguous()
    out = torch.empty((m, d, k), device=dev, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    ktiles = triton.cdiv(k, int(k_tile))

    _htrc_target_h_apply_small8_counted_kernel[(m, ktiles)](
        xc, wc, pgc, cc, nc, lic, lcc, lgc, lsc, lqc, out,
        B=b,
        L=l,
        D=d,
        K=k,
        K_TILE=int(k_tile),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def htrc_target_h_apply_tiled_counted(
    X: torch.Tensor,
    weights: torch.Tensor,
    parent_grad: torch.Tensor,
    cids: torch.Tensor,
    nids: torch.Tensor,
    node_leaf_idx: torch.Tensor,
    node_leaf_count: torch.Tensor,
    leaf_grad: torch.Tensor,
    leaf_eval: torch.Tensor,
    leaf_modes: torch.Tensor,
    *,
    k_tile: int = 4,
) -> torch.Tensor:
    """Exact tiled H*X that skips padded descendant-leaf slots."""
    if X.ndim != 3:
        raise ValueError("HTRC counted tiled H*X expects X=[M,D,K]")
    m, d, k = (int(v) for v in X.shape)
    if k_tile not in (2, 4):
        raise ValueError("HTRC counted tiled H*X k_tile must be 2 or 4")
    if weights.ndim != 2 or int(weights.shape[0]) != m:
        raise ValueError("HTRC counted tiled H*X weights must be [M,L]")
    l = int(weights.shape[1])
    if parent_grad.shape != (m, d):
        raise ValueError("HTRC counted tiled H*X parent_grad must be [M,D]")
    if cids.shape != (m,) or nids.shape != (m,):
        raise ValueError("HTRC counted tiled H*X cids/nids must be [M]")
    if node_leaf_idx.ndim != 2 or int(node_leaf_idx.shape[1]) != l:
        raise ValueError("HTRC counted tiled H*X leaf-table width mismatch")
    n = int(node_leaf_idx.shape[0])
    if node_leaf_count.shape != (n,):
        raise ValueError("HTRC counted tiled H*X leaf-count shape mismatch")
    ccount, b, ld = (int(v) for v in leaf_grad.shape)
    if ld != d:
        raise ValueError("HTRC counted tiled H*X D mismatch")
    if leaf_eval.shape != (ccount, b, 8):
        raise ValueError("HTRC counted tiled H*X leaf_eval must be [C,B,8]")
    if leaf_modes.shape != (ccount, b, d, 8):
        raise ValueError("HTRC counted tiled H*X leaf_modes must be [C,B,D,8]")

    if X.device.type != "cuda":
        raise RuntimeError("HTRC counted tiled H*X is a CUDA research kernel")

    dev = X.device
    _ensure_windows_triton_dll_search()
    xc = X.float().contiguous()
    wc = weights.float().contiguous()
    pgc = parent_grad.float().contiguous()
    cc = cids.to(device=dev, dtype=torch.int32).contiguous()
    nc = nids.to(device=dev, dtype=torch.int32).contiguous()
    lic = node_leaf_idx.to(device=dev, dtype=torch.int32).contiguous()
    lcc = node_leaf_count.to(device=dev, dtype=torch.int32).contiguous()
    lgc = leaf_grad.contiguous()
    lec = leaf_eval.float().contiguous()
    luc = leaf_modes.contiguous()
    out = torch.empty((m, d, k), device=dev, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    ktiles = triton.cdiv(k, int(k_tile))

    _htrc_target_h_apply_tiled_counted_kernel[(m, ktiles)](
        xc, wc, pgc, cc, nc, lic, lcc, lgc, lec, luc, out,
        B=b,
        L=l,
        D=d,
        K=k,
        K_TILE=int(k_tile),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def htrc_target_h_apply_tiled(
    X: torch.Tensor,
    weights: torch.Tensor,
    parent_grad: torch.Tensor,
    cids: torch.Tensor,
    nids: torch.Tensor,
    node_leaf_idx: torch.Tensor,
    node_leaf_mask: torch.Tensor,
    leaf_grad: torch.Tensor,
    leaf_eval: torch.Tensor,
    leaf_modes: torch.Tensor,
    *,
    k_tile: int = 4,
) -> torch.Tensor:
    """Tiled H*X kernel that reuses leaf state across K columns."""
    if X.ndim != 3:
        raise ValueError("HTRC tiled H*X expects X=[M,D,K]")
    m, d, k = (int(v) for v in X.shape)
    if k_tile not in (2, 4):
        raise ValueError("HTRC tiled H*X k_tile must be 2 or 4")
    if weights.ndim != 2 or int(weights.shape[0]) != m:
        raise ValueError("HTRC tiled H*X weights must be [M,L]")
    l = int(weights.shape[1])
    if parent_grad.shape != (m, d):
        raise ValueError("HTRC tiled H*X parent_grad must be [M,D]")
    if cids.shape != (m,) or nids.shape != (m,):
        raise ValueError("HTRC tiled H*X cids/nids must be [M]")
    if node_leaf_idx.ndim != 2 or node_leaf_mask.shape != node_leaf_idx.shape:
        raise ValueError("HTRC tiled H*X node leaf table shape mismatch")
    n = int(node_leaf_idx.shape[0])
    if int(node_leaf_idx.shape[1]) != l:
        raise ValueError("HTRC tiled H*X leaf-table width mismatch")
    ccount, b, ld = (int(v) for v in leaf_grad.shape)
    if ld != d:
        raise ValueError("HTRC tiled H*X D mismatch")
    if leaf_eval.shape != (ccount, b, 8):
        raise ValueError("HTRC tiled H*X leaf_eval must be [C,B,8]")
    if leaf_modes.shape != (ccount, b, d, 8):
        raise ValueError("HTRC tiled H*X leaf_modes must be [C,B,D,8]")
    if X.device.type != "cuda":
        return htrc_target_h_apply(
            X, weights, parent_grad, cids, nids, node_leaf_idx,
            node_leaf_mask, leaf_grad, leaf_eval, leaf_modes
        )

    dev = X.device
    _ensure_windows_triton_dll_search()
    xc = X.float().contiguous()
    wc = weights.float().contiguous()
    pgc = parent_grad.float().contiguous()
    cc = cids.to(device=dev, dtype=torch.int32).contiguous()
    nc = nids.to(device=dev, dtype=torch.int32).contiguous()
    lic = node_leaf_idx.to(device=dev, dtype=torch.int32).contiguous()
    lmc = node_leaf_mask.to(device=dev, dtype=torch.int8).contiguous()
    lgc = leaf_grad.contiguous()
    lec = leaf_eval.float().contiguous()
    luc = leaf_modes.contiguous()
    out = torch.empty((m, d, k), device=dev, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    ktiles = triton.cdiv(k, int(k_tile))

    _htrc_target_h_apply_tiled_kernel[(m, ktiles)](
        xc, wc, pgc, cc, nc, lic, lmc, lgc, lec, luc, out,
        B=b, N=n, L=l, D=d, K=k, K_TILE=int(k_tile),
        BLOCK_D=block_d, num_warps=4,
    )
    return out


@triton.jit
def _htrc_target_h_apply_kernel(
    X,
    weights,
    parent_grad,
    cids,
    nids,
    node_leaf_idx,
    node_leaf_mask,
    leaf_grad,
    leaf_eval,
    leaf_modes,
    out,
    B: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Matrix-free target-node Hessian action for one (target,k) program.

    Computes
      H_N x = sum_l w_l U_l diag(lambda_l) U_l^T x
            + sum_l w_l dg_l dg_l^T x
    directly from global leaf sufficient state without materializing gathered
    [M,L,D,8] tensors or dense [D,D] Hessians.
    """
    mid = tl.program_id(0)
    kid = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dmask = d < D

    cid = tl.load(cids + mid).to(tl.int32)
    nid = tl.load(nids + mid).to(tl.int32)
    xv = tl.load(
        X + (mid * D + d) * K + kid,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)
    pg = tl.load(
        parent_grad + mid * D + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # L is runtime-shape-specialized but must NOT be statically unrolled:
    # root descendant width is tens of leaves, and unrolling L * rank8 creates
    # a huge compiler IR on Windows. Keep only the fixed rank-8 loop unrolled.
    for li in tl.range(0, L):
        valid_leaf = tl.load(
            node_leaf_mask + nid * L + li
        ).to(tl.int1)
        bid = tl.load(
            node_leaf_idx + nid * L + li,
            mask=valid_leaf,
            other=0,
        ).to(tl.int32)
        w = tl.load(
            weights + mid * L + li,
            mask=valid_leaf,
            other=0.0,
        ).to(tl.float32)

        leaf_base = cid * B + bid
        gv = tl.load(
            leaf_grad + leaf_base * D + d,
            mask=dmask & valid_leaf,
            other=0.0,
        ).to(tl.float32)
        dg = gv - pg
        gd_dot = tl.sum(dg * xv, axis=0)
        acc += w * dg * gd_dot

        for r in tl.static_range(0, 8):
            uv = tl.load(
                leaf_modes + (leaf_base * D + d) * 8 + r,
                mask=dmask & valid_leaf,
                other=0.0,
            ).to(tl.float32)
            proj = tl.sum(uv * xv, axis=0)
            lv = tl.load(
                leaf_eval + leaf_base * 8 + r,
                mask=valid_leaf,
                other=0.0,
            ).to(tl.float32)
            acc += w * uv * lv * proj

    tl.store(
        out + (mid * D + d) * K + kid,
        acc,
        mask=dmask,
    )


def htrc_target_h_apply(
    X: torch.Tensor,
    weights: torch.Tensor,
    parent_grad: torch.Tensor,
    cids: torch.Tensor,
    nids: torch.Tensor,
    node_leaf_idx: torch.Tensor,
    node_leaf_mask: torch.Tensor,
    leaf_grad: torch.Tensor,
    leaf_eval: torch.Tensor,
    leaf_modes: torch.Tensor,
) -> torch.Tensor:
    """Fused matrix-free H*X for batched HTRC target-node states.

    X: [M,D,K]
    weights/node_leaf_*: target descendant-leaf authority
    leaf_*: global [C,B,...] leaf sufficient state.
    """
    if X.ndim != 3:
        raise ValueError("HTRC target H*X expects X=[M,D,K]")
    m, d, k = (int(v) for v in X.shape)
    if weights.ndim != 2 or int(weights.shape[0]) != m:
        raise ValueError("HTRC target H*X weights must be [M,L]")
    l = int(weights.shape[1])
    if parent_grad.shape != (m, d):
        raise ValueError("HTRC target H*X parent_grad must be [M,D]")
    if cids.shape != (m,) or nids.shape != (m,):
        raise ValueError("HTRC target H*X cids/nids must be [M]")
    if node_leaf_idx.ndim != 2 or node_leaf_mask.shape != node_leaf_idx.shape:
        raise ValueError("HTRC target H*X node leaf table shape mismatch")
    n = int(node_leaf_idx.shape[0])
    if int(node_leaf_idx.shape[1]) != l:
        raise ValueError("HTRC target H*X leaf-table width mismatch")
    if leaf_grad.ndim != 3:
        raise ValueError("HTRC target H*X leaf_grad must be [C,B,D]")
    ccount, b, ld = (int(v) for v in leaf_grad.shape)
    if ld != d:
        raise ValueError("HTRC target H*X D mismatch")
    if leaf_eval.shape != (ccount, b, 8):
        raise ValueError("HTRC target H*X leaf_eval must be [C,B,8]")
    if leaf_modes.shape != (ccount, b, d, 8):
        raise ValueError("HTRC target H*X leaf_modes must be [C,B,D,8]")
    if k <= 0 or k > 32:
        raise ValueError(f"HTRC target H*X supports K=1..32, got {k}")
    if d <= 0 or d > 256:
        raise ValueError(f"HTRC target H*X supports D=1..256, got {d}")
    if l <= 0 or l > 128:
        raise ValueError(f"HTRC target H*X supports L=1..128, got {l}")

    if X.device.type != "cuda":
        # Reference fallback for source/preflight tests.
        ids = node_leaf_idx.index_select(0, nids.to(torch.long))
        mask = node_leaf_mask.index_select(0, nids.to(torch.long))
        lg = leaf_grad[cids[:, None].to(torch.long), ids]
        le = leaf_eval[cids[:, None].to(torch.long), ids]
        lu = leaf_modes[cids[:, None].to(torch.long), ids]
        dg = lg.float() - parent_grad.float()[:, None, :]
        w = weights.float() * mask.to(weights.dtype)
        p = torch.einsum("mldr,mdk->mlrk", lu.float(), X.float())
        leaf = torch.einsum("ml,mldr,mlr,mlrk->mdk", w, lu.float(), le.float(), p)
        gd = torch.einsum("mld,mdk->mlk", dg, X.float())
        cov = torch.einsum("mld,ml,mlk->mdk", dg, w, gd)
        return leaf + cov

    dev = X.device
    tensors = (
        weights, parent_grad, cids, nids, node_leaf_idx, node_leaf_mask,
        leaf_grad, leaf_eval, leaf_modes,
    )
    if any(t.device != dev for t in tensors):
        raise ValueError("HTRC target H*X tensors must share one device")

    _ensure_windows_triton_dll_search()
    xc = X.float().contiguous()
    wc = weights.float().contiguous()
    pgc = parent_grad.float().contiguous()
    cc = cids.to(device=dev, dtype=torch.int32).contiguous()
    nc = nids.to(device=dev, dtype=torch.int32).contiguous()
    lic = node_leaf_idx.to(device=dev, dtype=torch.int32).contiguous()
    lmc = node_leaf_mask.to(device=dev, dtype=torch.int8).contiguous()
    lgc = leaf_grad.contiguous()
    lec = leaf_eval.float().contiguous()
    luc = leaf_modes.contiguous()
    out = torch.empty((m, d, k), device=dev, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)

    _htrc_target_h_apply_kernel[(m, k)](
        xc,
        wc,
        pgc,
        cc,
        nc,
        lic,
        lmc,
        lgc,
        lec,
        luc,
        out,
        B=b,
        N=n,
        L=l,
        D=d,
        K=k,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@triton.jit
def _htrc_node_terms(
    dq,
    logz,
    grad,
    lam,
    U,
    trace,
    captured_trace,
    cid,
    nid,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d = tl.arange(0, BLOCK_D)
    dmask = d < D
    base_node = cid * N + nid
    gv = tl.load(grad + base_node * D + d, mask=dmask, other=0.0).to(tl.float32)
    linear = tl.sum(dq * gv, axis=0)
    q2 = 0.0
    proj_norm2 = 0.0
    for r in tl.static_range(0, 8):
        up = U + (base_node * D * 8) + d * 8 + r
        uv = tl.load(up, mask=dmask, other=0.0).to(tl.float32)
        pr = tl.sum(dq * uv, axis=0)
        lv = tl.load(lam + base_node * 8 + r).to(tl.float32)
        q2 += lv * pr * pr
        proj_norm2 += pr * pr
    lz = tl.load(logz + base_node).to(tl.float32)
    tr = tl.load(trace + base_node).to(tl.float32)
    ctr = tl.load(captured_trace + base_node).to(tl.float32)
    route_pred = lz + linear + 0.5 * q2
    dq_norm2 = tl.sum(dq * dq, axis=0)
    residual_norm2 = tl.maximum(dq_norm2 - proj_norm2, 0.0)
    tail = (tr - ctr) * residual_norm2 / float(D - 8)
    accepted_mass = route_pred + 1.10 * tail
    cert = tl.abs(0.5 * q2)
    return route_pred, accepted_mass, cert


@triton.jit
def _htrc_tree_attention_kernel(
    q,
    assign,
    centers,
    logz,
    grad,
    value,
    lam,
    U,
    trace,
    captured_trace,
    root_ids,
    subtree_end,
    leaf_block,
    out,
    tau,
    N: tl.constexpr,
    ROOTS: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per query for one head, using preorder HTRC subtree skipping."""
    qi = tl.program_id(0)
    cid = tl.load(assign + qi).to(tl.int32)
    d = tl.arange(0, BLOCK_D)
    dmask = d < D
    qv = tl.load(q + qi * D + d, mask=dmask, other=0.0).to(tl.float32)
    cv = tl.load(
        centers + cid * D + d, mask=dmask, other=0.0
    ).to(tl.float32)
    dq = qv - cv

    # Rank8 root prediction is the routing mass authority.
    root_m = -float("inf")
    root_z = 0.0
    for ri in tl.range(0, ROOTS):
        nid = tl.load(root_ids + ri).to(tl.int32)
        rp, _am, _cert = _htrc_node_terms(
            dq, logz, grad, lam, U, trace, captured_trace,
            cid, nid, N=N, D=D, BLOCK_D=BLOCK_D,
        )
        nm = tl.maximum(root_m, rp)
        root_z = root_z * tl.exp(root_m - nm) + tl.exp(rp - nm)
        root_m = nm
    log_total = root_m + tl.log(tl.maximum(root_z, 1.0e-20))

    # DFS preorder needs no explicit stack because subtree_end skips an accepted
    # node's complete descendant range.
    m = -float("inf")
    z = 0.0
    numer = tl.zeros((BLOCK_D,), dtype=tl.float32)
    nid = 0
    while nid < N:
        route_pred, accepted_mass, cert = _htrc_node_terms(
            dq, logz, grad, lam, U, trace, captured_trace,
            cid, nid, N=N, D=D, BLOCK_D=BLOCK_D,
        )
        leaf = tl.load(leaf_block + nid).to(tl.int32) >= 0

        # log(expm1(cert)) with stable large-cert branch.
        small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
        log_unc = tl.where(cert > 20.0, cert, small)
        score = route_pred - log_total + log_unc
        accept = leaf | (score <= tau)

        if accept:
            vv = tl.load(
                value + (cid * N + nid) * D + d,
                mask=dmask,
                other=0.0,
            ).to(tl.float32)
            nm = tl.maximum(m, accepted_mass)
            old_scale = tl.exp(m - nm)
            new_scale = tl.exp(accepted_mass - nm)
            numer = numer * old_scale + vv * new_scale
            z = z * old_scale + new_scale
            m = nm
            nid = tl.load(subtree_end + nid).to(tl.int32)
        else:
            nid += 1

    outv = numer / tl.maximum(z, 1.0e-20)
    tl.store(out + qi * D + d, outv, mask=dmask)


def htrc_tree_attention(
    q: torch.Tensor,
    assign: torch.Tensor,
    *,
    centers: torch.Tensor,
    logz: torch.Tensor,
    grad: torch.Tensor,
    value: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    root_ids: torch.Tensor,
    subtree_end: torch.Tensor,
    leaf_block: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Fused HTRC preorder traversal for one attention head."""
    if q.ndim != 2:
        raise ValueError("HTRC evaluator expects q=[Q,D] for one head")
    qn, d = (int(v) for v in q.shape)
    ccount, n = (int(logz.shape[0]), int(logz.shape[1]))
    roots = int(root_ids.numel())
    if grad.shape != (ccount, n, d):
        raise ValueError("HTRC grad shape mismatch")
    if value.shape != (ccount, n, d):
        raise ValueError("HTRC value shape mismatch")
    if lam.shape != (ccount, n, 8):
        raise ValueError("HTRC lam shape mismatch")
    if U.shape != (ccount, n, d, 8):
        raise ValueError("HTRC U shape mismatch")
    if trace.shape != (ccount, n) or captured_trace.shape != (ccount, n):
        raise ValueError("HTRC trace shape mismatch")
    if assign.shape != (qn,):
        raise ValueError("HTRC assign must be [Q]")
    if roots <= 0 or roots > 128:
        raise ValueError(f"HTRC root count unsupported: {roots}")
    if q.device.type != "cuda":
        raise RuntimeError("live HTRC traversal requires CUDA")

    _ensure_windows_triton_dll_search()
    qc = q.float().contiguous()
    ac = assign.to(device=q.device, dtype=torch.int32).contiguous()
    cc = centers.float().contiguous()
    lzc = logz.float().contiguous()
    gc = grad.float().contiguous()
    vc = value.float().contiguous()
    lc = lam.float().contiguous()
    uc = U.float().contiguous()
    tc = trace.float().contiguous()
    ctc = captured_trace.float().contiguous()
    rc = root_ids.to(device=q.device, dtype=torch.int32).contiguous()
    se = subtree_end.to(device=q.device, dtype=torch.int32).contiguous()
    lb = leaf_block.to(device=q.device, dtype=torch.int32).contiguous()
    out = torch.empty((qn, d), dtype=torch.float32, device=q.device)
    block_d = triton.next_power_of_2(d)
    _htrc_tree_attention_kernel[(qn,)](
        qc, ac, cc, lzc, gc, vc, lc, uc, tc, ctc,
        rc, se, lb, out, float(tau),
        N=n,
        ROOTS=roots,
        C=ccount,
        D=d,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@triton.jit
def _htrc_frontier_local_scan_kernel(
    open_mask,
    root_ids,
    child_count,
    local_base,
    query_count,
    Q: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Per-query exclusive scan over only R roots."""
    qi = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    mask = r < R
    opened = tl.load(
        open_mask + qi * R + r, mask=mask, other=0
    ).to(tl.int32)
    rid = tl.load(root_ids + r, mask=mask, other=0).to(tl.int32)
    cnt = tl.load(
        child_count + rid, mask=mask, other=0
    ).to(tl.int32)
    x = tl.where(opened != 0, cnt, 0)
    inc = tl.cumsum(x, axis=0)
    exc = inc - x
    tl.store(local_base + qi * R + r, exc, mask=mask)
    total = tl.sum(x, axis=0)
    tl.store(query_count + qi, total)


@triton.jit
def _htrc_frontier_scatter_kernel(
    open_mask,
    root_ids,
    child_idx,
    child_count,
    local_base,
    query_base,
    out_query,
    out_node,
    Q: tl.constexpr,
    R: tl.constexpr,
    FANOUT: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Scatter children using per-query local and global query offsets."""
    qi = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    mask = r < R
    opened = tl.load(
        open_mask + qi * R + r, mask=mask, other=0
    ).to(tl.int32)
    rid = tl.load(root_ids + r, mask=mask, other=0).to(tl.int32)
    cnt = tl.load(
        child_count + rid, mask=mask, other=0
    ).to(tl.int32)
    lb = tl.load(local_base + qi * R + r, mask=mask, other=0).to(tl.int64)
    qb = tl.load(query_base + qi).to(tl.int64)

    for j in tl.static_range(0, FANOUT):
        dst = qb + lb + j
        take = mask & (opened != 0) & (j < cnt) & (dst < CAPACITY)
        nid = tl.load(
            child_idx + rid * FANOUT + j,
            mask=take,
            other=-1,
        ).to(tl.int32)
        tl.store(out_query + dst, qi, mask=take)
        tl.store(out_node + dst, nid, mask=take)


@triton.jit
def _htrc_frontier_query_count_kernel(
    open_mask,
    root_ids,
    child_count,
    query_count,
    Q: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    qi = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    mask = r < R
    opened = tl.load(open_mask + qi * R + r, mask=mask, other=0).to(tl.int32)
    rid = tl.load(root_ids + r, mask=mask, other=0).to(tl.int32)
    cnt = tl.load(child_count + rid, mask=mask, other=0).to(tl.int32)
    x = tl.where(opened != 0, cnt, 0)
    tl.store(query_count + qi, tl.sum(x, axis=0))


@triton.jit
def _htrc_frontier_scatter_recompute_kernel(
    open_mask,
    root_ids,
    child_idx,
    child_count,
    query_base,
    out_query,
    out_node,
    Q: tl.constexpr,
    R: tl.constexpr,
    FANOUT: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Scatter children while recomputing the short per-query root prefix."""
    qi = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    mask = r < R
    opened = tl.load(open_mask + qi * R + r, mask=mask, other=0).to(tl.int32)
    rid = tl.load(root_ids + r, mask=mask, other=0).to(tl.int32)
    cnt = tl.load(child_count + rid, mask=mask, other=0).to(tl.int32)
    x = tl.where(opened != 0, cnt, 0)
    inc = tl.cumsum(x, axis=0)
    lb = (inc - x).to(tl.int64)
    qb = tl.load(query_base + qi).to(tl.int64)

    for j in tl.static_range(0, FANOUT):
        dst = qb + lb + j
        take = mask & (opened != 0) & (j < cnt) & (dst < CAPACITY)
        nid = tl.load(
            child_idx + rid * FANOUT + j,
            mask=take,
            other=-1,
        ).to(tl.int32)
        tl.store(out_query + dst, qi, mask=take)
        tl.store(out_node + dst, nid, mask=take)


def htrc_frontier_compact(
    open_mask: torch.Tensor,
    root_ids: torch.Tensor,
    child_idx: torch.Tensor,
    child_count: torch.Tensor,
    *,
    capacity: int | None = None,
):
    """Two-stage GPU frontier compaction without atomics or global root scan.

    Stage 1 scans only R roots independently for each query. A short cumsum
    across Q query counts then supplies global offsets. Stage 2 scatters the
    actual child pairs.
    """
    if open_mask.ndim != 2:
        raise ValueError("HTRC frontier open_mask must be [Q,R]")
    qn, roots = (int(v) for v in open_mask.shape)
    if root_ids.shape != (roots,):
        raise ValueError("HTRC root_ids shape mismatch")
    if child_idx.ndim != 2:
        raise ValueError("HTRC child_idx must be [N,F]")
    fanout = int(child_idx.shape[1])
    if fanout <= 0 or fanout > 8:
        raise ValueError(f"HTRC frontier fanout unsupported: {fanout}")
    if child_count.ndim != 1 or int(child_count.shape[0]) != int(child_idx.shape[0]):
        raise ValueError("HTRC child_count shape mismatch")
    if open_mask.device.type != "cuda":
        raise RuntimeError("HTRC frontier compaction requires CUDA")

    dev = open_mask.device
    # open_mask is already a contiguous CUDA bool in the streaming path.
    # Keep it in-place; converting the full [Q,R] mask to int8 only duplicates
    # the same one-byte storage.
    om = open_mask.contiguous()
    rid = root_ids.to(device=dev, dtype=torch.int32).contiguous()
    ci = child_idx.to(device=dev, dtype=torch.int32).contiguous()
    cc = child_count.to(device=dev, dtype=torch.int32).contiguous()

    # Keep only per-query totals. The old local_base[Q,R] int32 matrix is
    # recomputed from the short R<=80 root vector in the scatter kernel.
    query_count = torch.empty((qn,), device=dev, dtype=torch.int32)
    block_r = triton.next_power_of_2(roots)
    _htrc_frontier_query_count_kernel[(qn,)](
        om, rid, cc, query_count,
        Q=qn, R=roots, BLOCK_R=block_r, num_warps=4,
    )

    query_prefix = torch.cumsum(query_count, dim=0, dtype=torch.int64)
    query_base = query_prefix - query_count.to(torch.int64)
    total = query_prefix[-1:].to(torch.int32)

    # For the production streaming path, allocate exactly after the tiny
    # per-query prefix is known. This removes the caller-side full-mask
    # reduction/workspace used only to predict capacity.
    if capacity is None:
        capacity = max(int(total.item()), 1)
    capacity = int(capacity)
    out_q = torch.empty((capacity,), device=dev, dtype=torch.int32)
    out_n = torch.empty((capacity,), device=dev, dtype=torch.int32)

    _htrc_frontier_scatter_recompute_kernel[(qn,)](
        om, rid, ci, cc, query_base, out_q, out_n,
        Q=qn, R=roots, FANOUT=fanout, CAPACITY=capacity,
        BLOCK_R=block_r, num_warps=4,
    )
    return out_q, out_n, total


@triton.jit
def _htrc_frontier_expand_kernel(
    frontier_q,
    frontier_node,
    frontier_count,
    next_count,
    prefix_base,
    child_idx,
    out_q,
    out_node,
    N: tl.constexpr,
    FANOUT: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    """Expand an already-evaluated compact frontier into its open children."""
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    cnt = tl.load(next_count + fi, mask=active, other=0).to(tl.int32)
    base = tl.load(prefix_base + fi, mask=active, other=0).to(tl.int64)
    valid_node = active & (nid >= 0) & (nid < N)
    for j in tl.static_range(0, FANOUT):
        dst = base + j
        take = valid_node & (j < cnt) & (dst < CAPACITY)
        child = tl.load(
            child_idx + nid * FANOUT + j,
            mask=take,
            other=-1,
        ).to(tl.int32)
        tl.store(out_q + dst, qi, mask=take)
        tl.store(out_node + dst, child, mask=take)


def htrc_frontier_expand(
    frontier_q: torch.Tensor,
    frontier_node: torch.Tensor,
    frontier_count: torch.Tensor,
    next_count: torch.Tensor,
    child_idx: torch.Tensor,
    *,
    capacity: int | None,
    active_only: bool = False,
):
    """Bounded GPU expansion for arbitrary compact frontier levels.

    When capacity is None, derive the exact allocation size from the prefix
    already required for expansion instead of requiring a separate caller-side
    next-count reduction.
    """
    if capacity is not None:
        capacity = int(capacity)
    if child_idx.ndim != 2:
        raise ValueError("HTRC child_idx must be [N,F]")
    fanout = int(child_idx.shape[1])
    if fanout <= 0 or fanout > 8:
        raise ValueError(f"HTRC frontier fanout unsupported: {fanout}")
    dev = frontier_q.device
    fq = frontier_q.to(device=dev, dtype=torch.int32).contiguous()
    fn = frontier_node.to(device=dev, dtype=torch.int32).contiguous()
    nc = next_count.to(device=dev, dtype=torch.int32).contiguous()
    fc = frontier_count.to(device=dev, dtype=torch.int32).contiguous()
    ci = child_idx.to(device=dev, dtype=torch.int32).contiguous()

    if active_only:
        # Caller guarantees next_count contains exactly the active frontier.
        # Avoid slots/where/zeros_like tail masking and scan only live rows.
        prefix = torch.cumsum(nc, dim=0, dtype=torch.int64)
        base = prefix - nc.to(torch.int64)
    else:
        # Generic safe path for fixed-capacity callers with a dirty tail.
        slots = torch.arange(nc.numel(), device=dev, dtype=torch.int32)
        nc_active = torch.where(slots < fc[0], nc, torch.zeros_like(nc))
        prefix = torch.cumsum(nc_active, dim=0, dtype=torch.int64)
        base = prefix - nc_active.to(torch.int64)
    total = prefix[-1:].to(torch.int32)

    if capacity is None:
        capacity = max(int(total.item()), 1)
    capacity = int(capacity)
    out_q = torch.empty((capacity,), device=dev, dtype=torch.int32)
    out_n = torch.empty((capacity,), device=dev, dtype=torch.int32)
    _htrc_frontier_expand_kernel[(int(nc.numel()),)](
        fq, fn, fc, nc, base, ci, out_q, out_n,
        N=int(ci.shape[0]), FANOUT=fanout, CAPACITY=capacity,
        num_warps=1,
    )
    return out_q, out_n, total


@triton.jit
def _htrc_frontier_level_max_kernel(
    frontier_q,
    frontier_count,
    accepted_mass,
    open_mask,
    level_max,
    CAPACITY: tl.constexpr,
):
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    opened = tl.load(open_mask + fi, mask=active, other=1).to(tl.int1)
    take = active & (~opened)
    mass = tl.load(
        accepted_mass + fi, mask=take, other=-float("inf")
    ).to(tl.float32)
    tl.atomic_max(level_max + qi, mass, mask=take)


@triton.jit
def _htrc_frontier_level_value_kernel(
    frontier_q,
    frontier_node,
    frontier_count,
    accepted_mass,
    open_mask,
    value,
    level_max,
    level_n,
    N: tl.constexpr,
    D: tl.constexpr,
    ROWS_PER_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    opened = tl.load(open_mask + fi, mask=active, other=1).to(tl.int1)
    take = active & (~opened) & (nid >= 0) & (nid < N)

    mass = tl.load(accepted_mass + fi, mask=take, other=-float("inf")).to(tl.float32)
    lm = tl.load(level_max + qi, mask=take, other=0.0).to(tl.float32)
    w = tl.exp(mass - lm)

    cid = qi // ROWS_PER_C
    d = tl.arange(0, BLOCK_D)
    dmask = take & (d < D)
    vv = tl.load(
        value + (cid * N + nid) * D + d,
        mask=dmask,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(level_n + qi * D + d, w * vv, mask=dmask)


@triton.jit
def _htrc_frontier_level_z_kernel(
    frontier_q,
    frontier_count,
    accepted_mass,
    open_mask,
    level_max,
    level_z,
    CAPACITY: tl.constexpr,
):
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    opened = tl.load(open_mask + fi, mask=active, other=1).to(tl.int1)
    take = active & (~opened)
    mass = tl.load(
        accepted_mass + fi, mask=take, other=-float("inf")
    ).to(tl.float32)
    lm = tl.load(level_max + qi, mask=take, other=0.0).to(tl.float32)
    w = tl.exp(mass - lm)
    tl.atomic_add(level_z + qi, w, mask=take)


def htrc_frontier_level_value(
    frontier_q: torch.Tensor,
    frontier_node: torch.Tensor,
    frontier_count: torch.Tensor,
    accepted_mass: torch.Tensor,
    open_mask: torch.Tensor,
    *,
    value: torch.Tensor,
    rows_per_centroid: int,
    qpad: int,
):
    """Reduce accepted compact-frontier nodes to per-query log-sum/value terms."""
    capacity = int(frontier_q.numel())
    if frontier_node.shape != (capacity,):
        raise ValueError("HTRC frontier node shape mismatch")
    if accepted_mass.shape != (capacity,) or open_mask.shape != (capacity,):
        raise ValueError("HTRC frontier accepted/open shape mismatch")
    ccount, n, d = (int(v) for v in value.shape)
    dev = frontier_q.device
    fq = frontier_q.to(device=dev, dtype=torch.int32).contiguous()
    fn = frontier_node.to(device=dev, dtype=torch.int32).contiguous()
    fc = frontier_count.to(device=dev, dtype=torch.int32).contiguous()
    mass = accepted_mass.float().contiguous()
    opened = open_mask.to(device=dev, dtype=torch.int8).contiguous()
    vc = value.contiguous()

    level_max = torch.full(
        (int(qpad),), -float("inf"), dtype=torch.float32, device=dev
    )
    level_z = torch.zeros((int(qpad),), dtype=torch.float32, device=dev)
    level_n = torch.zeros((int(qpad), d), dtype=torch.float32, device=dev)

    _htrc_frontier_level_max_kernel[(capacity,)](
        fq, fc, mass, opened, level_max,
        CAPACITY=capacity, num_warps=1,
    )
    block_d = triton.next_power_of_2(d)
    _htrc_frontier_level_z_kernel[(capacity,)](
        fq, fc, mass, opened, level_max, level_z,
        CAPACITY=capacity, num_warps=1,
    )
    _htrc_frontier_level_value_kernel[(capacity,)](
        fq, fn, fc, mass, opened, vc,
        level_max, level_n,
        N=n, D=d, ROWS_PER_C=int(rows_per_centroid),
        BLOCK_D=block_d, num_warps=4,
    )
    return level_max, level_z, level_n


@triton.jit
def _htrc_frontier_eval_kernel(
    dq,
    frontier_q,
    frontier_node,
    frontier_count,
    log_total,
    logz,
    grad,
    lam,
    U,
    trace,
    captured_trace,
    out_route,
    out_mass,
    out_score,
    CAPACITY: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Evaluate compacted child frontier with a fixed-capacity launch."""
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & active
    dqv = tl.load(dq + qi * D + d, mask=dmask, other=0.0).to(tl.float32)

    # q is padded/grouped by centroid; state rows are already flattened
    # [C,N,...] and caller supplies qi->centroid through frontier_q layout.
    # The centroid index is stored in the high-level packed row mapping:
    # qi / rows_per_centroid is passed implicitly via ROWS_PER_C below.
    # Kept separate in the wrapper to avoid dynamic tensor allocation.
    tl.store(out_route + fi, 0.0, mask=active)
    tl.store(out_mass + fi, 0.0, mask=active)
    tl.store(out_score + fi, 0.0, mask=active)


@triton.jit
def _htrc_frontier_step_int8_grouped_kernel(
    dq,
    frontier_q,
    frontier_node,
    frontier_count,
    log_total,
    logz,
    grad_q,
    grad_scale,
    lam,
    U_q,
    U_scale,
    trace,
    captured_trace,
    child_count,
    accepted_mass_out,
    open_out,
    next_count_out,
    tau,
    N: tl.constexpr,
    D: tl.constexpr,
    ROWS_PER_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Storage-only signed INT8 frontier step with selective on-the-fly decode."""
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    cid = qi // ROWS_PER_C
    base_node = cid * N + nid

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & active
    dqv = tl.load(dq + qi * D + d, mask=dmask, other=0.0).to(tl.float32)

    gq = tl.load(
        grad_q + base_node * D + d, mask=dmask, other=0
    ).to(tl.float32)
    gs = tl.load(grad_scale + base_node, mask=active, other=0.0).to(tl.float32)
    gv = gq * gs
    linear = tl.sum(dqv * gv, axis=0)

    q2 = 0.0
    proj_norm2 = 0.0
    for r in tl.static_range(0, 8):
        uq = tl.load(
            U_q + (base_node * 8 + r) * D + d,
            mask=dmask, other=0,
        ).to(tl.float32)
        us = tl.load(
            U_scale + base_node * 8 + r,
            mask=active, other=0.0,
        ).to(tl.float32)
        uv = uq * us
        pr = tl.sum(dqv * uv, axis=0)
        lv = tl.load(lam + base_node * 8 + r, mask=active, other=0.0).to(tl.float32)
        p2 = pr * pr
        q2 += lv * p2
        proj_norm2 += p2

    lz = tl.load(logz + base_node, mask=active, other=-float("inf")).to(tl.float32)
    route = lz + linear + 0.5 * q2
    dq_norm2 = tl.sum(dqv * dqv, axis=0)
    residual = tl.maximum(dq_norm2 - proj_norm2, 0.0)
    tr = tl.load(trace + base_node, mask=active, other=0.0).to(tl.float32)
    ctr = tl.load(captured_trace + base_node, mask=active, other=0.0).to(tl.float32)
    mass = route + 1.10 * (tr - ctr) * residual / float(D - 8)

    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    lt = tl.load(log_total + qi, mask=active, other=0.0).to(tl.float32)
    score = route - lt + log_unc
    cc = tl.load(child_count + nid, mask=active, other=0).to(tl.int32)
    # Leaves are terminal accepted nodes even when their score exceeds tau.
    opened = active & (cc > 0) & (score > tau)
    next_count = tl.where(opened, cc, 0)

    tl.store(accepted_mass_out + fi, mass, mask=active)
    tl.store(open_out + fi, opened.to(tl.int8), mask=active)
    tl.store(next_count_out + fi, next_count, mask=active)


def htrc_frontier_step_int8_grouped(
    dq, frontier_q, frontier_node, frontier_count, log_total, *,
    logz, grad_q, grad_scale, lam, U_q, U_scale,
    trace, captured_trace, child_count,
    rows_per_centroid: int, capacity: int, tau: float,
):
    qpad, d = (int(v) for v in dq.shape)
    ccount, n = (int(logz.shape[0]), int(logz.shape[1]))
    capacity = int(capacity)
    if grad_q.shape != (ccount, n, d):
        raise ValueError("INT8 grad_q shape mismatch")
    if grad_scale.shape != (ccount, n):
        raise ValueError("INT8 grad_scale shape mismatch")
    if U_q.shape != (ccount, n, 8, d):
        raise ValueError("INT8 U_q shape mismatch")
    if U_scale.shape != (ccount, n, 8):
        raise ValueError("INT8 U_scale shape mismatch")

    _ensure_windows_triton_dll_search()
    out_mass = torch.empty((capacity,), device=dq.device, dtype=torch.float32)
    out_open = torch.empty((capacity,), device=dq.device, dtype=torch.int8)
    out_next = torch.zeros((capacity,), device=dq.device, dtype=torch.int32)
    block_d = triton.next_power_of_2(d)
    _htrc_frontier_step_int8_grouped_kernel[(capacity,)](
        dq.float().contiguous(),
        frontier_q.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_node.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        log_total.float().contiguous(),
        logz.float().contiguous(),
        grad_q.contiguous(), grad_scale.float().contiguous(),
        lam.float().contiguous(),
        U_q.contiguous(), U_scale.float().contiguous(),
        trace.float().contiguous(), captured_trace.float().contiguous(),
        child_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        out_mass, out_open, out_next, float(tau),
        N=n, D=d, ROWS_PER_C=int(rows_per_centroid),
        BLOCK_D=block_d, num_warps=4,
    )
    return out_mass, out_open.bool(), out_next


@triton.jit
def _htrc_frontier_step_mixed_grouped_kernel(
    dq,
    frontier_q,
    frontier_node,
    frontier_count,
    log_total,
    logz,
    grad_fp,
    grad_q,
    grad_scale,
    lam,
    U_fp,
    U_q,
    U_scale,
    trace,
    captured_trace,
    child_count,
    accepted_mass_out,
    open_out,
    next_count_out,
    tau,
    USE_INT4_GRAD: tl.constexpr,
    USE_INT4_U: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    ROWS_PER_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Ablation kernel: independently choose INT4 or native storage for grad/U."""
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    cid = qi // ROWS_PER_C
    base_node = cid * N + nid

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & active
    dqv = tl.load(dq + qi * D + d, mask=dmask, other=0.0).to(tl.float32)
    byte_i = d // 2
    shift = (d & 1) * 4

    if USE_INT4_GRAD:
        gb = tl.load(
            grad_q + base_node * PACKED_D + byte_i,
            mask=dmask, other=0,
        ).to(tl.int32)
        gn = (gb >> shift) & 15
        gn = tl.where(gn >= 8, gn - 16, gn).to(tl.float32)
        gs = tl.load(grad_scale + base_node, mask=active, other=0.0).to(tl.float32)
        gv = gn * gs
    else:
        gv = tl.load(
            grad_fp + base_node * D + d, mask=dmask, other=0.0
        ).to(tl.float32)

    linear = tl.sum(dqv * gv, axis=0)

    q2 = 0.0
    proj_norm2 = 0.0
    for r in tl.static_range(0, 8):
        if USE_INT4_U:
            ub = tl.load(
                U_q + (base_node * 8 + r) * PACKED_D + byte_i,
                mask=dmask, other=0,
            ).to(tl.int32)
            un = (ub >> shift) & 15
            un = tl.where(un >= 8, un - 16, un).to(tl.float32)
            us = tl.load(
                U_scale + base_node * 8 + r, mask=active, other=0.0
            ).to(tl.float32)
            uv = un * us
        else:
            uv = tl.load(
                U_fp + base_node * D * 8 + d * 8 + r,
                mask=dmask, other=0.0
            ).to(tl.float32)

        pr = tl.sum(dqv * uv, axis=0)
        lv = tl.load(lam + base_node * 8 + r, mask=active, other=0.0).to(tl.float32)
        p2 = pr * pr
        q2 += lv * p2
        proj_norm2 += p2

    lz = tl.load(logz + base_node, mask=active, other=-float("inf")).to(tl.float32)
    route = lz + linear + 0.5 * q2
    dq_norm2 = tl.sum(dqv * dqv, axis=0)
    residual = tl.maximum(dq_norm2 - proj_norm2, 0.0)
    tr = tl.load(trace + base_node, mask=active, other=0.0).to(tl.float32)
    ctr = tl.load(captured_trace + base_node, mask=active, other=0.0).to(tl.float32)
    mass = route + 1.10 * (tr - ctr) * residual / float(D - 8)

    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    lt = tl.load(log_total + qi, mask=active, other=0.0).to(tl.float32)
    score = route - lt + log_unc
    cc = tl.load(child_count + nid, mask=active, other=0).to(tl.int32)
    # Leaves are terminal accepted nodes even when their score exceeds tau.
    opened = active & (cc > 0) & (score > tau)
    next_count = tl.where(opened, cc, 0)

    tl.store(accepted_mass_out + fi, mass, mask=active)
    tl.store(open_out + fi, opened.to(tl.int8), mask=active)
    tl.store(next_count_out + fi, next_count, mask=active)


def htrc_frontier_step_mixed_grouped(
    dq, frontier_q, frontier_node, frontier_count, log_total, *,
    logz, grad_fp, grad_q, grad_scale, lam,
    U_fp, U_q, U_scale, trace, captured_trace, child_count,
    rows_per_centroid: int, capacity: int, tau: float,
    use_int4_grad: bool, use_int4_u: bool,
):
    qpad, d = (int(v) for v in dq.shape)
    ccount, n = (int(logz.shape[0]), int(logz.shape[1]))
    packed_d = (d + 1) // 2
    out_mass = torch.empty((capacity,), device=dq.device, dtype=torch.float32)
    out_open = torch.empty((capacity,), device=dq.device, dtype=torch.int8)
    out_next = torch.zeros((capacity,), device=dq.device, dtype=torch.int32)
    block_d = triton.next_power_of_2(d)

    _htrc_frontier_step_mixed_grouped_kernel[(capacity,)](
        dq.float().contiguous(),
        frontier_q.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_node.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        log_total.float().contiguous(),
        logz.float().contiguous(),
        grad_fp.contiguous(), grad_q.contiguous(), grad_scale.float().contiguous(),
        lam.float().contiguous(),
        U_fp.contiguous(), U_q.contiguous(), U_scale.float().contiguous(),
        trace.float().contiguous(), captured_trace.float().contiguous(),
        child_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        out_mass, out_open, out_next, float(tau),
        USE_INT4_GRAD=bool(use_int4_grad),
        USE_INT4_U=bool(use_int4_u),
        N=n, D=d, PACKED_D=packed_d,
        ROWS_PER_C=int(rows_per_centroid),
        BLOCK_D=block_d, num_warps=4,
    )
    return out_mass, out_open.bool(), out_next


@triton.jit
def _htrc_frontier_step_int4_grouped_kernel(
    dq,
    frontier_q,
    frontier_node,
    frontier_count,
    log_total,
    logz,
    grad_q,
    grad_scale,
    lam,
    U_q,
    U_scale,
    trace,
    captured_trace,
    child_count,
    accepted_mass_out,
    open_out,
    next_count_out,
    tau,
    N: tl.constexpr,
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    ROWS_PER_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Storage-only signed INT4 frontier step with on-the-fly decode."""
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    cid = qi // ROWS_PER_C
    base_node = cid * N + nid

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & active
    dqv = tl.load(dq + qi * D + d, mask=dmask, other=0.0).to(tl.float32)

    # grad: packed two signed 4-bit values per byte along D.
    gd_byte = d // 2
    gd_shift = (d & 1) * 4
    gb = tl.load(
        grad_q + base_node * PACKED_D + gd_byte,
        mask=dmask,
        other=0,
    ).to(tl.int32)
    gn = (gb >> gd_shift) & 15
    gn = tl.where(gn >= 8, gn - 16, gn).to(tl.float32)
    gs = tl.load(grad_scale + base_node, mask=active, other=0.0).to(tl.float32)
    gv = gn * gs
    linear = tl.sum(dqv * gv, axis=0)

    q2 = 0.0
    proj_norm2 = 0.0
    for r in tl.static_range(0, 8):
        # U packed as [node, r, packed_D], scale [node, r].
        ub = tl.load(
            U_q + (base_node * 8 + r) * PACKED_D + gd_byte,
            mask=dmask,
            other=0,
        ).to(tl.int32)
        un = (ub >> gd_shift) & 15
        un = tl.where(un >= 8, un - 16, un).to(tl.float32)
        us = tl.load(
            U_scale + base_node * 8 + r,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        uv = un * us
        pr = tl.sum(dqv * uv, axis=0)
        lv = tl.load(
            lam + base_node * 8 + r, mask=active, other=0.0
        ).to(tl.float32)
        p2 = pr * pr
        q2 += lv * p2
        proj_norm2 += p2

    lz = tl.load(logz + base_node, mask=active, other=-float("inf")).to(tl.float32)
    route = lz + linear + 0.5 * q2
    dq_norm2 = tl.sum(dqv * dqv, axis=0)
    residual = tl.maximum(dq_norm2 - proj_norm2, 0.0)
    tr = tl.load(trace + base_node, mask=active, other=0.0).to(tl.float32)
    ctr = tl.load(
        captured_trace + base_node, mask=active, other=0.0
    ).to(tl.float32)
    tail = (tr - ctr) * residual / float(D - 8)
    mass = route + 1.10 * tail

    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    lt = tl.load(log_total + qi, mask=active, other=0.0).to(tl.float32)
    score = route - lt + log_unc
    cc = tl.load(child_count + nid, mask=active, other=0).to(tl.int32)
    # Leaves are terminal accepted nodes even when their score exceeds tau.
    opened = active & (cc > 0) & (score > tau)
    next_count = tl.where(opened, cc, 0)
    tl.store(accepted_mass_out + fi, mass, mask=active)
    tl.store(open_out + fi, opened.to(tl.int8), mask=active)
    tl.store(next_count_out + fi, next_count, mask=active)


def htrc_frontier_step_int4_grouped(
    dq: torch.Tensor,
    frontier_q: torch.Tensor,
    frontier_node: torch.Tensor,
    frontier_count: torch.Tensor,
    log_total: torch.Tensor,
    *,
    logz: torch.Tensor,
    grad_q: torch.Tensor,
    grad_scale: torch.Tensor,
    lam: torch.Tensor,
    U_q: torch.Tensor,
    U_scale: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    child_count: torch.Tensor,
    rows_per_centroid: int,
    capacity: int,
    tau: float,
):
    qpad, d = (int(v) for v in dq.shape)
    ccount, n = (int(logz.shape[0]), int(logz.shape[1]))
    packed_d = (d + 1) // 2
    capacity = int(capacity)
    if grad_q.shape != (ccount, n, packed_d):
        raise ValueError("INT4 grad_q shape mismatch")
    if grad_scale.shape != (ccount, n):
        raise ValueError("INT4 grad_scale shape mismatch")
    if U_q.shape != (ccount, n, 8, packed_d):
        raise ValueError("INT4 U_q shape mismatch")
    if U_scale.shape != (ccount, n, 8):
        raise ValueError("INT4 U_scale shape mismatch")

    _ensure_windows_triton_dll_search()
    out_mass = torch.empty((capacity,), device=dq.device, dtype=torch.float32)
    out_open = torch.empty((capacity,), device=dq.device, dtype=torch.int8)
    out_next_count = torch.zeros((capacity,), device=dq.device, dtype=torch.int32)
    block_d = triton.next_power_of_2(d)
    _htrc_frontier_step_int4_grouped_kernel[(capacity,)](
        dq.float().contiguous(),
        frontier_q.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_node.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        log_total.float().contiguous(),
        logz.float().contiguous(),
        grad_q.contiguous(),
        grad_scale.float().contiguous(),
        lam.float().contiguous(),
        U_q.contiguous(),
        U_scale.float().contiguous(),
        trace.float().contiguous(),
        captured_trace.float().contiguous(),
        child_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        out_mass, out_open, out_next_count, float(tau),
        N=n, D=d, PACKED_D=packed_d,
        ROWS_PER_C=int(rows_per_centroid),
        BLOCK_D=block_d, num_warps=4,
    )
    return out_mass, out_open.bool(), out_next_count


@triton.jit
def _htrc_frontier_step_grouped_kernel(
    dq,
    frontier_q,
    frontier_node,
    frontier_count,
    log_total,
    logz,
    grad,
    lam,
    U,
    trace,
    captured_trace,
    child_count,
    accepted_mass_out,
    open_out,
    next_count_out,
    tau,
    N: tl.constexpr,
    D: tl.constexpr,
    ROWS_PER_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fuse child node evaluation + routing decision + next-frontier count."""
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    cid = qi // ROWS_PER_C

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & active
    dqv = tl.load(dq + qi * D + d, mask=dmask, other=0.0).to(tl.float32)

    base_node = cid * N + nid
    gv = tl.load(
        grad + base_node * D + d, mask=dmask, other=0.0
    ).to(tl.float32)
    linear = tl.sum(dqv * gv, axis=0)

    q2 = 0.0
    proj_norm2 = 0.0
    for r in tl.static_range(0, 8):
        uv = tl.load(
            U + base_node * D * 8 + d * 8 + r,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        pr = tl.sum(dqv * uv, axis=0)
        lv = tl.load(
            lam + base_node * 8 + r, mask=active, other=0.0
        ).to(tl.float32)
        p2 = pr * pr
        q2 += lv * p2
        proj_norm2 += p2

    lz = tl.load(logz + base_node, mask=active, other=-float("inf")).to(tl.float32)
    route = lz + linear + 0.5 * q2
    dq_norm2 = tl.sum(dqv * dqv, axis=0)
    residual = tl.maximum(dq_norm2 - proj_norm2, 0.0)
    tr = tl.load(trace + base_node, mask=active, other=0.0).to(tl.float32)
    ctr = tl.load(
        captured_trace + base_node, mask=active, other=0.0
    ).to(tl.float32)
    tail = (tr - ctr) * residual / float(D - 8)
    mass = route + 1.10 * tail

    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    lt = tl.load(log_total + qi, mask=active, other=0.0).to(tl.float32)
    score = route - lt + log_unc
    cc = tl.load(child_count + nid, mask=active, other=0).to(tl.int32)
    # Leaves are terminal accepted nodes even when their score exceeds tau.
    opened = active & (cc > 0) & (score > tau)
    next_count = tl.where(opened, cc, 0)

    tl.store(accepted_mass_out + fi, mass, mask=active)
    tl.store(open_out + fi, opened.to(tl.int8), mask=active)
    tl.store(next_count_out + fi, next_count, mask=active)




@triton.jit
def _htrc_frontier_row_keys_kernel(
    frontier_q,
    frontier_node,
    query_centroid,
    out,
    N,
    NODE_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    qi = tl.load(frontier_q + off, mask=mask, other=0).to(tl.int32)
    ni = tl.load(frontier_node + off, mask=mask, other=0).to(tl.int32)
    ci = tl.load(query_centroid + qi, mask=mask, other=0).to(tl.int32)
    tl.store(out + off, ci * NODE_COUNT + ni, mask=mask)


def htrc_frontier_row_keys(
    frontier_q: torch.Tensor,
    frontier_node: torch.Tensor,
    query_centroid: torch.Tensor,
    *,
    node_count: int,
) -> torch.Tensor:
    """Map packed frontier query ids to exact (centroid,node) keys."""
    if frontier_q.shape != frontier_node.shape:
        raise ValueError("HTRC frontier row-key shape mismatch")
    dev = frontier_q.device
    fq = frontier_q.to(device=dev, dtype=torch.int32).contiguous()
    fn = frontier_node.to(device=dev, dtype=torch.int32).contiguous()
    qc = query_centroid.to(device=dev, dtype=torch.int32).contiguous()
    n = int(fq.numel())
    out = torch.empty((n,), device=dev, dtype=torch.int32)
    if n == 0:
        return out
    if dev.type != "cuda":
        ql = fq.to(torch.long)
        return qc.index_select(0, ql) * int(node_count) + fn
    _ensure_windows_triton_dll_search()
    block = 256
    _htrc_frontier_row_keys_kernel[(triton.cdiv(n, block),)](
        fq, fn, qc, out, n,
        NODE_COUNT=int(node_count), BLOCK=block, num_warps=4,
    )
    return out



@triton.jit
def _htrc_frontier_step_packed_kernel(
    dq,
    frontier_q,
    frontier_node,
    frontier_slot,
    frontier_count,
    log_total,
    logz,
    grad,
    lam,
    U,
    trace,
    captured_trace,
    child_count,
    accepted_mass_out,
    open_out,
    next_count_out,
    tau,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Evaluate one streaming frontier from packed current-level state."""
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    slot = tl.load(frontier_slot + fi, mask=active, other=0).to(tl.int32)

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & active
    dqv = tl.load(dq + qi * D + d, mask=dmask, other=0.0).to(tl.float32)
    gv = tl.load(
        grad + slot * D + d, mask=dmask, other=0.0
    ).to(tl.float32)
    linear = tl.sum(dqv * gv, axis=0)

    q2 = 0.0
    proj_norm2 = 0.0
    for r in tl.static_range(0, 8):
        uv = tl.load(
            U + (slot * D + d) * 8 + r,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        pr = tl.sum(dqv * uv, axis=0)
        lv = tl.load(
            lam + slot * 8 + r, mask=active, other=0.0
        ).to(tl.float32)
        p2 = pr * pr
        q2 += lv * p2
        proj_norm2 += p2

    lz = tl.load(logz + slot, mask=active, other=-float("inf")).to(tl.float32)
    route = lz + linear + 0.5 * q2
    dq_norm2 = tl.sum(dqv * dqv, axis=0)
    residual = tl.maximum(dq_norm2 - proj_norm2, 0.0)
    tr = tl.load(trace + slot, mask=active, other=0.0).to(tl.float32)
    ctr = tl.load(captured_trace + slot, mask=active, other=0.0).to(tl.float32)
    mass = route + 1.10 * (tr - ctr) * residual / float(D - 8)

    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    lt = tl.load(log_total + qi, mask=active, other=0.0).to(tl.float32)
    score = route - lt + log_unc
    cc = tl.load(child_count + nid, mask=active, other=0).to(tl.int32)
    opened = active & (cc > 0) & (score > tau)
    next_count = tl.where(opened, cc, 0)

    tl.store(accepted_mass_out + fi, mass, mask=active)
    tl.store(open_out + fi, opened.to(tl.int8), mask=active)
    tl.store(next_count_out + fi, next_count, mask=active)


def htrc_frontier_step_packed(
    dq: torch.Tensor,
    frontier_q: torch.Tensor,
    frontier_node: torch.Tensor,
    frontier_slot: torch.Tensor,
    frontier_count: torch.Tensor,
    log_total: torch.Tensor,
    *,
    logz: torch.Tensor,
    grad: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    child_count: torch.Tensor,
    capacity: int,
    tau: float,
):
    """Streaming frontier decision over packed current-level state rows."""
    qpad, d = (int(v) for v in dq.shape)
    capacity = int(capacity)
    m = int(logz.shape[0])
    if grad.shape != (m, d):
        raise ValueError("packed HTRC grad must be [M,D]")
    if lam.shape != (m, 8) or U.shape != (m, d, 8):
        raise ValueError("packed HTRC rank8 state shape mismatch")
    if trace.shape != (m,) or captured_trace.shape != (m,):
        raise ValueError("packed HTRC trace state must be [M]")
    for name, t in (
        ("frontier_q", frontier_q),
        ("frontier_node", frontier_node),
        ("frontier_slot", frontier_slot),
    ):
        if int(t.shape[0]) < capacity:
            raise ValueError(f"packed HTRC {name} smaller than capacity")
    if log_total.shape != (qpad,):
        raise ValueError("packed HTRC log_total must be [Qpad]")

    _ensure_windows_triton_dll_search()
    out_mass = torch.empty((capacity,), device=dq.device, dtype=torch.float32)
    out_open = torch.empty((capacity,), device=dq.device, dtype=torch.int8)
    out_next_count = torch.zeros(
        (capacity,), device=dq.device, dtype=torch.int32
    )
    block_d = triton.next_power_of_2(d)
    _htrc_frontier_step_packed_kernel[(capacity,)](
        dq.float().contiguous(),
        frontier_q.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_node.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_slot.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        log_total.float().contiguous(),
        logz.float().contiguous(),
        grad.contiguous(),
        lam.float().contiguous(),
        U.contiguous(),
        trace.float().contiguous(),
        captured_trace.float().contiguous(),
        child_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        out_mass,
        out_open,
        out_next_count,
        float(tau),
        D=d,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out_mass, out_open.bool(), out_next_count


@triton.jit
def _htrc_frontier_level_value_packed_kernel(
    frontier_q,
    frontier_slot,
    frontier_count,
    accepted_mass,
    open_mask,
    value,
    level_max,
    level_n,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    opened = tl.load(open_mask + fi, mask=active, other=1) != 0
    take = active & (~opened)
    qi = tl.load(frontier_q + fi, mask=take, other=0).to(tl.int32)
    slot = tl.load(frontier_slot + fi, mask=take, other=0).to(tl.int32)
    mass = tl.load(accepted_mass + fi, mask=take, other=-float("inf")).to(tl.float32)
    lm = tl.load(level_max + qi, mask=take, other=-float("inf")).to(tl.float32)
    w = tl.exp(mass - lm)

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & take
    vv = tl.load(
        value + slot * D + d, mask=dmask, other=0.0
    ).to(tl.float32)
    tl.atomic_add(level_n + qi * D + d, w * vv, mask=dmask)


@triton.jit
def _htrc_frontier_merge_prepare_kernel(
    global_m,
    global_z,
    global_n,
    level_max,
    level_z,
    chunk_scale_out,
    Q,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    qi = tl.program_id(0)
    if qi >= Q:
        return
    lz = tl.load(level_z + qi).to(tl.float32)
    has = lz > 0.0
    gm = tl.load(global_m + qi).to(tl.float32)
    gz = tl.load(global_z + qi).to(tl.float32)
    lm = tl.load(level_max + qi).to(tl.float32)

    nm = tl.maximum(gm, lm)
    old_scale = tl.where(has, tl.exp(gm - nm), 1.0)
    chunk_scale = tl.where(has, tl.exp(lm - nm), 0.0)

    d = tl.arange(0, BLOCK_D)
    dmask = d < D
    gn = tl.load(global_n + qi * D + d, mask=dmask, other=0.0).to(tl.float32)
    tl.store(global_n + qi * D + d, gn * old_scale, mask=dmask)
    tl.store(global_m + qi, tl.where(has, nm, gm))
    tl.store(global_z + qi, tl.where(has, gz * old_scale + lz * chunk_scale, gz))
    tl.store(chunk_scale_out + qi, chunk_scale)


@triton.jit
def _htrc_frontier_level_value_merge_packed_kernel(
    frontier_q,
    frontier_slot,
    frontier_count,
    accepted_mass,
    open_mask,
    value,
    level_max,
    chunk_scale,
    global_n,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    opened = tl.load(open_mask + fi, mask=active, other=1) != 0
    take = active & (~opened)
    qi = tl.load(frontier_q + fi, mask=take, other=0).to(tl.int32)
    slot = tl.load(frontier_slot + fi, mask=take, other=0).to(tl.int32)
    mass = tl.load(accepted_mass + fi, mask=take, other=-float("inf")).to(tl.float32)
    lm = tl.load(level_max + qi, mask=take, other=0.0).to(tl.float32)
    cs = tl.load(chunk_scale + qi, mask=take, other=0.0).to(tl.float32)
    w = cs * tl.exp(mass - lm)

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & take
    vv = tl.load(value + slot * D + d, mask=dmask, other=0.0).to(tl.float32)
    tl.atomic_add(global_n + qi * D + d, w * vv, mask=dmask)


def htrc_frontier_level_merge_packed_(
    frontier_q: torch.Tensor,
    frontier_slot: torch.Tensor,
    frontier_count: torch.Tensor,
    accepted_mass: torch.Tensor,
    open_mask: torch.Tensor,
    *,
    value: torch.Tensor,
    global_m: torch.Tensor,
    global_z: torch.Tensor,
    global_n: torch.Tensor,
):
    """Stable frontier value merge directly into global accumulators.

    Avoids materializing the per-level [Q,D] numerator.
    """
    capacity = int(frontier_q.shape[0])
    d = int(value.shape[-1])
    qpad = int(global_m.numel())
    dev = value.device
    if global_z.shape != global_m.shape:
        raise ValueError("HTRC frontier direct merge m/z shape mismatch")
    if global_n.shape != (qpad, d):
        raise ValueError("HTRC frontier direct merge n shape mismatch")
    if any(t.dtype != torch.float32 for t in (global_m, global_z, global_n)):
        raise TypeError("HTRC frontier direct merge requires fp32 accumulators")

    fq = frontier_q.to(device=dev, dtype=torch.int32).contiguous()
    fs = frontier_slot.to(device=dev, dtype=torch.int32).contiguous()
    fc = frontier_count.to(device=dev, dtype=torch.int32).contiguous()
    mass = accepted_mass.float().contiguous()
    opened = open_mask.to(device=dev, dtype=torch.int8).contiguous()
    vc = value.contiguous()

    level_max = torch.full((qpad,), -float("inf"), dtype=torch.float32, device=dev)
    level_z = torch.zeros((qpad,), dtype=torch.float32, device=dev)
    chunk_scale = torch.zeros((qpad,), dtype=torch.float32, device=dev)

    _htrc_frontier_level_max_kernel[(capacity,)](
        fq, fc, mass, opened, level_max,
        CAPACITY=capacity, num_warps=1,
    )
    _htrc_frontier_level_z_kernel[(capacity,)](
        fq, fc, mass, opened, level_max, level_z,
        CAPACITY=capacity, num_warps=1,
    )

    block_d = triton.next_power_of_2(d)
    _htrc_frontier_merge_prepare_kernel[(qpad,)](
        global_m, global_z, global_n,
        level_max, level_z, chunk_scale, qpad,
        D=d, BLOCK_D=block_d, num_warps=4,
    )
    _htrc_frontier_level_value_merge_packed_kernel[(capacity,)](
        fq, fs, fc, mass, opened, vc,
        level_max, chunk_scale, global_n,
        D=d, BLOCK_D=block_d, num_warps=4,
    )
    return level_max, level_z, chunk_scale


def htrc_frontier_level_value_packed(
    frontier_q: torch.Tensor,
    frontier_slot: torch.Tensor,
    frontier_count: torch.Tensor,
    accepted_mass: torch.Tensor,
    open_mask: torch.Tensor,
    *,
    value: torch.Tensor,
    qpad: int,
):
    """Accumulate accepted packed frontier values for one streaming level."""
    capacity = int(frontier_q.shape[0])
    d = int(value.shape[-1])
    dev = value.device
    fq = frontier_q.to(device=dev, dtype=torch.int32).contiguous()
    fs = frontier_slot.to(device=dev, dtype=torch.int32).contiguous()
    fc = frontier_count.to(device=dev, dtype=torch.int32).contiguous()
    mass = accepted_mass.float().contiguous()
    opened = open_mask.to(device=dev, dtype=torch.int8).contiguous()

    # Reuse the existing compact frontier max/z kernels; they depend only on
    # query row, mass and open mask, not on node-state layout.
    level_max = torch.full(
        (int(qpad),), -float("inf"), dtype=torch.float32, device=dev
    )
    level_z = torch.zeros((int(qpad),), dtype=torch.float32, device=dev)
    level_n = torch.zeros((int(qpad), d), dtype=torch.float32, device=dev)
    _htrc_frontier_level_max_kernel[(capacity,)](
        fq, fc, mass, opened, level_max,
        CAPACITY=capacity,
        num_warps=1,
    )
    _htrc_frontier_level_z_kernel[(capacity,)](
        fq, fc, mass, opened, level_max, level_z,
        CAPACITY=capacity,
        num_warps=1,
    )
    block_d = triton.next_power_of_2(d)
    _htrc_frontier_level_value_packed_kernel[(capacity,)](
        fq, fs, fc, mass, opened, value.contiguous(),
        level_max, level_n,
        D=d,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return level_max, level_z, level_n


def htrc_frontier_step_grouped(
    dq: torch.Tensor,
    frontier_q: torch.Tensor,
    frontier_node: torch.Tensor,
    frontier_count: torch.Tensor,
    log_total: torch.Tensor,
    *,
    logz: torch.Tensor,
    grad: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    child_count: torch.Tensor,
    rows_per_centroid: int,
    capacity: int,
    tau: float,
):
    """Fused frontier eval+decision. Avoids materializing route/score tensors."""
    if dq.ndim != 2:
        raise ValueError("HTRC frontier dq must be [Qpad,D]")
    qpad, d = (int(v) for v in dq.shape)
    ccount, n = (int(logz.shape[0]), int(logz.shape[1]))
    capacity = int(capacity)
    if frontier_q.shape[0] < capacity or frontier_node.shape[0] < capacity:
        raise ValueError("HTRC frontier buffers smaller than capacity")
    if log_total.shape != (qpad,):
        raise ValueError("HTRC frontier log_total must be [Qpad]")
    if child_count.shape != (n,):
        raise ValueError("HTRC child_count shape mismatch")

    _ensure_windows_triton_dll_search()
    out_mass = torch.empty((capacity,), device=dq.device, dtype=torch.float32)
    out_open = torch.empty((capacity,), device=dq.device, dtype=torch.int8)
    out_next_count = torch.zeros((capacity,), device=dq.device, dtype=torch.int32)
    block_d = triton.next_power_of_2(d)
    _htrc_frontier_step_grouped_kernel[(capacity,)](
        dq.float().contiguous(),
        frontier_q.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_node.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        log_total.float().contiguous(),
        logz.float().contiguous(),
        grad.contiguous(),
        lam.float().contiguous(),
        U.contiguous(),
        trace.float().contiguous(),
        captured_trace.float().contiguous(),
        child_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        out_mass, out_open, out_next_count, float(tau),
        N=n, D=d, ROWS_PER_C=int(rows_per_centroid),
        BLOCK_D=block_d, num_warps=4,
    )
    return out_mass, out_open.bool(), out_next_count


@triton.jit
def _htrc_frontier_eval_grouped_kernel(
    dq,
    frontier_q,
    frontier_node,
    frontier_count,
    log_total,
    logz,
    grad,
    lam,
    U,
    trace,
    captured_trace,
    out_route,
    out_mass,
    out_score,
    N: tl.constexpr,
    D: tl.constexpr,
    ROWS_PER_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qi = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    cid = qi // ROWS_PER_C

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & active
    dqv = tl.load(dq + qi * D + d, mask=dmask, other=0.0).to(tl.float32)

    base_node = cid * N + nid
    gv = tl.load(
        grad + base_node * D + d, mask=dmask, other=0.0
    ).to(tl.float32)
    linear = tl.sum(dqv * gv, axis=0)

    q2 = 0.0
    proj_norm2 = 0.0
    for r in tl.static_range(0, 8):
        uv = tl.load(
            U + base_node * D * 8 + d * 8 + r,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        pr = tl.sum(dqv * uv, axis=0)
        lv = tl.load(
            lam + base_node * 8 + r, mask=active, other=0.0
        ).to(tl.float32)
        p2 = pr * pr
        q2 += lv * p2
        proj_norm2 += p2

    lz = tl.load(logz + base_node, mask=active, other=-float("inf")).to(tl.float32)
    route = lz + linear + 0.5 * q2
    dq_norm2 = tl.sum(dqv * dqv, axis=0)
    residual = tl.maximum(dq_norm2 - proj_norm2, 0.0)
    tr = tl.load(trace + base_node, mask=active, other=0.0).to(tl.float32)
    ctr = tl.load(
        captured_trace + base_node, mask=active, other=0.0
    ).to(tl.float32)
    tail = (tr - ctr) * residual / float(D - 8)
    mass = route + 1.10 * tail

    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    lt = tl.load(log_total + qi, mask=active, other=0.0).to(tl.float32)
    score = route - lt + log_unc

    tl.store(out_route + fi, route, mask=active)
    tl.store(out_mass + fi, mass, mask=active)
    tl.store(out_score + fi, score, mask=active)


def htrc_frontier_eval_grouped(
    dq: torch.Tensor,
    frontier_q: torch.Tensor,
    frontier_node: torch.Tensor,
    frontier_count: torch.Tensor,
    log_total: torch.Tensor,
    *,
    logz: torch.Tensor,
    grad: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    rows_per_centroid: int,
    capacity: int,
):
    """Evaluate a fixed-capacity compact HTRC child frontier on GPU."""
    if dq.ndim != 2:
        raise ValueError("HTRC frontier dq must be [Qpad,D]")
    qpad, d = (int(v) for v in dq.shape)
    ccount, n = (int(logz.shape[0]), int(logz.shape[1]))
    capacity = int(capacity)
    if frontier_q.shape[0] < capacity or frontier_node.shape[0] < capacity:
        raise ValueError("HTRC frontier buffers smaller than capacity")
    if log_total.shape != (qpad,):
        raise ValueError("HTRC frontier log_total must be [Qpad]")
    if grad.shape != (ccount, n, d):
        raise ValueError("HTRC frontier grad shape mismatch")
    if lam.shape != (ccount, n, 8) or U.shape != (ccount, n, d, 8):
        raise ValueError("HTRC frontier rank8 state shape mismatch")

    _ensure_windows_triton_dll_search()
    out_route = torch.empty((capacity,), device=dq.device, dtype=torch.float32)
    out_mass = torch.empty_like(out_route)
    out_score = torch.empty_like(out_route)
    block_d = triton.next_power_of_2(d)
    _htrc_frontier_eval_grouped_kernel[(capacity,)](
        dq.float().contiguous(),
        frontier_q.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_node.to(device=dq.device, dtype=torch.int32).contiguous(),
        frontier_count.to(device=dq.device, dtype=torch.int32).contiguous(),
        log_total.float().contiguous(),
        logz.float().contiguous(),
        grad.contiguous(),
        lam.float().contiguous(),
        U.contiguous(),
        trace.float().contiguous(),
        captured_trace.float().contiguous(),
        out_route, out_mass, out_score,
        N=n, D=d, ROWS_PER_C=int(rows_per_centroid),
        BLOCK_D=block_d, num_warps=4,
    )
    return out_route, out_mass, out_score



@triton.jit
def _htrc_root_route_direct_kernel(
    dq,
    lam,
    logz,
    grad,
    U,
    trace,
    captured_trace,
    route_out,
    cert_out,
    mass_out,
    Q,
    R: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS_PER_C: tl.constexpr,
    D_MINUS_RANK: tl.constexpr,
):
    """Direct root route/cert/mass from dq and root state; no QxRx8 proj."""
    pid = tl.program_id(0)
    total = Q * R
    active = pid < total
    qi = pid // R
    r = pid - qi * R
    cid = qi // ROWS_PER_C
    state = cid * R + r

    d = tl.arange(0, BLOCK_D)
    dmask = (d < D) & active
    dqv = tl.load(dq + qi * D + d, mask=dmask, other=0.0).to(tl.float32)
    gv = tl.load(
        grad + state * D + d, mask=dmask, other=0.0
    ).to(tl.float32)
    linear = tl.sum(dqv * gv, axis=0)
    dq_norm2 = tl.sum(dqv * dqv, axis=0)

    q2 = 0.0
    proj_norm2 = 0.0
    for k in tl.static_range(0, 8):
        uv = tl.load(
            U + (state * D + d) * 8 + k,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        p = tl.sum(dqv * uv, axis=0)
        lv = tl.load(
            lam + state * 8 + k, mask=active, other=0.0
        ).to(tl.float32)
        p2 = p * p
        q2 += lv * p2
        proj_norm2 += p2

    lz = tl.load(logz + state, mask=active, other=-float("inf")).to(tl.float32)
    route = lz + linear + 0.5 * q2
    residual = tl.maximum(dq_norm2 - proj_norm2, 0.0)
    tr = tl.load(trace + state, mask=active, other=0.0).to(tl.float32)
    ctr = tl.load(
        captured_trace + state, mask=active, other=0.0
    ).to(tl.float32)
    mass = route + 1.10 * (tr - ctr) * residual / float(D_MINUS_RANK)
    cert = tl.abs(0.5 * q2)

    tl.store(route_out + pid, route, mask=active)
    tl.store(cert_out + pid, cert, mask=active)
    tl.store(mass_out + pid, mass, mask=active)


@triton.jit
def _htrc_root_score_from_route_kernel(
    route,
    cert_score,
    log_total,
    Q,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Reduce root route mass and overwrite cert buffer with routing score."""
    qi = tl.program_id(0)
    if qi >= Q:
        return
    r = tl.arange(0, BLOCK_R)
    mask = r < R
    off = qi * R + r
    rv = tl.load(route + off, mask=mask, other=-float("inf")).to(tl.float32)
    rm = tl.max(rv, axis=0)
    rz = tl.sum(tl.exp(rv - rm), axis=0)
    lt = rm + tl.log(tl.maximum(rz, 1.0e-20))
    cert = tl.load(cert_score + off, mask=mask, other=0.0).to(tl.float32)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    score = rv - lt + log_unc
    tl.store(cert_score + off, score, mask=mask)
    tl.store(log_total + qi, lt)


def htrc_root_route_direct(
    dq: torch.Tensor,
    *,
    lam: torch.Tensor,
    logz: torch.Tensor,
    grad: torch.Tensor,
    U: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    rows_per_centroid: int,
):
    """Direct root route/score/mass without materializing rank8 projections."""
    if dq.ndim != 2:
        raise ValueError("HTRC direct root dq must be [Qpad,D]")
    qpad, d = (int(x) for x in dq.shape)
    ccount, roots, gd = (int(x) for x in grad.shape)
    if gd != d:
        raise ValueError("HTRC direct root grad D mismatch")
    if U.shape != (ccount, roots, d, 8):
        raise ValueError("HTRC direct root U shape mismatch")
    if lam.shape != (ccount, roots, 8):
        raise ValueError("HTRC direct root lam shape mismatch")
    for name, t in (
        ("logz", logz),
        ("trace", trace),
        ("captured_trace", captured_trace),
    ):
        if t.shape != (ccount, roots):
            raise ValueError(f"HTRC direct root {name} shape mismatch")
    if int(rows_per_centroid) <= 0 or ccount * int(rows_per_centroid) != qpad:
        raise ValueError("HTRC direct root rows_per_centroid mismatch")

    dev = dq.device
    _ensure_windows_triton_dll_search()
    dqf = dq.float().contiguous()
    lamf = lam.float().contiguous()
    lzf = logz.float().contiguous()
    gf = grad.contiguous()
    uf = U.contiguous()
    trf = trace.float().contiguous()
    ctrf = captured_trace.float().contiguous()

    route = torch.empty((qpad, roots), device=dev, dtype=torch.float32)
    score = torch.empty_like(route)
    mass = torch.empty_like(route)
    log_total = torch.empty((qpad,), device=dev, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    total = qpad * roots
    _htrc_root_route_direct_kernel[(total,)](
        dqf, lamf, lzf, gf, uf, trf, ctrf,
        route, score, mass,
        qpad,
        R=roots,
        D=d,
        BLOCK_D=block_d,
        ROWS_PER_C=int(rows_per_centroid),
        D_MINUS_RANK=max(d - 8, 1),
        num_warps=4,
    )
    block_r = triton.next_power_of_2(roots)
    _htrc_root_score_from_route_kernel[(qpad,)](
        route, score, log_total, qpad,
        R=roots, BLOCK_R=block_r, num_warps=4,
    )
    return route, score, mass, log_total



@triton.jit
def _htrc_root_chunk_post_kernel(
    linear,
    proj,
    dq_norm2,
    lam,
    logz,
    trace,
    captured_trace,
    route_out,
    cert_out,
    mass_out,
    Q,
    C: tl.constexpr,
    RB: tl.constexpr,
    ROWS_PER_C: tl.constexpr,
    D_MINUS_RANK: tl.constexpr,
):
    """Fuse root-chunk q2/cert/tail postprocessing after GEMM."""
    pid = tl.program_id(0)
    total = Q * RB
    active = pid < total
    qi = pid // RB
    r = pid - qi * RB
    cid = qi // ROWS_PER_C
    state = cid * RB + r

    q2 = 0.0
    proj_norm2 = 0.0
    base = (qi * RB + r) * 8
    for k in tl.static_range(0, 8):
        pv = tl.load(proj + base + k, mask=active, other=0.0).to(tl.float32)
        lv = tl.load(lam + state * 8 + k, mask=active, other=0.0).to(tl.float32)
        p2 = pv * pv
        q2 += p2 * lv
        proj_norm2 += p2

    lin = tl.load(linear + qi * RB + r, mask=active, other=0.0).to(tl.float32)
    lz = tl.load(logz + state, mask=active, other=0.0).to(tl.float32)
    route = lz + lin + 0.5 * q2
    dn2 = tl.load(dq_norm2 + qi, mask=active, other=0.0).to(tl.float32)
    residual = tl.maximum(dn2 - proj_norm2, 0.0)
    tr = tl.load(trace + state, mask=active, other=0.0).to(tl.float32)
    ctr = tl.load(captured_trace + state, mask=active, other=0.0).to(tl.float32)
    mass = route + 1.10 * (tr - ctr) * residual / float(D_MINUS_RANK)

    tl.store(route_out + qi * RB + r, route, mask=active)
    tl.store(cert_out + qi * RB + r, tl.abs(0.5 * q2), mask=active)
    tl.store(mass_out + qi * RB + r, mass, mask=active)


def htrc_root_chunk_post(
    linear: torch.Tensor,
    proj: torch.Tensor,
    dq_norm2: torch.Tensor,
    *,
    lam: torch.Tensor,
    logz: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    dim: int,
    rows_per_centroid: int,
):
    """Fused root-chunk post-GEMM route/certificate/mass."""
    if linear.ndim != 2:
        raise ValueError("HTRC root chunk linear must be [Q,RB]")
    qn, rb = (int(x) for x in linear.shape)
    if proj.shape != (qn, rb, 8):
        raise ValueError("HTRC root chunk proj must be [Q,RB,8]")
    if dq_norm2.shape != (qn,):
        raise ValueError("HTRC root chunk dq_norm2 must be [Q]")
    if lam.ndim != 3 or lam.shape[1:] != (rb, 8):
        raise ValueError("HTRC root chunk lam must be [C,RB,8]")
    ccount = int(lam.shape[0])
    for name, t in (
        ("logz", logz),
        ("trace", trace),
        ("captured_trace", captured_trace),
    ):
        if t.shape != (ccount, rb):
            raise ValueError(f"HTRC root chunk {name} must be [C,RB]")
    if int(rows_per_centroid) <= 0 or ccount * int(rows_per_centroid) != qn:
        raise ValueError("HTRC root chunk rows_per_centroid mismatch")

    dev = linear.device
    route = torch.empty((qn, rb), device=dev, dtype=torch.float32)
    cert = torch.empty_like(route)
    mass = torch.empty_like(route)
    _ensure_windows_triton_dll_search()
    _htrc_root_chunk_post_kernel[(qn * rb,)](
        linear.float().contiguous(),
        proj.float().contiguous(),
        dq_norm2.float().contiguous(),
        lam.float().contiguous(),
        logz.float().contiguous(),
        trace.float().contiguous(),
        captured_trace.float().contiguous(),
        route,
        cert,
        mass,
        qn,
        C=ccount,
        RB=rb,
        ROWS_PER_C=int(rows_per_centroid),
        D_MINUS_RANK=max(int(dim) - 8, 1),
        num_warps=4,
    )
    return route, cert, mass




@triton.jit
def _htrc_row_norm2_kernel(
    x,
    out,
    Q,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    qi = tl.program_id(0)
    if qi >= Q:
        return
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    xv = tl.load(x + qi * D + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + qi, tl.sum(xv * xv, axis=0))


def htrc_row_norm2(x: torch.Tensor) -> torch.Tensor:
    """Compute fp32 per-row squared L2 norm without materializing x.square()."""
    if x.ndim != 2:
        raise ValueError("HTRC row norm2 expects [Q,D]")
    qn, d = (int(v) for v in x.shape)
    if d <= 0 or d > 256:
        raise ValueError(f"HTRC row norm2 supports D=1..256, got {d}")
    if x.device.type != "cuda":
        return x.float().square().sum(-1)
    _ensure_windows_triton_dll_search()
    xc = x.float().contiguous()
    out = torch.empty((qn,), device=x.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    _htrc_row_norm2_kernel[(qn,)](
        xc, out, qn,
        D=d, BLOCK_D=block_d, num_warps=4,
    )
    return out




@triton.jit
def _htrc_cert_logunc_inplace_kernel(
    x,
    N,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    q2 = tl.load(x + off, mask=mask, other=0.0).to(tl.float32)
    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    tl.store(x + off, log_unc, mask=mask)


def htrc_cert_logunc_inplace(x: torch.Tensor) -> torch.Tensor:
    """Convert q2 to log(expm1(abs(.5*q2))) in-place."""
    if x.dtype != torch.float32:
        raise TypeError("HTRC cert/log-unc transform requires fp32")
    if x.device.type != "cuda":
        cert = x.abs().mul_(0.5)
        x.copy_(
            torch.where(
                cert > 20.0,
                cert,
                torch.log(torch.expm1(cert).clamp_min(1.0e-30)),
            )
        )
        return x
    _ensure_windows_triton_dll_search()
    xc = x.contiguous()
    if xc.data_ptr() != x.data_ptr():
        raise ValueError("HTRC cert/log-unc input must be contiguous")
    n = int(x.numel())
    block = 256
    _htrc_cert_logunc_inplace_kernel[(triton.cdiv(n, block),)](
        xc, n, BLOCK=block, num_warps=4,
    )
    return x



@triton.jit
def _htrc_root_score_inplace_kernel(
    route,
    cert,
    log_total,
    Q,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Overwrite route[Q,R] with score using cert and per-query log_total."""
    qi = tl.program_id(0)
    if qi >= Q:
        return
    r = tl.arange(0, BLOCK_R)
    mask = r < R
    off = qi * R + r
    rv = tl.load(route + off, mask=mask, other=-float("inf")).to(tl.float32)
    cv = tl.load(cert + off, mask=mask, other=0.0).to(tl.float32)
    lt = tl.load(log_total + qi).to(tl.float32)
    small = tl.log(tl.maximum(tl.exp(cv) - 1.0, 1.0e-30))
    log_unc = tl.where(cv > 20.0, cv, small)
    tl.store(route + off, rv - lt + log_unc, mask=mask)


def htrc_root_score_inplace(
    route: torch.Tensor,
    cert: torch.Tensor,
    log_total: torch.Tensor,
) -> torch.Tensor:
    """Convert route[Q,R] to routing score in-place without full-size temporaries."""
    if route.ndim != 2 or cert.shape != route.shape:
        raise ValueError("HTRC root score expects matching route/cert [Q,R]")
    qn, roots = (int(x) for x in route.shape)
    if log_total.shape != (qn,):
        raise ValueError("HTRC root score log_total must be [Q]")
    if route.dtype != torch.float32 or cert.dtype != torch.float32:
        raise TypeError("HTRC root score route/cert must be fp32")
    if route.device.type != "cuda":
        route.copy_(
            route - log_total[:, None]
            + torch.where(
                cert > 20.0,
                cert,
                torch.log(torch.expm1(cert).clamp_min(1.0e-30)),
            )
        )
        return route
    _ensure_windows_triton_dll_search()
    rc = route.contiguous()
    cc = cert.contiguous()
    lc = log_total.float().contiguous()
    if rc.data_ptr() != route.data_ptr():
        raise ValueError("HTRC root score route must be contiguous")
    block_r = triton.next_power_of_2(roots)
    _htrc_root_score_inplace_kernel[(qn,)](
        rc, cc, lc, qn,
        R=roots, BLOCK_R=block_r, num_warps=4,
    )
    return route


@triton.jit
def _htrc_root_decision_global_kernel(
    linear,
    proj,
    dq_norm2,
    centroid_id,
    lam,
    logz,
    trace,
    captured_trace,
    accepted_mass_out,
    score_out,
    open_out,
    log_total_out,
    tau,
    Q,
    C: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    D_MINUS_RANK: tl.constexpr,
):
    """Evaluate all HTRC roots for one query row and emit routing decisions."""
    qi = tl.program_id(0)
    if qi >= Q:
        return

    cid = tl.load(centroid_id + qi).to(tl.int32)
    r = tl.arange(0, BLOCK_R)
    rmask = r < R
    cr = cid * R + r

    q2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
    proj_norm2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
    base = (qi * R + r) * 8
    for k in tl.static_range(0, 8):
        pv = tl.load(proj + base + k, mask=rmask, other=0.0).to(tl.float32)
        lv = tl.load(lam + cr * 8 + k, mask=rmask, other=0.0).to(tl.float32)
        p2 = pv * pv
        q2 += p2 * lv
        proj_norm2 += p2

    dn2 = tl.load(dq_norm2 + qi).to(tl.float32)
    residual = tl.maximum(dn2 - proj_norm2, 0.0)
    tr = tl.load(trace + cr, mask=rmask, other=0.0).to(tl.float32)
    ctr = tl.load(
        captured_trace + cr, mask=rmask, other=0.0
    ).to(tl.float32)
    tail = (tr - ctr) * residual / float(D_MINUS_RANK)

    lin = tl.load(
        linear + qi * R + r, mask=rmask, other=-float("inf")
    ).to(tl.float32)
    lz = tl.load(
        logz + cr, mask=rmask, other=-float("inf")
    ).to(tl.float32)
    route_pred = lz + lin + 0.5 * q2
    route_pred = tl.where(rmask, route_pred, -float("inf"))
    accepted_mass = route_pred + 1.10 * tail

    rm = tl.max(route_pred, axis=0)
    rz = tl.sum(tl.exp(route_pred - rm), axis=0)
    log_total = rm + tl.log(tl.maximum(rz, 1.0e-20))

    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    score = route_pred - log_total + log_unc
    is_open = score > tau

    tl.store(
        accepted_mass_out + qi * R + r,
        accepted_mass,
        mask=rmask,
    )
    tl.store(score_out + qi * R + r, score, mask=rmask)
    tl.store(open_out + qi * R + r, is_open.to(tl.int8), mask=rmask)
    tl.store(log_total_out + qi, log_total)


def htrc_root_decision_global(
    linear: torch.Tensor,
    proj: torch.Tensor,
    dq_norm2: torch.Tensor,
    centroid_id: torch.Tensor,
    *,
    lam: torch.Tensor,
    logz: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    dim: int,
    tau: float,
):
    """Return accepted root mass, routing score and open mask in one launch."""
    if linear.ndim != 2:
        raise ValueError("HTRC root decision linear must be [Q,R]")
    qn, roots = (int(v) for v in linear.shape)
    if proj.shape != (qn, roots, 8):
        raise ValueError("HTRC root decision proj must be [Q,R,8]")
    if dq_norm2.shape != (qn,) or centroid_id.shape != (qn,):
        raise ValueError("HTRC root decision row metadata mismatch")
    ccount = int(lam.shape[0])
    if lam.shape != (ccount, roots, 8):
        raise ValueError("HTRC root decision lam must be [C,R,8]")
    for name, t in (
        ("logz", logz),
        ("trace", trace),
        ("captured_trace", captured_trace),
    ):
        if t.shape != (ccount, roots):
            raise ValueError(f"HTRC root decision {name} must be [C,R]")

    _ensure_windows_triton_dll_search()
    lc = linear.float().contiguous()
    pc = proj.float().contiguous()
    dc = dq_norm2.float().contiguous()
    cc = centroid_id.to(device=linear.device, dtype=torch.int32).contiguous()
    lamc = lam.float().contiguous()
    lzc = logz.float().contiguous()
    tc = trace.float().contiguous()
    ctc = captured_trace.float().contiguous()
    accepted = torch.empty((qn, roots), dtype=torch.float32, device=linear.device)
    score = torch.empty_like(accepted)
    opened = torch.empty((qn, roots), dtype=torch.int8, device=linear.device)
    log_total = torch.empty((qn,), dtype=torch.float32, device=linear.device)
    block_r = triton.next_power_of_2(roots)
    _htrc_root_decision_global_kernel[(qn,)](
        lc, pc, dc, cc, lamc, lzc, tc, ctc,
        accepted, score, opened, log_total, float(tau), qn,
        C=ccount, R=roots, BLOCK_R=block_r,
        D_MINUS_RANK=max(int(dim) - 8, 1),
        num_warps=4,
    )
    return accepted, score, opened.bool(), log_total


@triton.jit
def _htrc_mass_post_global_kernel(
    linear,
    proj,
    dq_norm2,
    centroid_id,
    lam,
    logz,
    trace,
    captured_trace,
    out_w,
    Q: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    D_MINUS_RANK: tl.constexpr,
):
    """One launch for all sorted queries across all centroid groups."""
    qi = tl.program_id(0)
    if qi >= Q:
        return

    cid = tl.load(centroid_id + qi).to(tl.int32)
    r = tl.arange(0, BLOCK_R)
    rmask = r < R
    cr = cid * R + r

    q2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
    proj_norm2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
    base = (qi * R + r) * 8
    for k in tl.static_range(0, 8):
        pv = tl.load(proj + base + k, mask=rmask, other=0.0).to(tl.float32)
        lv = tl.load(
            lam + cr * 8 + k, mask=rmask, other=0.0
        ).to(tl.float32)
        p2 = pv * pv
        q2 += p2 * lv
        proj_norm2 += p2

    dn2 = tl.load(dq_norm2 + qi).to(tl.float32)
    residual = tl.maximum(dn2 - proj_norm2, 0.0)

    tr = tl.load(trace + cr, mask=rmask, other=0.0).to(tl.float32)
    ctr = tl.load(
        captured_trace + cr, mask=rmask, other=0.0
    ).to(tl.float32)
    tail = (tr - ctr) * residual / float(D_MINUS_RANK)

    lin = tl.load(
        linear + qi * R + r, mask=rmask, other=-float("inf")
    ).to(tl.float32)
    lz = tl.load(
        logz + cr, mask=rmask, other=-float("inf")
    ).to(tl.float32)
    mass = lz + lin + 0.5 * q2 + 1.10 * tail
    mass = tl.where(rmask, mass, -float("inf"))

    mm = tl.max(mass, axis=0)
    ew = tl.exp(mass - mm)
    zz = tl.sum(ew, axis=0)
    w = ew / tl.maximum(zz, 1.0e-20)
    tl.store(out_w + qi * R + r, w, mask=rmask)


def htrc_mass_post_global(
    linear: torch.Tensor,
    proj: torch.Tensor,
    dq_norm2: torch.Tensor,
    centroid_id: torch.Tensor,
    *,
    lam: torch.Tensor,
    logz: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Fused HTRC root mass/softmax for all centroid groups in one launch."""
    if linear.ndim != 2:
        raise ValueError("global HTRC mass post linear must be [Q,R]")
    qn, roots = (int(v) for v in linear.shape)
    if proj.shape != (qn, roots, 8):
        raise ValueError("global HTRC mass post proj must be [Q,R,8]")
    if dq_norm2.shape != (qn,) or centroid_id.shape != (qn,):
        raise ValueError("global HTRC mass post row metadata mismatch")
    ccount = int(lam.shape[0])
    if lam.shape != (ccount, roots, 8):
        raise ValueError("global HTRC mass post lam must be [C,R,8]")
    for name, t in (
        ("logz", logz),
        ("trace", trace),
        ("captured_trace", captured_trace),
    ):
        if t.shape != (ccount, roots):
            raise ValueError(f"global HTRC mass post {name} must be [C,R]")
    if linear.device.type != "cuda":
        raise RuntimeError("global HTRC mass post requires CUDA")

    _ensure_windows_triton_dll_search()
    lc = linear.float().contiguous()
    pc = proj.float().contiguous()
    dc = dq_norm2.float().contiguous()
    cc = centroid_id.to(
        device=linear.device, dtype=torch.int32
    ).contiguous()
    lamc = lam.float().contiguous()
    lzc = logz.float().contiguous()
    tc = trace.float().contiguous()
    ctc = captured_trace.float().contiguous()
    out = torch.empty(
        (qn, roots), dtype=torch.float32, device=linear.device
    )
    block_r = triton.next_power_of_2(roots)
    _htrc_mass_post_global_kernel[(qn,)](
        lc, pc, dc, cc, lamc, lzc, tc, ctc, out,
        Q=qn,
        C=ccount,
        R=roots,
        BLOCK_R=block_r,
        D_MINUS_RANK=max(int(dim) - 8, 1),
        num_warps=4,
    )
    return out


@triton.jit
def _htrc_mass_post_kernel(
    linear,
    proj,
    dq_norm2,
    lam,
    logz,
    trace,
    captured_trace,
    out_w,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    D_MINUS_RANK: tl.constexpr,
):
    """Fuse rank8 q2, residual-tail accepted mass and root softmax.

    One program owns one query row. R is small (80 in current authority), so
    all root masses stay in registers and no intermediate q2/tail/mass tensors
    are materialized.
    """
    qi = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    rmask = r < R

    q2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
    proj_norm2 = tl.zeros((BLOCK_R,), dtype=tl.float32)
    base = (qi * R + r) * 8
    for k in tl.static_range(0, 8):
        pv = tl.load(proj + base + k, mask=rmask, other=0.0).to(tl.float32)
        lv = tl.load(lam + r * 8 + k, mask=rmask, other=0.0).to(tl.float32)
        p2 = pv * pv
        q2 += p2 * lv
        proj_norm2 += p2

    dn2 = tl.load(dq_norm2 + qi).to(tl.float32)
    residual = tl.maximum(dn2 - proj_norm2, 0.0)
    tr = tl.load(trace + r, mask=rmask, other=0.0).to(tl.float32)
    ctr = tl.load(captured_trace + r, mask=rmask, other=0.0).to(tl.float32)
    tail = (tr - ctr) * residual / float(D_MINUS_RANK)

    lin = tl.load(linear + qi * R + r, mask=rmask, other=-float("inf")).to(tl.float32)
    lz = tl.load(logz + r, mask=rmask, other=-float("inf")).to(tl.float32)
    mass = lz + lin + 0.5 * q2 + 1.10 * tail
    mass = tl.where(rmask, mass, -float("inf"))

    mm = tl.max(mass, axis=0)
    ew = tl.exp(mass - mm)
    zz = tl.sum(ew, axis=0)
    w = ew / tl.maximum(zz, 1.0e-20)
    tl.store(out_w + qi * R + r, w, mask=rmask)


def htrc_mass_post(
    linear: torch.Tensor,
    proj: torch.Tensor,
    dq_norm2: torch.Tensor,
    *,
    lam: torch.Tensor,
    logz: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Fused accepted-root HTRC mass/softmax for one centroid group."""
    if linear.ndim != 2:
        raise ValueError("HTRC mass post linear must be [Q,R]")
    qn, roots = (int(v) for v in linear.shape)
    if proj.shape != (qn, roots, 8):
        raise ValueError("HTRC mass post proj must be [Q,R,8]")
    if dq_norm2.shape != (qn,):
        raise ValueError("HTRC mass post dq_norm2 must be [Q]")
    if lam.shape != (roots, 8):
        raise ValueError("HTRC mass post lam must be [R,8]")
    for name, t in (
        ("logz", logz), ("trace", trace), ("captured_trace", captured_trace)
    ):
        if t.shape != (roots,):
            raise ValueError(f"HTRC mass post {name} must be [R]")
    if linear.device.type != "cuda":
        raise RuntimeError("HTRC fused mass post requires CUDA")
    if roots <= 0 or roots > 128:
        raise ValueError(f"HTRC fused mass post supports 1..128 roots, got {roots}")

    _ensure_windows_triton_dll_search()
    lc = linear.float().contiguous()
    pc = proj.float().contiguous()
    dc = dq_norm2.float().contiguous()
    lamc = lam.float().contiguous()
    lzc = logz.float().contiguous()
    tc = trace.float().contiguous()
    cc = captured_trace.float().contiguous()
    out = torch.empty((qn, roots), dtype=torch.float32, device=linear.device)
    block_r = triton.next_power_of_2(roots)
    _htrc_mass_post_kernel[(qn,)](
        lc, pc, dc, lamc, lzc, tc, cc, out,
        R=roots,
        BLOCK_R=block_r,
        D_MINUS_RANK=max(int(dim) - 8, 1),
        num_warps=4,
    )
    return out



def prepare_htrc_root_gemm(
    assign: torch.Tensor,
    *,
    centers: torch.Tensor,
    logz: torch.Tensor,
    grad: torch.Tensor,
    value: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    trace: torch.Tensor,
    captured_trace: torch.Tensor,
    root_ids: torch.Tensor,
    projection_dtype: torch.dtype = torch.float32,
):
    """Prepare grouped-by-centroid root tensors for GEMM HTRC evaluation.

    This is intentionally outside the timed root hot path. Query assignment is
    fixed for one HTRC evaluation, so sorting/group metadata is one-time state.
    """
    if assign.ndim != 1:
        raise ValueError("HTRC root GEMM assign must be [Q]")
    ccount, n = (int(logz.shape[0]), int(logz.shape[1]))
    d = int(grad.shape[-1])
    roots = int(root_ids.numel())
    if roots <= 0:
        raise ValueError("HTRC root GEMM requires roots")

    order = torch.argsort(assign.to(torch.int64))
    counts = torch.bincount(
        assign.to(torch.int64), minlength=ccount
    ).cpu().tolist()
    rid = root_ids.to(device=logz.device, dtype=torch.long)

    groups = []
    start = 0
    for cid, count in enumerate(counts):
        stop = start + int(count)
        if count:
            gr = grad[cid].index_select(0, rid).to(projection_dtype).contiguous()
            ur = U[cid].index_select(0, rid)
            # [R,D,8] -> [D,R*8] for one large projection GEMM.
            uflat = (
                ur.permute(1, 0, 2)
                .reshape(d, roots * 8)
                .to(projection_dtype)
                .contiguous()
            )
            groups.append({
                "cid": int(cid),
                "start": int(start),
                "stop": int(stop),
                "center": centers[cid].to(projection_dtype).contiguous(),
                "grad_t": gr.transpose(0, 1).contiguous(),
                "u_flat": uflat,
                "logz": logz[cid].index_select(0, rid).float().contiguous(),
                "lam": lam[cid].index_select(0, rid).float().contiguous(),
                "trace": trace[cid].index_select(0, rid).float().contiguous(),
                "captured_trace": captured_trace[cid].index_select(0, rid).float().contiguous(),
                "value": value[cid].index_select(0, rid).to(projection_dtype).contiguous(),
            })
        start = stop

    return {
        "order": order.contiguous(),
        "groups": groups,
        "roots": roots,
        "dim": d,
        "projection_dtype": projection_dtype,
    }


@torch.inference_mode()
def htrc_root_gemm_attention(
    q: torch.Tensor,
    prepared,
) -> torch.Tensor:
    """All-roots-accepted HTRC evaluator using grouped GEMMs.

    This implements the exact current rank8+1.10*tail accepted-root mass law,
    but deliberately performs no child traversal. It is the lower-bound/root
    stage used to replace 80 serial root projections in the original Triton DFS.
    """
    if q.ndim != 2:
        raise ValueError("HTRC root GEMM q must be [Q,D]")
    qn, d = (int(v) for v in q.shape)
    if d != int(prepared["dim"]):
        raise ValueError("HTRC root GEMM D mismatch")

    order = prepared["order"]
    if int(order.numel()) != qn:
        raise ValueError("HTRC root GEMM Q mismatch")
    dtype = prepared["projection_dtype"]
    roots = int(prepared["roots"])

    qsorted = q.index_select(0, order).to(dtype)
    out_sorted = torch.empty((qn, d), dtype=dtype, device=q.device)

    for g in prepared["groups"]:
        s = int(g["start"])
        e = int(g["stop"])
        dq = qsorted[s:e] - g["center"][None, :]

        # Two tensor-core-friendly GEMMs replace 80*(1+8) serial dot products.
        linear = (dq @ g["grad_t"]).float()                     # [Qc,R]
        proj = (dq @ g["u_flat"]).float().reshape(e - s, roots, 8)
        proj2 = proj.square()
        q2 = (proj2 * g["lam"][None]).sum(-1)
        residual_norm2 = (
            dq.float().square().sum(-1, keepdim=True)
            - proj2.sum(-1)
        ).clamp_min_(0.0)
        tail = (
            (g["trace"] - g["captured_trace"])[None]
            * residual_norm2
            / float(max(d - 8, 1))
        )
        mass = g["logz"][None] + linear + 0.5 * q2 + 1.10 * tail
        weights = torch.softmax(mass, dim=1).to(dtype)
        out_sorted[s:e] = weights @ g["value"]

    out = torch.empty_like(out_sorted)
    out.index_copy_(0, order, out_sorted)
    return out.float()




@triton.jit
def _compact_source_reduce_kernel(
    k,
    v,
    out_k,
    out_v,
    out_log_count,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP_TOKENS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Reduce one contiguous K/V chunk into fixed-size compact sources.

    Grid is [groups_in_chunk, heads]. Each program owns one source/head and
    accumulates D channels across GROUP_TOKENS rows.
    """
    gid = tl.program_id(0)
    hidx = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dmask = d < D

    start = gid * GROUP_TOKENS
    count = tl.minimum(GROUP_TOKENS, ROWS - start)
    ksum = tl.zeros((BLOCK_D,), dtype=tl.float32)
    vsum = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for r in tl.static_range(0, GROUP_TOKENS):
        valid = r < count
        off = ((start + r) * H + hidx) * D + d
        kval = tl.load(k + off, mask=valid & dmask, other=0.0).to(tl.float32)
        vval = tl.load(v + off, mask=valid & dmask, other=0.0).to(tl.float32)
        ksum += kval
        vsum += vval

    inv = 1.0 / tl.maximum(count.to(tl.float32), 1.0)
    ooff = (gid * H + hidx) * D + d
    tl.store(out_k + ooff, ksum * inv, mask=dmask)
    tl.store(out_v + ooff, vsum * inv, mask=dmask)
    if hidx == 0:
        tl.store(out_log_count + gid, tl.log(count.to(tl.float32)))


def compact_source_reduce(
    k: torch.Tensor,
    v: torch.Tensor,
    group_tokens: int,
):
    """Triton K/V reduction for one streaming chunk.

    k/v: [T,H,D], contiguous
    returns fp32 source_k/source_v [G,H,D] and log_count [G].
    """
    if k.ndim != 3 or v.shape != k.shape:
        raise ValueError("compact_source_reduce expects matching [T,H,D]")
    rows, h, d = (int(x) for x in k.shape)
    group_tokens = int(group_tokens)
    if group_tokens <= 0 or group_tokens > 1024:
        raise ValueError("group_tokens must be in 1..1024")
    if d <= 0 or d > 256:
        raise ValueError(f"compact source reduction supports D<=256, got {d}")
    if v.device != k.device:
        raise ValueError("compact source reduction tensors must share device")

    g = (rows + group_tokens - 1) // group_tokens
    if k.device.type != "cuda":
        pad = g * group_tokens - rows
        kf = k.float()
        vf = v.float()
        if pad:
            kf = torch.cat(
                [kf, torch.zeros((pad, h, d), dtype=kf.dtype, device=kf.device)],
                dim=0,
            )
            vf = torch.cat(
                [vf, torch.zeros((pad, h, d), dtype=vf.dtype, device=vf.device)],
                dim=0,
            )
        counts = torch.full((g,), float(group_tokens), dtype=torch.float32, device=k.device)
        if rows % group_tokens:
            counts[-1] = float(rows % group_tokens)
        sk = kf.reshape(g, group_tokens, h, d).sum(1) / counts[:, None, None]
        sv = vf.reshape(g, group_tokens, h, d).sum(1) / counts[:, None, None]
        return sk, sv, counts.log()

    _ensure_windows_triton_dll_search()
    kc = k.contiguous()
    vc = v.contiguous()
    sk = torch.empty((g, h, d), dtype=torch.float32, device=k.device)
    sv = torch.empty_like(sk)
    lc = torch.empty((g,), dtype=torch.float32, device=k.device)
    block_d = triton.next_power_of_2(d)
    _compact_source_reduce_kernel[(g, h)](
        kc,
        vc,
        sk,
        sv,
        lc,
        ROWS=rows,
        H=h,
        D=d,
        GROUP_TOKENS=group_tokens,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return sk, sv, lc


@triton.jit
def _compact_source_attention_kernel(
    q,
    source_k,
    source_v,
    source_log_count,
    out,
    Q: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    G,
    BLOCK_D: tl.constexpr,
):
    """Fused compact-source softmax/value attention.

    One program owns one [head, query] row. It streams over compact sources,
    performs online log-sum-exp, and accumulates the value numerator without
    materializing [H,Q,G] logits or probabilities.
    """
    row = tl.program_id(0)
    qidx = row % Q
    hidx = row // Q

    d = tl.arange(0, BLOCK_D)
    dmask = d < D
    qoff = (qidx * H + hidx) * D + d
    qv = tl.load(q + qoff, mask=dmask, other=0.0).to(tl.float32)

    m = -float("inf")
    z = 0.0
    numer = tl.zeros((BLOCK_D,), dtype=tl.float32)
    scale = 1.0 / tl.sqrt(float(D))

    for g in tl.range(0, G):
        soff = (g * H + hidx) * D + d
        kv = tl.load(source_k + soff, mask=dmask, other=0.0).to(tl.float32)
        vv = tl.load(source_v + soff, mask=dmask, other=0.0).to(tl.float32)
        log_count = tl.load(source_log_count + g).to(tl.float32)

        score = tl.sum(qv * kv, axis=0) * scale + log_count
        nm = tl.maximum(m, score)
        old_scale = tl.exp(m - nm)
        new_scale = tl.exp(score - nm)
        numer = numer * old_scale + vv * new_scale
        z = z * old_scale + new_scale
        m = nm

    outv = numer / tl.maximum(z, 1.0e-20)
    ooff = (qidx * H + hidx) * D + d
    tl.store(out + ooff, outv, mask=dmask)


def compact_source_attention(
    q: torch.Tensor,
    source_k: torch.Tensor,
    source_v: torch.Tensor,
    source_log_count: torch.Tensor,
) -> torch.Tensor:
    """Fused CUDA evaluator for Q against compact source summaries.

    q: [Q,H,D]
    source_k/source_v: [G,H,D], fp32 or bf16
    source_log_count: [G], fp32
    returns fp32 [Q,H,D]
    """
    if q.ndim != 3:
        raise ValueError("compact_source_attention expects q=[Q,H,D]")
    if source_k.ndim != 3 or source_v.shape != source_k.shape:
        raise ValueError("compact sources must be matching [G,H,D] tensors")
    qn, h, d = (int(v) for v in q.shape)
    g, hs, ds = (int(v) for v in source_k.shape)
    if (hs, ds) != (h, d):
        raise ValueError("compact source H/D do not match q")
    if source_log_count.shape != (g,):
        raise ValueError("source_log_count must be [G]")
    if any(t.device != q.device for t in (source_k, source_v, source_log_count)):
        raise ValueError("compact attention tensors must share one device")
    if d <= 0 or d > 256:
        raise ValueError(f"compact attention supports D<=256, got {d}")

    if q.device.type != "cuda":
        logits = torch.einsum(
            "qhd,ghd->qhg", q.float(), source_k.float()
        ) / math.sqrt(float(d))
        logits = logits + source_log_count.float()[None, None, :]
        p = torch.softmax(logits, dim=-1)
        return torch.einsum("qhg,ghd->qhd", p, source_v.float())

    _ensure_windows_triton_dll_search()
    qc = q.contiguous()
    kc = source_k.contiguous()
    vc = source_v.contiguous()
    lc = source_log_count.float().contiguous()
    out = torch.empty((qn, h, d), dtype=torch.float32, device=q.device)
    block_d = triton.next_power_of_2(d)
    _compact_source_attention_kernel[(qn * h,)](
        qc,
        kc,
        vc,
        lc,
        out,
        Q=qn,
        H=h,
        D=d,
        G=g,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@triton.jit
def _antipodal_sq_mean_kernel(
    s,
    pair_m,
    pair_d,
    out_mu,
    out_log_pair_z,
    H: tl.constexpr,
    Q: tl.constexpr,
):
    """One program per [block, head, query] row; fixed K=P=8."""
    row = tl.program_id(0)
    qidx = row % Q
    bh = row // Q
    hidx = bh % H
    bidx = bh // H

    k = tl.arange(0, 8)[None, :]
    p = tl.arange(0, 8)[:, None]

    sv = tl.load(s + row * 8 + tl.arange(0, 8))
    base = (bidx * H + hidx) * 64
    mv = tl.load(pair_m + base + p * 8 + k)
    dv = tl.load(pair_d + base + p * 8 + k)

    a = tl.sum(mv * sv[None, :], axis=1)
    hh = tl.sum(dv * sv[None, :], axis=1)

    # Positive center surrogate used by the validated capture gate.
    half = 1.0 + 0.5 * a
    ew = half * half + 1.0e-6
    dew = half

    eh = tl.exp(hh)
    emh = tl.exp(-hh)
    ch = 0.5 * (eh + emh)
    sh = 0.5 * (eh - emh)

    z = tl.sum(ew * ch, axis=0)
    zsafe = tl.maximum(z, 1.0e-20)

    # dK/ds for e(a)=(1+a/2)^2:
    # sum_p [ e(m cosh + d sinh) + e'(a)m cosh ] / Z
    c_m = ch * (ew + dew)
    c_d = sh * ew
    num = tl.sum(
        c_m[:, None] * mv + c_d[:, None] * dv,
        axis=0,
    )
    mu = num / zsafe

    tl.store(out_mu + row * 8 + tl.arange(0, 8), mu)
    tl.store(out_log_pair_z + row, 0.6931471805599453 + tl.log(zsafe))


def antipodal_sq_mean(
    s: torch.Tensor,
    pair_m: torch.Tensor,
    pair_d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused positive-square antipodal CGF mean for RRC3 reduced rows.

    s: [B,H,Q,8]
    pair_m/pair_d: [B,H,8,8]
    Returns (mu [B,H,Q,8], log_pair_z [B,H,Q]).
    """
    if s.ndim != 4 or s.shape[-1] != 8:
        raise ValueError("RRC3 antipodal kernel expects s as [B,H,Q,8]")
    if pair_m.ndim != 4 or tuple(pair_m.shape[-2:]) != (8, 8):
        raise ValueError("RRC3 antipodal kernel expects pair_m as [B,H,8,8]")
    if pair_d.shape != pair_m.shape:
        raise ValueError("RRC3 antipodal pair_m/pair_d shape mismatch")
    if tuple(s.shape[:2]) != tuple(pair_m.shape[:2]):
        raise ValueError("RRC3 antipodal batch/head dimensions mismatch")
    if s.dtype != torch.float32 or pair_m.dtype != torch.float32 or pair_d.dtype != torch.float32:
        raise TypeError("RRC3 antipodal kernel requires fp32 tensors")
    if s.device != pair_m.device or s.device != pair_d.device:
        raise ValueError("RRC3 antipodal tensors must share one device")

    b, h, q, _ = (int(v) for v in s.shape)
    rows = b * h * q
    out_mu = torch.empty_like(s)
    out_logz = torch.empty((b, h, q), dtype=torch.float32, device=s.device)
    if rows == 0:
        return out_mu, out_logz

    if s.device.type != "cuda":
        a = torch.einsum("bhqk,bhpk->bhqp", s, pair_m)
        hh = torch.einsum("bhqk,bhpk->bhqp", s, pair_d)
        ch = torch.cosh(hh)
        sh = torch.sinh(hh)
        half = 1.0 + 0.5 * a
        ew = half.square() + 1e-6
        dew = half
        z = (ew * ch).sum(dim=-1).clamp_min(1e-20)
        num = (
            (ch * (ew + dew))[..., None] * pair_m[:, :, None, :, :]
            + (sh * ew)[..., None] * pair_d[:, :, None, :, :]
        ).sum(dim=-2)
        out_mu.copy_(num / z[..., None])
        out_logz.copy_(math.log(2.0) + torch.log(z))
        return out_mu, out_logz

    _ensure_windows_triton_dll_search()
    if not s.is_contiguous() or not pair_m.is_contiguous() or not pair_d.is_contiguous():
        raise ValueError("RRC3 antipodal kernel requires contiguous tensors")
    _antipodal_sq_mean_kernel[(rows,)](
        s,
        pair_m,
        pair_d,
        out_mu,
        out_logz,
        H=h,
        Q=q,
        num_warps=1,
    )
    return out_mu, out_logz


@triton.jit
def _interior_lift_kernel(
    p,
    anchor,
    status,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    t = tl.arange(0, 16)
    mask = row < ROWS

    p0 = tl.load(p + row * 16 + t, mask=mask, other=0.0)
    pa = tl.load(anchor + row * 16 + t, mask=mask, other=0.0)

    min0 = tl.min(p0, axis=0)
    mina = tl.min(pa, axis=0)
    fp32_max = 3.4028234663852886e38
    finite0 = tl.sum((tl.abs(p0) <= fp32_max).to(tl.int32), axis=0) == 16
    finitea = tl.sum((tl.abs(pa) <= fp32_max).to(tl.int32), axis=0) == 16
    already = finite0 & (min0 > 1.0e-8)

    target = 1.0e-8 + 0.02 * tl.maximum(mina - 1.0e-8, 0.0)
    denom = mina - min0
    predictor_ok = finite0 & finitea & (mina > 1.0e-8) & (denom > 1.0e-20)
    lam = tl.maximum(
        0.0,
        tl.minimum(1.0, (target - min0) / tl.maximum(denom, 1.0e-20)),
    )
    cand = p0 + lam * (pa - p0)
    cand_min = tl.min(cand, axis=0)
    lifted = (~already) & predictor_ok & (cand_min > 1.0e-9)

    out = tl.where(lifted, cand, p0)
    tl.store(p + row * 16 + t, out, mask=mask)

    # 0=unresolved, 1=already interior, 2=lifted by exact-k8 anchor.
    code = tl.where(already, 1, tl.where(lifted, 2, 0))
    tl.store(status + row, code, mask=mask)


def interior_lift_(
    p: torch.Tensor,
    anchor: torch.Tensor,
) -> torch.Tensor:
    """In-place exact-k8 interior lift for flattened [R,16] probability rows.

    Returns uint8 status per row: 0 unresolved, 1 already interior, 2 lifted.
    """
    if p.ndim != 2 or tuple(p.shape[1:]) != (16,):
        raise ValueError("RRC3 interior lift expects p as [R,16]")
    if anchor.shape != p.shape:
        raise ValueError("RRC3 interior lift anchor shape mismatch")
    if p.dtype != torch.float32 or anchor.dtype != torch.float32:
        raise TypeError("RRC3 interior lift requires fp32 tensors")
    if p.device != anchor.device:
        raise ValueError("RRC3 interior lift tensors must share one device")

    rows = int(p.shape[0])
    status = torch.empty((rows,), dtype=torch.uint8, device=p.device)
    if rows == 0:
        return status

    if p.device.type != "cuda":
        min0 = p.amin(dim=-1)
        mina = anchor.amin(dim=-1)
        finite0 = torch.isfinite(p).all(dim=-1)
        finitea = torch.isfinite(anchor).all(dim=-1)
        already = finite0 & (min0 > 1e-8)
        target = 1e-8 + 0.02 * (mina - 1e-8).clamp_min(0.0)
        denom = mina - min0
        predictor_ok = (
            finite0
            & finitea
            & (mina > 1e-8)
            & (denom > 1e-20)
        )
        lam = ((target - min0) / denom.clamp_min(1e-20)).clamp(0.0, 1.0)
        candidate = p + lam[:, None] * (anchor - p)
        lifted = (
            (~already)
            & predictor_ok
            & torch.isfinite(candidate).all(dim=-1)
            & (candidate.amin(dim=-1) > 1e-9)
        )
        if bool(lifted.any()):
            p[lifted] = candidate[lifted]
        status.zero_()
        status[already] = 1
        status[lifted] = 2
        return status

    _ensure_windows_triton_dll_search()
    if not p.is_contiguous() or not anchor.is_contiguous():
        raise ValueError("RRC3 interior lift requires contiguous tensors")
    _interior_lift_kernel[(rows,)](
        p,
        anchor,
        status,
        ROWS=rows,
        num_warps=1,
    )
    return status





@triton.jit
def _weighted_value_reduce_kernel(
    p,
    block_w,
    v,
    out,
    B: tl.constexpr,
    H: tl.constexpr,
    Q: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hq = tl.program_id(0)
    h = hq // Q
    q = hq - h * Q
    d = tl.arange(0, BLOCK_D)
    dmask = d < D

    acc = tl.zeros((BLOCK_D,), tl.float32)
    for b in range(0, B):
        w = tl.load(block_w + (b * H + h) * Q + q)
        tok = tl.arange(0, 16)
        pv = tl.load(
            p + (((b * H + h) * Q + q) * 16) + tok
        )
        for t in range(0, 16):
            pt = tl.sum(pv * (tok == t).to(tl.float32), axis=0)
            vv = tl.load(
                v + (((b * H + h) * 16 + t) * D) + d,
                mask=dmask,
                other=0.0,
            )
            acc += (w * pt) * vv

    tl.store(out + hq * D + d, acc, mask=dmask)


def weighted_value_reduce(
    p: torch.Tensor,
    block_w: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Reduce weighted block-local values directly to [H,Q,D].

    p: [B,H,Q,16], block_w: [B,H,Q], v: [B,H,16,D].
    Avoids materializing weighted_p=[B,H,Q,16].
    """
    if p.ndim != 4 or p.shape[-1] != 16:
        raise ValueError("RRC3 weighted value reduce expects p as [B,H,Q,16]")
    if block_w.shape != p.shape[:-1]:
        raise ValueError("RRC3 weighted value reduce block_w shape mismatch")
    if v.ndim != 4 or tuple(v.shape[:3]) != (
        int(p.shape[0]), int(p.shape[1]), 16
    ):
        raise ValueError("RRC3 weighted value reduce expects v as [B,H,16,D]")
    if p.dtype != torch.float32 or block_w.dtype != torch.float32 or v.dtype != torch.float32:
        raise TypeError("RRC3 weighted value reduce requires fp32 tensors")
    if p.device != block_w.device or p.device != v.device:
        raise ValueError("RRC3 weighted value reduce tensors must share one device")

    b, h, q, _ = (int(x) for x in p.shape)
    d = int(v.shape[-1])
    if p.device.type != "cuda":
        return torch.einsum("bhq,bhqt,bhtd->hqd", block_w, p, v)

    _ensure_windows_triton_dll_search()
    if not p.is_contiguous() or not block_w.is_contiguous() or not v.is_contiguous():
        raise ValueError("RRC3 weighted value reduce requires contiguous tensors")
    out = torch.empty((h, q, d), dtype=torch.float32, device=p.device)
    block_d = triton.next_power_of_2(d)
    _weighted_value_reduce_kernel[(h * q,)](
        p,
        block_w,
        v,
        out,
        B=b,
        H=h,
        Q=q,
        D=d,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


@triton.jit
def _newton_cg_kernel(
    p,
    n,
    out_p,
    status,
    ROWS: tl.constexpr,
    REG: tl.constexpr,
):
    row = tl.program_id(0)
    t = tl.arange(0, 16)[:, None]
    r = tl.arange(0, 8)[None, :]
    rmask = r < 7

    pv = tl.load(p + row * 16 + tl.arange(0, 16))
    safe = tl.maximum(pv, 1.0e-9)
    nv = tl.load(
        n + row * (16 * 7) + t * 7 + r,
        mask=rmask,
        other=0.0,
    )

    # f = N^T log(p)
    logp = tl.log(safe)[:, None]
    f = tl.sum(logp * nv, axis=0)

    # Fixed-dimension CG for (N^T diag(1/p) N + REG I) d = f.
    dvec = tl.zeros((8,), tl.float32)
    rr = f
    search = rr
    rr_old = tl.sum(rr * rr, axis=0)

    for _ in tl.static_range(0, 7):
        # J * search without materializing J.
        ns = tl.sum(nv * search[None, :], axis=1)
        weighted = (ns / safe)[:, None]
        jv = tl.sum(nv * weighted, axis=0) + REG * search
        denom = tl.sum(search * jv, axis=0)
        alpha_cg = rr_old / tl.maximum(denom, 1.0e-20)
        dvec = dvec + alpha_cg * search
        rr_new = rr - alpha_cg * jv
        rr_new_norm = tl.sum(rr_new * rr_new, axis=0)
        beta = rr_new_norm / tl.maximum(rr_old, 1.0e-20)
        search = rr_new + beta * search
        rr = rr_new
        rr_old = rr_new_norm

    nd = tl.sum(nv * dvec[None, :], axis=1)
    dp = -nd
    ratio = tl.where(
        dp < 0.0,
        (pv - 1.0e-9) / tl.maximum(-dp, 1.0e-9),
        float("inf"),
    )
    step = tl.minimum(1.0, 0.99 * tl.min(ratio, axis=0))
    step = tl.maximum(step, 0.0)
    candidate = pv + step * dp

    # Relative residual of the CG solve. Recompute Jd-f once at the end.
    nd2 = tl.sum(nv * dvec[None, :], axis=1)
    weighted2 = (nd2 / safe)[:, None]
    jd = tl.sum(nv * weighted2, axis=0) + REG * dvec
    res = jd - f
    res2 = tl.sum(res * res, axis=0)
    f2 = tl.sum(f * f, axis=0)
    rel2 = res2 / tl.maximum(f2, 1.0e-20)

    fp32_max = 3.4028234663852886e38
    finite = tl.sum((tl.abs(candidate) <= fp32_max).to(tl.int32), axis=0) == 16
    positive = tl.min(candidate, axis=0) > 1.0e-9
    # 1e-6 relative residual tolerance squared. Rows outside this bound use
    # the exact torch.linalg.solve fallback.
    ok = finite & positive & (rel2 <= 1.0e-12)

    offs = row * 16 + tl.arange(0, 16)
    tl.store(out_p + offs, candidate)
    tl.store(status + row, ok.to(tl.uint8))


def newton_cg(
    p: torch.Tensor,
    n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One entropy-nullspace Newton step using fixed 7-step CG on CUDA.

    Returns (candidate, status). status==1 means the CG residual/positivity gate
    accepted the candidate; status==0 must use the exact reference solve.
    """
    if p.ndim != 2 or p.shape[1] != 16:
        raise ValueError("RRC3 Newton CG expects p as [R,16]")
    if n.ndim != 3 or tuple(n.shape[1:]) != (16, 7):
        raise ValueError("RRC3 Newton CG expects n as [R,16,7]")
    if int(n.shape[0]) != int(p.shape[0]):
        raise ValueError("RRC3 Newton CG row count mismatch")
    if p.dtype != torch.float32 or n.dtype != torch.float32:
        raise TypeError("RRC3 Newton CG requires fp32 tensors")
    if p.device != n.device:
        raise ValueError("RRC3 Newton CG tensors must share one device")

    rows = int(p.shape[0])
    out = torch.empty_like(p)
    status = torch.empty((rows,), dtype=torch.uint8, device=p.device)
    if rows == 0:
        return out, status

    if p.device.type != "cuda":
        # CPU path is intentionally reference-only.
        safe = p.clamp_min(1e-9)
        f = torch.einsum("mt,mtr->mr", torch.log(safe), n)
        invp = 1.0 / safe
        jac = torch.einsum("mti,mt,mtj->mij", n, invp, n)
        eye = torch.eye(7, dtype=p.dtype, device=p.device)
        d = torch.linalg.solve(
            jac + 1e-8 * eye[None],
            f.unsqueeze(-1),
        ).squeeze(-1)
        dp = -torch.einsum("mr,mtr->mt", d, n)
        ratio = torch.where(
            dp < 0.0,
            (p - 1e-9) / (-dp).clamp_min(1e-9),
            torch.full_like(dp, float("inf")),
        )
        alpha = torch.minimum(
            torch.ones_like(p[:, 0]),
            0.99 * ratio.amin(dim=-1),
        ).clamp_min(0.0)
        out.copy_(p + alpha[:, None] * dp)
        status.fill_(1)
        return out, status

    _ensure_windows_triton_dll_search()
    if not p.is_contiguous() or not n.is_contiguous():
        raise ValueError("RRC3 Newton CG requires contiguous tensors")
    _newton_cg_kernel[(rows,)](
        p,
        n,
        out,
        status,
        ROWS=rows,
        REG=1.0e-8,
        num_warps=1,
    )
    return out, status




@triton.jit
def _polynomial_candidate_kernel(
    mu,
    uk,
    out_p,
    out_paff,
    H: tl.constexpr,
    Q: tl.constexpr,
):
    row = tl.program_id(0)
    q = row % Q
    bh = row // Q
    hidx = bh % H
    bidx = bh // H

    t = tl.arange(0, 16)[:, None]
    k = tl.arange(0, 8)[None, :]
    mu_vec = tl.load(mu + row * 8 + tl.arange(0, 8))
    uk_base = ((bidx * H + hidx) * 16 * 8)
    uk_mat = tl.load(uk + uk_base + t * 8 + k)

    # z = sqrt(16) * uk = 4 * uk
    htok = 4.0 * tl.sum(uk_mat * mu_vec[None, :], axis=1)
    rho2 = tl.sum(mu_vec * mu_vec, axis=0)

    # Pn(y) = y - mean(y) - (y uk) uk^T
    y2 = htok * htok
    c2 = y2 - tl.sum(y2, axis=0) / 16.0
    coeff2 = tl.sum(c2[:, None] * uk_mat, axis=0)
    proj2 = c2 - tl.sum(uk_mat * coeff2[None, :], axis=1)
    x2 = 0.5 * proj2

    y3 = htok * x2 - (htok * htok * htok) / 3.0
    c3 = y3 - tl.sum(y3, axis=0) / 16.0
    coeff3 = tl.sum(c3[:, None] * uk_mat, axis=0)
    x3 = c3 - tl.sum(uk_mat * coeff3[None, :], axis=1)

    paff = (1.0 + htok) / 16.0
    p = paff + (
        x2 / (1.0 + rho2 / 9.0)
        + x3 / (1.0 + 1.5 * rho2)
    ) / 16.0

    offs = row * 16 + tl.arange(0, 16)
    tl.store(out_p + offs, p)
    tl.store(out_paff + offs, paff)



@triton.jit
def _polynomial_candidate_p_kernel(
    mu,
    uk,
    out_p,
    H: tl.constexpr,
    Q: tl.constexpr,
):
    row = tl.program_id(0)
    q = row % Q
    bh = row // Q
    hidx = bh % H
    bidx = bh // H

    t = tl.arange(0, 16)[:, None]
    k = tl.arange(0, 8)[None, :]
    mu_vec = tl.load(mu + row * 8 + tl.arange(0, 8))
    uk_base = ((bidx * H + hidx) * 16 * 8)
    uk_mat = tl.load(uk + uk_base + t * 8 + k)

    htok = 4.0 * tl.sum(uk_mat * mu_vec[None, :], axis=1)
    rho2 = tl.sum(mu_vec * mu_vec, axis=0)

    y2 = htok * htok
    c2 = y2 - tl.sum(y2, axis=0) / 16.0
    coeff2 = tl.sum(c2[:, None] * uk_mat, axis=0)
    proj2 = c2 - tl.sum(uk_mat * coeff2[None, :], axis=1)
    x2 = 0.5 * proj2

    y3 = htok * x2 - (htok * htok * htok) / 3.0
    c3 = y3 - tl.sum(y3, axis=0) / 16.0
    coeff3 = tl.sum(c3[:, None] * uk_mat, axis=0)
    x3 = c3 - tl.sum(uk_mat * coeff3[None, :], axis=1)

    paff = (1.0 + htok) / 16.0
    p = paff + (
        x2 / (1.0 + rho2 / 9.0)
        + x3 / (1.0 + 1.5 * rho2)
    ) / 16.0

    offs = row * 16 + tl.arange(0, 16)
    tl.store(out_p + offs, p)


def polynomial_candidate_p(
    mu: torch.Tensor,
    uk: torch.Tensor,
) -> torch.Tensor:
    """Fused RRC3 polynomial p only; paff is reconstructed on rare fallback."""
    if mu.ndim != 4 or mu.shape[-1] != 8:
        raise ValueError("RRC3 polynomial expects mu as [B,H,Q,8]")
    if uk.ndim != 4 or tuple(uk.shape[-2:]) != (16, 8):
        raise ValueError("RRC3 polynomial expects uk as [B,H,16,8]")
    if tuple(mu.shape[:2]) != tuple(uk.shape[:2]):
        raise ValueError("RRC3 polynomial batch/head dimensions mismatch")
    if mu.dtype != torch.float32 or uk.dtype != torch.float32:
        raise TypeError("RRC3 polynomial requires fp32 tensors")
    if mu.device != uk.device:
        raise ValueError("RRC3 polynomial tensors must share one device")

    b, h, q, _ = (int(v) for v in mu.shape)
    rows = b * h * q
    out_p = torch.empty((rows, 16), dtype=torch.float32, device=mu.device)
    if rows == 0:
        return out_p

    if mu.device.type != "cuda":
        z = 4.0 * uk
        htok = torch.einsum("bhqk,bhtk->bhqt", mu, z)
        rho2 = torch.einsum("bhqk,bhqk->bhq", mu, mu)

        def project(y: torch.Tensor) -> torch.Tensor:
            centered = y - y.mean(dim=-1, keepdim=True)
            coeff = torch.einsum("bhqt,bhtk->bhqk", centered, uk)
            return centered - torch.einsum("bhqk,bhtk->bhqt", coeff, uk)

        x2 = 0.5 * project(htok.square())
        x3 = project(htok * x2 - htok.pow(3) / 3.0)
        paff = (1.0 + htok) / 16.0
        p = paff + (
            x2 / (1.0 + rho2[..., None] / 9.0)
            + x3 / (1.0 + 1.5 * rho2[..., None])
        ) / 16.0
        out_p.copy_(p.reshape(rows, 16))
        return out_p

    _ensure_windows_triton_dll_search()
    if not mu.is_contiguous() or not uk.is_contiguous():
        raise ValueError("RRC3 polynomial requires contiguous tensors")
    _polynomial_candidate_p_kernel[(rows,)](
        mu,
        uk,
        out_p,
        H=h,
        Q=q,
        num_warps=1,
    )
    return out_p


def polynomial_candidate(
    mu: torch.Tensor,
    uk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RRC3 polynomial candidate for [B,H,Q,8] and [B,H,16,8]."""
    if mu.ndim != 4 or mu.shape[-1] != 8:
        raise ValueError("RRC3 polynomial expects mu as [B,H,Q,8]")
    if uk.ndim != 4 or tuple(uk.shape[-2:]) != (16, 8):
        raise ValueError("RRC3 polynomial expects uk as [B,H,16,8]")
    if tuple(mu.shape[:2]) != tuple(uk.shape[:2]):
        raise ValueError("RRC3 polynomial batch/head dimensions mismatch")
    if mu.dtype != torch.float32 or uk.dtype != torch.float32:
        raise TypeError("RRC3 polynomial requires fp32 tensors")
    if mu.device != uk.device:
        raise ValueError("RRC3 polynomial tensors must share one device")

    b, h, q, _ = (int(v) for v in mu.shape)
    rows = b * h * q
    out_p = torch.empty((rows, 16), dtype=torch.float32, device=mu.device)
    out_paff = torch.empty_like(out_p)
    if rows == 0:
        return out_p, out_paff

    if mu.device.type != "cuda":
        z = 4.0 * uk
        htok = torch.einsum("bhqk,bhtk->bhqt", mu, z)
        rho2 = torch.einsum("bhqk,bhqk->bhq", mu, mu)

        def project(y: torch.Tensor) -> torch.Tensor:
            centered = y - y.mean(dim=-1, keepdim=True)
            coeff = torch.einsum("bhqt,bhtk->bhqk", centered, uk)
            return centered - torch.einsum("bhqk,bhtk->bhqt", coeff, uk)

        x2 = 0.5 * project(htok.square())
        x3 = project(htok * x2 - htok.pow(3) / 3.0)
        paff = (1.0 + htok) / 16.0
        p = paff + (
            x2 / (1.0 + rho2[..., None] / 9.0)
            + x3 / (1.0 + 1.5 * rho2[..., None])
        ) / 16.0
        out_p.copy_(p.reshape(rows, 16))
        out_paff.copy_(paff.reshape(rows, 16))
        return out_p, out_paff

    _ensure_windows_triton_dll_search()
    if not mu.is_contiguous() or not uk.is_contiguous():
        raise ValueError("RRC3 polynomial requires contiguous tensors")
    _polynomial_candidate_kernel[(rows,)](
        mu,
        uk,
        out_p,
        out_paff,
        H=h,
        Q=q,
        num_warps=1,
    )
    return out_p, out_paff




@triton.jit
def _newton_ldlt_kernel(
    p,
    n,
    active,
    status,
    ROWS: tl.constexpr,
    REG: tl.constexpr,
):
    row = tl.program_id(0)
    tok = tl.arange(0, 16)
    ridx = tl.arange(0, 8)
    ri = ridx[:, None]
    cj = ridx[None, :]

    pv = tl.load(p + row * 16 + tok)
    row_active = tl.load(active + row) != 0
    safe = tl.maximum(pv, 1.0e-9)

    # Load N as [16,8] with the padded eighth column held at zero.
    t2 = tok[:, None]
    r2 = ridx[None, :]
    nv = tl.load(
        n + row * (16 * 7) + t2 * 7 + r2,
        mask=r2 < 7,
        other=0.0,
    )

    logp = tl.log(safe)
    fvec = tl.sum(logp[:, None] * nv, axis=0)

    # J = N^T diag(1/p) N + REG I, padded to 8x8. The padded lane is
    # identity and is never part of the seven-dimensional solve.
    weighted_n = nv / safe[:, None]
    a = tl.sum(
        nv[:, :, None] * weighted_n[:, None, :],
        axis=0,
    )
    valid_mat = (ri < 7) & (cj < 7)
    diag_mask = ri == cj
    a = tl.where(
        valid_mat,
        a + tl.where(diag_mask, REG, 0.0),
        tl.where(diag_mask, 1.0, 0.0),
    )

    # Unit-lower LDL^T factorization. 7 is fixed by RRC3's B=16,k=8
    # normalization+moment nullspace dimension.
    lmat = tl.zeros((8, 8), tl.float32)
    dvec = tl.zeros((8,), tl.float32) + 1.0
    good_factor = (row >= 0)

    for kk in tl.static_range(0, 7):
        lrow_k = tl.sum(
            lmat * (ri == kk).to(tl.float32),
            axis=0,
        )
        prev = tl.sum(
            lmat
            * (lrow_k[None, :] * dvec[None, :])
            * (cj < kk).to(tl.float32),
            axis=1,
        )
        acol = tl.sum(
            a * (cj == kk).to(tl.float32),
            axis=1,
        )
        numer = acol - prev
        pivot = tl.sum(
            numer * (ridx == kk).to(tl.float32),
            axis=0,
        )
        pivot_ok = (pivot > 1.0e-12) & (tl.abs(pivot) <= 3.4028234663852886e38)
        good_factor = good_factor & pivot_ok
        safe_pivot = tl.maximum(pivot, 1.0e-12)

        dvec = tl.where(ridx == kk, safe_pivot, dvec)
        col = numer / safe_pivot
        lmat = tl.where(
            (cj == kk) & (ri > kk),
            col[:, None],
            lmat,
        )
        lmat = tl.where(
            (ri == kk) & (cj == kk),
            1.0,
            lmat,
        )

    # Forward solve L y = f.
    y = tl.zeros((8,), tl.float32)
    for kk in tl.static_range(0, 7):
        lrow = tl.sum(
            lmat * (ri == kk).to(tl.float32),
            axis=0,
        )
        prev = tl.sum(
            lrow * y * (ridx < kk).to(tl.float32),
            axis=0,
        )
        fk = tl.sum(
            fvec * (ridx == kk).to(tl.float32),
            axis=0,
        )
        y = tl.where(ridx == kk, fk - prev, y)

    z = tl.where(ridx < 7, y / dvec, 0.0)

    # Backward solve L^T d = z.
    dsol = tl.zeros((8,), tl.float32)
    for ii in tl.static_range(0, 7):
        kk = 6 - ii
        lcol = tl.sum(
            lmat * (cj == kk).to(tl.float32),
            axis=1,
        )
        nxt = tl.sum(
            lcol * dsol * (ridx > kk).to(tl.float32),
            axis=0,
        )
        zk = tl.sum(
            z * (ridx == kk).to(tl.float32),
            axis=0,
        )
        dsol = tl.where(ridx == kk, zk - nxt, dsol)

    nd = tl.sum(nv * dsol[None, :], axis=1)
    dp = -nd
    ratio = tl.where(
        dp < 0.0,
        (pv - 1.0e-9) / tl.maximum(-dp, 1.0e-9),
        float("inf"),
    )
    step = tl.maximum(
        0.0,
        tl.minimum(1.0, 0.99 * tl.min(ratio, axis=0)),
    )
    candidate = pv + step * dp

    # Check the direct solve against the same J before accepting it.
    ad = tl.sum(a * dsol[None, :], axis=1)
    res = ad - fvec
    active7 = ridx < 7
    res2 = tl.sum(
        tl.where(active7, res * res, 0.0),
        axis=0,
    )
    f2 = tl.sum(
        tl.where(active7, fvec * fvec, 0.0),
        axis=0,
    )
    rel2 = res2 / tl.maximum(f2, 1.0e-20)

    fp32_max = 3.4028234663852886e38
    finite = tl.sum(
        (tl.abs(candidate) <= fp32_max).to(tl.int32),
        axis=0,
    ) == 16
    positive = tl.min(candidate, axis=0) > 1.0e-9
    ok = row_active & good_factor & finite & positive & (rel2 <= 1.0e-10)
    rejected = row_active & (~ok)

    # Keep the original probability row for inactive/rejected rows so the
    # caller can run the exact reference solve only on status==2.
    offs = row * 16 + tok
    tl.store(p + offs, tl.where(ok, candidate, pv))
    # 0=inactive, 1=accepted LDLT, 2=rejected -> exact fallback.
    code = tl.where(ok, 1, tl.where(rejected, 2, 0))
    tl.store(status + row, code.to(tl.uint8))



@triton.jit
def _newton_ldlt_base_n_kernel(
    p,
    nbase,
    active,
    status,
    Q: tl.constexpr,
    REG: tl.constexpr,
):
    geom = tl.program_id(0)
    q = tl.program_id(1)
    row = geom * Q + q

    tok = tl.arange(0, 16)
    ridx = tl.arange(0, 8)
    ri = ridx[:, None]
    cj = ridx[None, :]

    pv = tl.load(p + row * 16 + tok)
    row_active = tl.load(active + row) != 0
    safe = tl.maximum(pv, 1.0e-9)

    # N is query-invariant. 2D launch gives geometry and query coordinates
    # directly, so no row//Q addressing and no physical [G,Q,16,7] expansion.
    t2 = tok[:, None]
    r2 = ridx[None, :]
    nv = tl.load(
        nbase + geom * (16 * 7) + t2 * 7 + r2,
        mask=r2 < 7,
        other=0.0,
    )

    logp = tl.log(safe)
    fvec = tl.sum(logp[:, None] * nv, axis=0)
    weighted_n = nv / safe[:, None]
    a = tl.sum(
        nv[:, :, None] * weighted_n[:, None, :],
        axis=0,
    )
    valid_mat = (ri < 7) & (cj < 7)
    diag_mask = ri == cj
    a = tl.where(
        valid_mat,
        a + tl.where(diag_mask, REG, 0.0),
        tl.where(diag_mask, 1.0, 0.0),
    )

    lmat = tl.zeros((8, 8), tl.float32)
    dvec = tl.zeros((8,), tl.float32) + 1.0
    good_factor = q >= 0

    for kk in tl.static_range(0, 7):
        lrow_k = tl.sum(
            lmat * (ri == kk).to(tl.float32),
            axis=0,
        )
        prev = tl.sum(
            lmat
            * (lrow_k[None, :] * dvec[None, :])
            * (cj < kk).to(tl.float32),
            axis=1,
        )
        acol = tl.sum(
            a * (cj == kk).to(tl.float32),
            axis=1,
        )
        numer = acol - prev
        pivot = tl.sum(
            numer * (ridx == kk).to(tl.float32),
            axis=0,
        )
        pivot_ok = (pivot > 1.0e-12) & (tl.abs(pivot) <= 3.4028234663852886e38)
        good_factor = good_factor & pivot_ok
        safe_pivot = tl.maximum(pivot, 1.0e-12)

        dvec = tl.where(ridx == kk, safe_pivot, dvec)
        col = numer / safe_pivot
        lmat = tl.where(
            (cj == kk) & (ri > kk),
            col[:, None],
            lmat,
        )
        lmat = tl.where(
            (ri == kk) & (cj == kk),
            1.0,
            lmat,
        )

    y = tl.zeros((8,), tl.float32)
    for kk in tl.static_range(0, 7):
        lrow = tl.sum(
            lmat * (ri == kk).to(tl.float32),
            axis=0,
        )
        prev = tl.sum(
            lrow * y * (ridx < kk).to(tl.float32),
            axis=0,
        )
        fk = tl.sum(
            fvec * (ridx == kk).to(tl.float32),
            axis=0,
        )
        y = tl.where(ridx == kk, fk - prev, y)

    z = tl.where(ridx < 7, y / dvec, 0.0)

    dsol = tl.zeros((8,), tl.float32)
    for ii in tl.static_range(0, 7):
        kk = 6 - ii
        lcol = tl.sum(
            lmat * (cj == kk).to(tl.float32),
            axis=1,
        )
        nxt = tl.sum(
            lcol * dsol * (ridx > kk).to(tl.float32),
            axis=0,
        )
        zk = tl.sum(
            z * (ridx == kk).to(tl.float32),
            axis=0,
        )
        dsol = tl.where(ridx == kk, zk - nxt, dsol)

    nd = tl.sum(nv * dsol[None, :], axis=1)
    dp = -nd
    ratio = tl.where(
        dp < 0.0,
        (pv - 1.0e-9) / tl.maximum(-dp, 1.0e-9),
        float("inf"),
    )
    step = tl.maximum(
        0.0,
        tl.minimum(1.0, 0.99 * tl.min(ratio, axis=0)),
    )
    candidate = pv + step * dp

    ad = tl.sum(a * dsol[None, :], axis=1)
    res = ad - fvec
    active7 = ridx < 7
    res2 = tl.sum(
        tl.where(active7, res * res, 0.0),
        axis=0,
    )
    f2 = tl.sum(
        tl.where(active7, fvec * fvec, 0.0),
        axis=0,
    )
    rel2 = res2 / tl.maximum(f2, 1.0e-20)

    fp32_max = 3.4028234663852886e38
    finite = tl.sum(
        (tl.abs(candidate) <= fp32_max).to(tl.int32),
        axis=0,
    ) == 16
    positive = tl.min(candidate, axis=0) > 1.0e-9
    ok = row_active & good_factor & finite & positive & (rel2 <= 1.0e-10)
    rejected = row_active & (~ok)

    offs = row * 16 + tok
    tl.store(p + offs, tl.where(ok, candidate, pv))
    code = tl.where(ok, 1, tl.where(rejected, 2, 0))
    tl.store(status + row, code.to(tl.uint8))


def newton_ldlt_base_n_inplace_(
    p: torch.Tensor,
    nbase: torch.Tensor,
    active: torch.Tensor,
    q_rows: int,
) -> torch.Tensor:
    """In-place guarded LDLT using query-invariant N as [G,16,7].

    The CUDA launch grid is [geometry_row, query], avoiding materialization of
    [G,Q,16,7]. CPU falls back to the existing expanded reference wrapper.
    """
    if p.ndim != 2 or p.shape[1] != 16:
        raise ValueError("RRC3 base-N LDLT expects p as [R,16]")
    if nbase.ndim != 3 or tuple(nbase.shape[1:]) != (16, 7):
        raise ValueError("RRC3 base-N LDLT expects nbase as [G,16,7]")
    q_rows = int(q_rows)
    if q_rows <= 0:
        raise ValueError("RRC3 base-N LDLT q_rows must be positive")
    geom_rows = int(nbase.shape[0])
    rows = int(p.shape[0])
    if rows != geom_rows * q_rows:
        raise ValueError(
            "RRC3 base-N LDLT row count must equal geometry_rows*q_rows"
        )
    if active.ndim != 1 or int(active.shape[0]) != rows:
        raise ValueError("RRC3 base-N LDLT active mask row count mismatch")
    if p.dtype != torch.float32 or nbase.dtype != torch.float32:
        raise TypeError("RRC3 base-N LDLT requires fp32 tensors")
    if active.dtype not in (torch.bool, torch.uint8):
        raise TypeError("RRC3 base-N LDLT active mask must be bool/uint8")
    if p.device != nbase.device or p.device != active.device:
        raise ValueError("RRC3 base-N LDLT tensors must share one device")

    if p.device.type != "cuda":
        nexp = (
            nbase[:, None]
            .expand(-1, q_rows, -1, -1)
            .reshape(rows, 16, 7)
            .contiguous()
        )
        return newton_ldlt_inplace_(p, nexp, active)

    _ensure_windows_triton_dll_search()
    if (
        not p.is_contiguous()
        or not nbase.is_contiguous()
        or not active.is_contiguous()
    ):
        raise ValueError("RRC3 base-N LDLT requires contiguous tensors")

    status = torch.empty((rows,), dtype=torch.uint8, device=p.device)
    if rows == 0:
        return status
    _newton_ldlt_base_n_kernel[(geom_rows, q_rows)](
        p,
        nbase,
        active,
        status,
        Q=q_rows,
        REG=1.0e-8,
        num_warps=1,
    )
    return status


def newton_ldlt_inplace_(
    p: torch.Tensor,
    n: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """In-place guarded direct 7x7 entropy-Newton solve.

    status codes: 0 inactive, 1 accepted LDLT, 2 rejected for exact fallback.
    Rejected rows are left unchanged in p.
    """
    if p.ndim != 2 or p.shape[1] != 16:
        raise ValueError("RRC3 Newton LDLT expects p as [R,16]")
    if n.ndim != 3 or tuple(n.shape[1:]) != (16, 7):
        raise ValueError("RRC3 Newton LDLT expects n as [R,16,7]")
    if int(n.shape[0]) != int(p.shape[0]):
        raise ValueError("RRC3 Newton LDLT row count mismatch")
    if active.ndim != 1 or int(active.shape[0]) != int(p.shape[0]):
        raise ValueError("RRC3 Newton LDLT active mask row count mismatch")
    if p.dtype != torch.float32 or n.dtype != torch.float32:
        raise TypeError("RRC3 Newton LDLT requires fp32 tensors")
    if active.dtype not in (torch.bool, torch.uint8):
        raise TypeError("RRC3 Newton LDLT active mask must be bool/uint8")
    if p.device != n.device or p.device != active.device:
        raise ValueError("RRC3 Newton LDLT tensors must share one device")

    rows = int(p.shape[0])
    status = torch.empty((rows,), dtype=torch.uint8, device=p.device)
    if rows == 0:
        return status

    if p.device.type != "cuda":
        status.zero_()
        ai = active.nonzero(as_tuple=False).squeeze(-1)
        if int(ai.numel()) == 0:
            return status
        pa = p[ai]
        na = n[ai]
        safe = pa.clamp_min(1e-9)
        fvec = torch.einsum("mt,mtr->mr", torch.log(safe), na)
        invp = 1.0 / safe
        jac = torch.einsum("mti,mt,mtj->mij", na, invp, na)
        eye = torch.eye(7, dtype=p.dtype, device=p.device)
        d = torch.linalg.solve(
            jac + 1e-8 * eye[None],
            fvec.unsqueeze(-1),
        ).squeeze(-1)
        dp = -torch.einsum("mr,mtr->mt", d, na)
        ratio = torch.where(
            dp < 0.0,
            (pa - 1e-9) / (-dp).clamp_min(1e-9),
            torch.full_like(dp, float("inf")),
        )
        alpha = torch.minimum(
            torch.ones_like(pa[:, 0]),
            0.99 * ratio.amin(dim=-1),
        ).clamp_min(0.0)
        p[ai] = pa + alpha[:, None] * dp
        status[ai] = 1
        return status

    _ensure_windows_triton_dll_search()
    if not p.is_contiguous() or not n.is_contiguous() or not active.is_contiguous():
        raise ValueError("RRC3 Newton LDLT requires contiguous tensors")
    _newton_ldlt_kernel[(rows,)](
        p,
        n,
        active,
        status,
        ROWS=rows,
        REG=1.0e-8,
        num_warps=1,
    )
    return status


__all__ = ["online_merge_", "htrc_frontier_level_value", "htrc_frontier_level_value_packed", "htrc_frontier_expand", "htrc_frontier_step_packed", "htrc_frontier_step_int8_grouped", "htrc_frontier_step_mixed_grouped", "htrc_frontier_step_int4_grouped", "htrc_frontier_step_grouped", "htrc_frontier_eval_grouped", "htrc_frontier_compact", "htrc_root_decision_global", "htrc_mass_post_global", "htrc_mass_post", "prepare_htrc_root_gemm", "htrc_root_gemm_attention", "htrc_tree_attention", "compact_source_reduce", "compact_source_attention", "interior_lift_", "weighted_value_reduce", "newton_cg", "newton_ldlt_inplace_", "newton_ldlt_base_n_inplace_", "polynomial_candidate", "polynomial_candidate_p"]
