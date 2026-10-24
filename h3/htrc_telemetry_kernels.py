from __future__ import annotations

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
    for path in candidates:
        try:
            if path.is_dir():
                _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(str(path)))
        except Exception:
            continue


@triton.jit
def _mark_root_occupancy_kernel(
    assign,
    roots,
    opened,
    evaluated_state,
    accepted_state,
    opened_state,
    accepted_node,
    opened_node,
    N: tl.constexpr,
    R: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    qid = offs // R
    ridx = offs - qid * R
    cid = tl.load(assign + qid, mask=mask, other=0).to(tl.int32)
    nid = tl.load(roots + ridx, mask=mask, other=0).to(tl.int32)
    op = tl.load(opened + offs, mask=mask, other=0).to(tl.int1)
    key = cid * N + nid

    tl.atomic_or(evaluated_state + key, 1, mask=mask)
    tl.atomic_or(opened_state + key, 1, mask=mask & op)
    tl.atomic_or(accepted_state + key, 1, mask=mask & (~op))
    tl.atomic_or(opened_node + nid, 1, mask=mask & op)
    tl.atomic_or(accepted_node + nid, 1, mask=mask & (~op))


def mark_root_occupancy(
    assign: torch.Tensor,
    roots: torch.Tensor,
    opened: torch.Tensor,
    *,
    evaluated_state: torch.Tensor,
    accepted_state: torch.Tensor,
    opened_state: torch.Tensor,
    accepted_node: torch.Tensor,
    opened_node: torch.Tensor,
    nodes: int,
) -> None:
    if assign.device.type != "cuda":
        raise RuntimeError("HTRC telemetry requires CUDA")
    qn = int(assign.numel())
    r = int(roots.numel())
    if opened.shape != (qn, r):
        raise ValueError("opened root mask shape mismatch")
    _ensure_windows_triton_dll_search()
    block = 256
    total = qn * r
    _mark_root_occupancy_kernel[(triton.cdiv(total, block),)](
        assign.to(dtype=torch.int32).contiguous(),
        roots.to(device=assign.device, dtype=torch.int32).contiguous(),
        opened.to(device=assign.device, dtype=torch.int8).contiguous(),
        evaluated_state,
        accepted_state,
        opened_state,
        accepted_node,
        opened_node,
        N=int(nodes),
        R=r,
        TOTAL=total,
        BLOCK=block,
        num_warps=4,
    )


@triton.jit
def _frontier_r6_occupancy_kernel(
    q,
    assign,
    centers,
    frontier_q,
    frontier_node,
    frontier_count,
    totals,
    logz,
    grad,
    lam,
    U,
    child_count,
    evaluated_state,
    accepted_state,
    opened_state,
    accepted_node,
    opened_node,
    next_count,
    tau,
    N: tl.constexpr,
    D: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    fi = tl.program_id(0)
    count = tl.load(frontier_count).to(tl.int32)
    active = fi < count
    qid = tl.load(frontier_q + fi, mask=active, other=0).to(tl.int32)
    nid = tl.load(frontier_node + fi, mask=active, other=0).to(tl.int32)
    cid = tl.load(assign + qid, mask=active, other=0).to(tl.int32)

    d = tl.arange(0, BLOCK_D)
    dmask = active & (d < D)
    qv = tl.load(q + qid * D + d, mask=dmask, other=0.0).to(tl.float32)
    cv = tl.load(
        centers + cid * D + d, mask=dmask, other=0.0
    ).to(tl.float32)
    dq = qv - cv

    base = cid * N + nid
    gv = tl.load(grad + base * D + d, mask=dmask, other=0.0).to(tl.float32)
    linear = tl.sum(dq * gv, axis=0)

    q2 = 0.0
    for r in tl.static_range(0, RANK):
        uv = tl.load(
            U + base * D * RANK + d * RANK + r,
            mask=dmask,
            other=0.0,
        ).to(tl.float32)
        pr = tl.sum(dq * uv, axis=0)
        lv = tl.load(lam + base * RANK + r, mask=active, other=0.0).to(tl.float32)
        q2 += lv * pr * pr

    lz = tl.load(logz + base, mask=active, other=-float("inf")).to(tl.float32)
    route = lz + linear + 0.5 * q2
    cert = tl.abs(0.5 * q2)
    small = tl.log(tl.maximum(tl.exp(cert) - 1.0, 1.0e-30))
    log_unc = tl.where(cert > 20.0, cert, small)
    total = tl.load(totals + qid, mask=active, other=0.0).to(tl.float32)
    score = route - total + log_unc

    cc = tl.load(child_count + nid, mask=active, other=0).to(tl.int32)
    op = active & (cc > 0) & (score > tau)
    ac = active & (~op)

    tl.atomic_or(evaluated_state + base, 1, mask=active)
    tl.atomic_or(opened_state + base, 1, mask=op)
    tl.atomic_or(accepted_state + base, 1, mask=ac)
    tl.atomic_or(opened_node + nid, 1, mask=op)
    tl.atomic_or(accepted_node + nid, 1, mask=ac)
    tl.store(next_count + fi, tl.where(op, cc, 0), mask=active)


def frontier_r6_occupancy(
    q: torch.Tensor,
    assign: torch.Tensor,
    centers: torch.Tensor,
    frontier_q: torch.Tensor,
    frontier_node: torch.Tensor,
    frontier_count: torch.Tensor,
    totals: torch.Tensor,
    *,
    logz: torch.Tensor,
    grad: torch.Tensor,
    lam: torch.Tensor,
    U: torch.Tensor,
    child_count: torch.Tensor,
    tau: float,
    evaluated_state: torch.Tensor,
    accepted_state: torch.Tensor,
    opened_state: torch.Tensor,
    accepted_node: torch.Tensor,
    opened_node: torch.Tensor,
    rank: int = 6,
) -> torch.Tensor:
    if q.ndim != 2:
        raise ValueError("q must be [Q,D]")
    qn, d = (int(v) for v in q.shape)
    ccount, n = (int(v) for v in logz.shape)
    if assign.shape != (qn,):
        raise ValueError("assign shape mismatch")
    if centers.shape != (ccount, d):
        raise ValueError("centers shape mismatch")
    if lam.shape != (ccount, n, int(rank)):
        raise ValueError("lam shape mismatch")
    if U.shape != (ccount, n, d, int(rank)):
        raise ValueError("U shape mismatch")

    _ensure_windows_triton_dll_search()
    capacity = int(frontier_q.numel())
    out = torch.zeros(
        capacity, device=q.device, dtype=torch.int32
    )
    block_d = triton.next_power_of_2(d)
    _frontier_r6_occupancy_kernel[(capacity,)](
        q.float().contiguous(),
        assign.to(device=q.device, dtype=torch.int32).contiguous(),
        centers.float().contiguous(),
        frontier_q.to(device=q.device, dtype=torch.int32).contiguous(),
        frontier_node.to(device=q.device, dtype=torch.int32).contiguous(),
        frontier_count.to(device=q.device, dtype=torch.int32).contiguous(),
        totals.float().contiguous(),
        logz.float().contiguous(),
        grad.float().contiguous(),
        lam.float().contiguous(),
        U.float().contiguous(),
        child_count.to(device=q.device, dtype=torch.int32).contiguous(),
        evaluated_state,
        accepted_state,
        opened_state,
        accepted_node,
        opened_node,
        out,
        float(tau),
        N=n,
        D=d,
        RANK=int(rank),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


__all__ = [
    "mark_root_occupancy",
    "frontier_r6_occupancy",
]
