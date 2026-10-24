"""E1 direct-descendant solver, preserving the pinned streaming arithmetic.

Only output ownership changes: two reusable FP32 iteration buffers and direct
writes to retained state slots. No benchmark/tool module is imported.
"""
from __future__ import annotations

import torch
from . import rrc3_kernels as kernels


@torch.inference_mode()
def build_state(keys, leaf, forest_gpu, workspace, *, node_count, seed_base):
    """Return seven workspace views. Valid until the next state build."""
    keys = keys.to(torch.long).contiguous()
    count = int(keys.numel())
    workspace.require("states", count, workspace.plan.state_capacity)
    if count == 0:
        raise ValueError("state build needs a nonempty canonical unique key set")
    names = ("state_logz", "state_grad", "state_value", "state_lambda",
             "state_u", "state_trace", "state_captured_trace")
    outputs = tuple(workspace.buffers[name][:count] for name in names)
    lz_out, grad_out, value_out, lam_out, u_out, trace_out, ctr_out = outputs
    order, start, length = forest_gpu
    b = int(leaf["logz"].shape[1])
    batch = workspace.buffers["solver_a"].shape[0]
    for b0 in range(0, count, batch):
        b1 = min(b0 + batch, count)
        m = b1 - b0
        kb = keys[b0:b1]
        cid = torch.div(kb, node_count, rounding_mode="floor").to(torch.int32)
        nid = kb.remainder(node_count).to(torch.int32)
        lz = lz_out[b0:b1]
        tr = trace_out[b0:b1]
        pg = workspace.buffers["parent_grad"][:m]
        pv = workspace.buffers["parent_value"][:m]
        kernels._htrc_target_reduce_range_kernel[(m,)](
            cid, nid, order, start, length,
            leaf["logz"], leaf["grad"], leaf["value"], leaf["trace"],
            lz, pg, pv, tr, B=b, D=128, BLOCK_D=128, num_warps=4,
        )
        value_out[b0:b1].copy_(pv)
        x = workspace.buffers["solver_a"][:m]
        y = workspace.buffers["solver_b"][:m]
        gen = torch.Generator(device=keys.device)
        gen.manual_seed(int(seed_base + b0))
        # Same shape, dtype, canonical order and seed as legacy torch.randn.
        # This keeps legacy batch dependence explicit; no state reordering.
        torch.randn(x.shape, generator=gen, out=x)

        def apply(src, dst):
            kernels._htrc_target_h_apply_small8_range_kernel[(m, 4)](
                src, lz, pg, cid, nid, order, start, length,
                leaf["logz"], leaf["grad"], leaf["small"], leaf["qleaf"],
                dst, B=b, D=128, K=16, K_TILE=4, BLOCK_D=128, num_warps=4,
            )

        for _ in range(2):
            apply(x, y)
            apply(y, x)
            kernels._htrc_mgs_kernel[(m,)](
                x, y, D=128, K=16, BLOCK_D=128,
                PASSES=2, EPS=1.0e-20, num_warps=4,
            )
            x, y = y, x
        apply(x, y)
        t = torch.matmul(x.transpose(1, 2), y)
        t = 0.5 * (t + t.transpose(1, 2))
        ev, vectors = torch.linalg.eigh(t)
        chosen = torch.argsort(ev.abs(), dim=1, descending=True)[:, :8]
        lam = torch.gather(ev, 1, chosen)
        idx = chosen[:, None, :].expand(-1, vectors.shape[1], -1)
        vr = torch.gather(vectors, 2, idx)
        u = torch.matmul(x, vr)
        lam_out[b0:b1].copy_(lam)
        u_out[b0:b1].copy_(u)
        grad_out[b0:b1].copy_(pg)
        ctr_out[b0:b1].copy_(lam.sum(dim=1))
    return outputs


@torch.inference_mode()
def build_leaf_operator_authority(
    htrc,
    centers,
    g,
    *,
    centroid_chunk: int = 4,
):
    """Build exact leaf state without eigendecomposing the 8x8 Hessian.

    The physical leaf Hessian is represented exactly as Q8 @ small @ Q8.T.
    """
    C = int(centers.shape[0])
    B = int(g["kappa"].shape[0])
    D = int(centers.shape[-1])
    dev = centers.device

    logz = torch.empty((C, B), device=dev, dtype=torch.float32)
    grad = torch.empty((C, B, D), device=dev, dtype=torch.float32)
    value = torch.empty((C, B, D), device=dev, dtype=torch.float32)
    small = torch.empty((C, B, 8, 8), device=dev, dtype=torch.float32)
    trace = torch.empty((C, B), device=dev, dtype=torch.float32)
    qleaf, coord8 = htrc.prepare_leaf_curvature_basis(g)

    chunk = max(1, min(C, int(centroid_chunk)))
    for c0 in range(0, C, chunk):
        c1 = min(C, c0 + chunk)
        cc = centers[c0:c1]
        summary = htrc.centroid_summary_batch(cc, g)

        a = torch.einsum("cd,bpd->cbp", cc, g["c"])
        hh = torch.einsum("cd,bpd->cbp", cc, g["r"])
        e = (1.0 + 0.5 * a).square() + 1e-8
        de = 1.0 + 0.5 * a
        alpha = de / e
        th = torch.tanh(hh)
        pi = summary["pi"]
        P = int(a.shape[-1])

        pa = pi * alpha
        pt = pi * th
        zm = torch.cat([pa, pt], dim=-1)
        A = -(zm[..., :, None] * zm[..., None, :])

        diag = torch.cat(
            [
                pi * (alpha.square() + summary["alpha_prime"]),
                pi * (th.square() + summary["sech2"]),
            ],
            dim=-1,
        )
        A.diagonal(dim1=-2, dim2=-1).add_(diag)
        cross = pi * alpha * th
        A.diagonal(offset=P, dim1=-2, dim2=-1).add_(cross)
        A.diagonal(offset=-P, dim1=-2, dim2=-1).add_(cross)

        sm = torch.einsum("bxi,cbxy,byj->cbij", coord8, A, coord8)
        logz[c0:c1].copy_(summary["logZ"])
        grad[c0:c1].copy_(summary["grad"])
        value[c0:c1].copy_(summary["V0"])
        small[c0:c1].copy_(sm)
        trace[c0:c1].copy_(sm.diagonal(dim1=-2, dim2=-1).sum(-1))
        del summary, a, hh, e, de, alpha, th, pi, pa, pt, zm, A, diag, cross, sm

    return {
        "logz": logz,
        "grad": grad,
        "value": value,
        "small": small,
        "qleaf": qleaf.contiguous(),
        "trace": trace,
    }

