"""Reusable leaf producer with one host validation boundary.

This is a separate E2 candidate. It preserves the reference arithmetic order,
reuses a/h/alpha/tanh and pair_mass, and writes into planned FP32 leaf buffers.
It does not fuse the PyTorch operations into a single CUDA kernel.
"""
from __future__ import annotations

import torch
from . import htrc_live as htrc

CENTROID_CHUNK = 4
FINITE_STAGES = (
    "pair_mass", "Z", "pi", "N", "V0",
    "grad", "small8", "trace", "logZ", "qleaf",
)


class LeafNumericalError(RuntimeError):
    def __init__(self, stage, centroid_start, centroid_end):
        self.stage = stage
        self.centroid_start = centroid_start
        self.centroid_end = centroid_end
        super().__init__(
            "HTRC leaf producer non-finite values: "
            f"stage={stage}, centroids=[{centroid_start},{centroid_end})"
        )


@torch.inference_mode()
def build_leaf_reuse(centers, g, workspace):
    """Return workspace views, valid until the next invocation.

    Flags are written on the current CUDA stream. All checks used by the
    reference value producer remain, with additional checks of emitted
    curvature/gradient fields. No flag is read back within the chunk loop.
    The solver is called only after the final validation succeeds.
    """
    c = int(centers.shape[0])
    b = int(g["kappa"].shape[0])
    if b != workspace.plan.leaf_count or centers.shape[-1] != 128:
        raise ValueError("leaf geometry does not match the D128 workspace plan")
    workspace.require("centroids", c, workspace.plan.centroid_capacity)
    names = {
        "logz": "leaf_logz", "grad": "leaf_grad", "value": "leaf_value",
        "small": "leaf_small", "trace": "leaf_trace",
    }
    leaf = {key: workspace.buffers[name][:c] for key, name in names.items()}
    leaf["qleaf"] = workspace.buffers["leaf_q8"]
    flags = workspace.buffers["leaf_nonfinite"]
    flags.zero_()
    stage_ids = {name: i for i, name in enumerate(FINITE_STAGES)}

    def mark(chunk, name, tensor):
        # Scalar CUDA -> scalar CUDA copy; no Python truth-value conversion.
        flags[chunk, stage_ids[name]].copy_(~torch.isfinite(tensor).all())

    qleaf, coord8 = htrc.prepare_leaf_curvature_basis(g)
    leaf["qleaf"].copy_(qleaf)
    mark(0, "qleaf", qleaf)
    del qleaf
    base = g["vm"][:, None, :] + g["M"]

    for c0 in range(0, c, CENTROID_CHUNK):
        c1 = min(c0 + CENTROID_CHUNK, c)
        ci = c0 // CENTROID_CHUNK
        cc = centers[c0:c1]
        a = torch.einsum("cd,bpd->cbp", cc, g["c"])
        h = torch.einsum("cd,bpd->cbp", cc, g["r"])
        t = torch.einsum("cd,bd->cb", cc, g["kappa"])
        ch = torch.cosh(h)
        sh = torch.sinh(h)
        th = torch.tanh(h)
        e = (1.0 + 0.5 * a).square() + 1e-8
        de = 1.0 + 0.5 * a
        et = 2.0 * torch.exp(t[..., None])
        pair_mass = et * e * ch
        mark(ci, "pair_mass", pair_mass)
        z = pair_mass.sum(-1).clamp_min(1e-20)
        mark(ci, "Z", z)
        pi = pair_mass / z[..., None]
        mark(ci, "pi", pi)
        alpha = de / e
        alpha_prime = (0.5 * e - de.square()) / e.square()
        sech2 = 1.0 / ch.square().clamp_min(1e-20)

        grad = (
            g["kappa"][None]
            + torch.einsum("cbp,bpd->cbd", pi * alpha, g["c"])
            + torch.einsum("cbp,bpd->cbd", pi * th, g["r"])
        )
        mark(ci, "grad", grad)
        # pair_mass is exactly the reference w_base expression, unchanged
        # by the preceding divisions/reductions (none is in-place).
        w_dv = et * e * sh
        w_m = et * de * ch
        numerator = (
            torch.einsum("cbp,bpd->cbd", pair_mass, base)
            + torch.einsum("cbp,bpd->cbd", w_dv, g["Dv"])
            + torch.einsum("cbp,bpd->cbd", w_m, g["M"])
        )
        mark(ci, "N", numerator)
        value = numerator / z[..., None]
        mark(ci, "V0", value)
        logz = torch.log(z)
        mark(ci, "logZ", logz)

        p = a.shape[-1]
        pa, pt = pi * alpha, pi * th
        zm = torch.cat([pa, pt], dim=-1)
        matrix = -(zm[..., :, None] * zm[..., None, :])
        diag = torch.cat([
            pi * (alpha.square() + alpha_prime),
            pi * (th.square() + sech2),
        ], dim=-1)
        matrix.diagonal(dim1=-2, dim2=-1).add_(diag)
        cross = pi * alpha * th
        matrix.diagonal(offset=p, dim1=-2, dim2=-1).add_(cross)
        matrix.diagonal(offset=-p, dim1=-2, dim2=-1).add_(cross)
        small = torch.einsum("bxi,cbxy,byj->cbij", coord8, matrix, coord8)
        mark(ci, "small8", small)
        trace = small.diagonal(dim1=-2, dim2=-1).sum(-1)
        mark(ci, "trace", trace)

        leaf["logz"][c0:c1].copy_(logz)
        leaf["grad"][c0:c1].copy_(grad)
        leaf["value"][c0:c1].copy_(value)
        leaf["small"][c0:c1].copy_(small)
        leaf["trace"][c0:c1].copy_(trace)
        del (a, h, t, ch, sh, th, e, de, et, pair_mass, z, pi,
             alpha, alpha_prime, sech2, grad, w_dv, w_m, numerator,
             value, logz, pa, pt, zm, matrix, diag, cross, small, trace)

    # One normal-path host observation, after all chunk writes/checks.
    # Invalid intermediates never reach a solver/eigensolve.
    if bool(flags.any().item()):
        failed = flags.nonzero(as_tuple=False).cpu().tolist()
        # qleaf is shared by every centroid; report it first if invalid.
        shared = stage_ids["qleaf"]
        ci, si = ((0, shared) if any(s == shared for _, s in failed) else failed[0])
        start, end = ((0, c) if si == shared else
                      (ci * CENTROID_CHUNK, min((ci + 1) * CENTROID_CHUNK, c)))
        status = workspace.buffers["status"]
        status[0].fill_(2)
        status[1].fill_(si + 1)
        status[2].fill_(start)
        status[3].fill_(end)
        raise LeafNumericalError(FINITE_STAGES[si], start, end)
    return leaf
