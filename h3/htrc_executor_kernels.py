"""Bounded frontier adapters for the E1 executor.

These launch the existing authority kernels. Counts are deliberately read on
the host in E1, before scatter, so overflow never publishes a truncated queue.
"""
from __future__ import annotations
import torch
import triton
from . import rrc3_kernels as kernels


def compact_roots(open_mask, roots, child_idx, child_count, workspace):
    qn, rn = open_mask.shape
    roots = roots.to(dtype=torch.int32).contiguous()
    om = open_mask.contiguous()
    counts = torch.empty((qn,), device=om.device, dtype=torch.int32)
    kernels._htrc_frontier_query_count_kernel[(qn,)](
        om, roots, child_count, counts,
        Q=qn, R=rn, BLOCK_R=triton.next_power_of_2(rn), num_warps=4,
    )
    prefix = torch.cumsum(counts, 0, dtype=torch.int64)
    base = prefix - counts.to(torch.int64)
    required = int(prefix[-1].item())
    workspace.require("frontier", required, workspace.plan.config.frontier_capacity)
    out_q = workspace.buffers["frontier_q_a"]
    out_n = workspace.buffers["frontier_n_a"]
    kernels._htrc_frontier_scatter_recompute_kernel[(qn,)](
        om, roots, child_idx, child_count, base, out_q, out_n,
        Q=qn, R=rn, FANOUT=child_idx.shape[1], CAPACITY=out_q.numel(),
        BLOCK_R=triton.next_power_of_2(rn), num_warps=4,
    )
    return out_q[:required], out_n[:required], prefix[-1:].to(torch.int32)


def expand_frontier(fq, fn, fc, next_count, child_idx, workspace, side):
    prefix = torch.cumsum(next_count, 0, dtype=torch.int64)
    base = prefix - next_count.to(torch.int64)
    required = int(prefix[-1].item())
    workspace.require("frontier", required, workspace.plan.config.frontier_capacity)
    oq = workspace.buffers["frontier_q_" + side]
    on = workspace.buffers["frontier_n_" + side]
    if required:
        kernels._htrc_frontier_expand_kernel[(next_count.numel(),)](
            fq, fn, fc, next_count, base, child_idx, oq, on,
            N=child_idx.shape[0], FANOUT=child_idx.shape[1],
            CAPACITY=oq.numel(), num_warps=1,
        )
    return oq[:required], on[:required], prefix[-1:].to(torch.int32)
