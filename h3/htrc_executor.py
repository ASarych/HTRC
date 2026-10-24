"""Opt-in single-head HTRC E1 executor.

This is the geometry + prefix-core boundary of the streaming research authority.
It does NOT implement source-tail merging or replace production Sol attention.
E1 still uses CPU topology, host counts, library eigh and transient allocations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading

import torch

from . import htrc_live as htrc
from . import rrc3_kernels as kernels
from .htrc_executor_plan import ExecutorConfig, ExecutorPlan, make_plan
from .htrc_executor_solver import build_leaf_operator_authority, build_state
from .htrc_executor_kernels import compact_roots, expand_frontier
from .htrc_executor_producer import build_leaf_reuse
from .htrc_executor_queue import build_state_queue


class ExecutorCapacityError(RuntimeError):
    def __init__(self, resource, required, available):
        self.resource, self.required, self.available = resource, required, available
        super().__init__(
            f"HTRC NEED_CAPACITY: {resource} requires {required}, available {available}"
        )


class ExecutorWorkspace:
    """Exclusive scratch; returned outputs never alias this storage."""

    def __init__(self, plan: ExecutorPlan, device="cuda"):
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("HTRC executor requires CUDA")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.plan, self.device = plan, dev
        self._lock = threading.Lock()
        self._completion = None
        self._poisoned = False
        with torch.cuda.device(dev):
            self.buffers = {
                spec.name: torch.empty(spec.shape, device=dev,
                                       dtype=getattr(torch, spec.dtype))
                for spec in plan.buffers
            }

    def acquire(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("HTRC workspace is already in use")
        if self._poisoned:
            self._lock.release()
            raise RuntimeError("HTRC workspace is unusable after a CUDA failure")
        try:
            if self._completion is not None and not self._completion.query():
                raise RuntimeError("HTRC workspace's previous CUDA work is still in flight")
        except BaseException:
            self._lock.release()
            raise
        try:
            self.buffers["status"].zero_()
        except BaseException:
            self._lock.release()
            raise

    def release(self, stream):
        try:
            event = torch.cuda.Event()
            event.record(stream)
            self._completion = event
            return event
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._lock.release()

    def require(self, resource, required, available):
        if required > available:
            status = self.buffers["status"]
            status[0].fill_(1)
            status[1].fill_({"states": 1, "frontier": 2, "nodes": 3,
                            "centroids": 4, "roots": 5}[resource])
            status[2].fill_(required)
            status[3].fill_(available)
            raise ExecutorCapacityError(resource, required, available)


@dataclass(frozen=True)
class ExecutorResult:
    output: torch.Tensor
    completion: object
    status: torch.Tensor
    metadata: dict


def allocate_workspace(plan: ExecutorPlan, device="cuda"):
    return ExecutorWorkspace(plan, device)


def _pack_queries(q, centers):
    """Same stable inputs and <=4 bucket DP as the streaming authority."""
    dev = q.device
    ccount = centers.shape[0]
    assign = htrc.assign_queries(q, centers)
    assign64 = assign.to(torch.int64)
    order = torch.argsort(assign64).contiguous()
    counts = [int(x) for x in torch.bincount(assign64, minlength=ccount).cpu().tolist()]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    dq = q.index_select(0, order).float()
    centroid = assign.index_select(0, order).to(torch.int32).contiguous()
    for cid, count in enumerate(counts):
        if count:
            dq[offsets[cid]:offsets[cid + 1]].sub_(centers[cid].float())
    active = sorted((cid for cid, count in enumerate(counts) if count),
                    key=lambda cid: counts[cid], reverse=True)
    nb, na = min(4, len(active)), len(active)
    dp = [[10**30] * (na + 1) for _ in range(nb + 1)]
    prev = [[-1] * (na + 1) for _ in range(nb + 1)]
    dp[0][0] = 0
    for b in range(1, nb + 1):
        for i in range(1, na + 1):
            for j in range(b - 1, i):
                cost = dp[b - 1][j] + (i - j) * counts[active[j]]
                if cost < dp[b][i]:
                    dp[b][i], prev[b][i] = cost, j
    b = min(range(1, nb + 1), key=lambda bi: dp[bi][na])
    groups, i = [], na
    while b:
        j = prev[b][i]
        if j < 0:
            raise RuntimeError("HTRC bucket partition reconstruction failed")
        groups.append(active[j:i])
        i, b = j, b - 1
    buckets = []
    for group in reversed(groups):
        rows = max(counts[cid] for cid in group)
        local, global_ = [], []
        for bi, cid in enumerate(group):
            local.append(torch.arange(bi*rows, bi*rows + counts[cid],
                                      device=dev, dtype=torch.long))
            global_.append(torch.arange(offsets[cid], offsets[cid + 1],
                                        device=dev, dtype=torch.long))
        buckets.append((len(group), rows,
                        torch.cat(local).contiguous(), torch.cat(global_).contiguous(),
                        torch.tensor(group, device=dev, dtype=torch.long)))
    return dq, centroid, order, buckets


def _root_route(dq, buckets, state, roots, child_count, workspace):
    """Authority root equations; large projection fusion is a later stage."""
    qn, d = dq.shape
    ccount = state[0].numel() // roots.numel()
    rn = roots.numel()
    logz, grad, value, lam, u, trace, ctr = state
    logz, lam = logz.view(ccount, rn), lam.view(ccount, rn, 8)
    grad, value = grad.view(ccount, rn, d), value.view(ccount, rn, d)
    u = u.view(ccount, rn, d, 8)
    trace, ctr = trace.view(ccount, rn), ctr.view(ccount, rn)
    score = workspace.buffers["root_score"][:, :rn]
    mass = workspace.buffers["root_mass"][:, :rn]
    log_total = workspace.buffers["log_total"]
    log_total.fill_(-torch.inf)
    norm = kernels.htrc_row_norm2(dq)
    chunk = min(workspace.plan.config.root_chunk, rn)
    for size, rows, local, global_, ids in buckets:
        qb = torch.zeros((size, rows, d), device=dq.device, dtype=torch.float32)
        qb.view(-1, d).index_copy_(0, local, dq.index_select(0, global_))
        nb = torch.zeros((size, rows), device=dq.device, dtype=torch.float32)
        nb.view(-1).index_copy_(0, local, norm.index_select(0, global_))
        total = torch.full((size, rows), -torch.inf, device=dq.device, dtype=torch.float32)
        gb, ub = grad.index_select(0, ids).float(), u.index_select(0, ids).float()
        lzb, lb = logz.index_select(0, ids).float(), lam.index_select(0, ids).float()
        tb, cb = trace.index_select(0, ids).float(), ctr.index_select(0, ids).float()
        for r0 in range(0, rn, chunk):
            r1 = min(r0 + chunk, rn)
            rb = r1 - r0
            gt = gb[:, r0:r1].transpose(1, 2).contiguous()
            linear = torch.bmm(qb, gt)
            ut = ub[:, r0:r1].permute(0, 2, 1, 3).reshape(size, d, rb*8).contiguous()
            proj = torch.bmm(qb, ut).view(size, rows, rb, 8)
            proj.square_()
            proj_norm = proj.sum(-1)
            proj.mul_(lb[:, None, r0:r1])
            q2 = proj.sum(-1)
            linear.add_(lzb[:, None, r0:r1])
            linear.add_(q2, alpha=0.5)
            torch.logaddexp(total, torch.logsumexp(linear, dim=-1), out=total)
            kernels.htrc_cert_logunc_inplace(q2)
            q2.add_(linear)
            proj_norm.neg_().add_(nb[:, :, None]).clamp_min_(0.0)
            proj_norm.mul_((tb[:, r0:r1] - cb[:, r0:r1])[:, None, :])
            proj_norm.div_(float(d - 8)).mul_(1.10).add_(linear)
            score[:, r0:r1].index_copy_(0, global_, q2.view(-1, rb).index_select(0, local))
            mass[:, r0:r1].index_copy_(0, global_, proj_norm.view(-1, rb).index_select(0, local))
            del gt, linear, ut, proj, proj_norm, q2
        log_total.index_copy_(0, global_, total.view(-1).index_select(0, local))
        del qb, nb, total, gb, ub, lzb, lb, tb, cb
    score.sub_(log_total[:, None])
    can_open = child_count.index_select(0, roots) > 0
    openable_count = qn * int(can_open.sum().item())
    if openable_count <= 0:
        # The authority errors for this degenerate topology as well.
        raise RuntimeError("HTRC root routing has no openable valid scores")
    score[:, ~can_open] = -torch.inf
    fraction = min(max(workspace.plan.config.open_fraction, 1e-6), 1.0-1e-6)
    rank = (1.0-fraction) * float(openable_count-1)
    lo, hi = math.floor(rank), math.ceil(rank)
    top = torch.topk(score.flatten(), k=openable_count-lo,
                     largest=True, sorted=True).values
    xlo, xhi = top[-1], top[-2] if hi > lo else top[-1]
    tau = float((xlo + (xhi-xlo) * float(rank-lo)).item())
    opened = score > tau
    mass.masked_fill_(opened, -torch.inf)
    root_m = mass.amax(-1)
    safe_m = torch.where(~opened.all(-1), root_m, torch.zeros_like(root_m))
    mass.sub_(safe_m[:, None]).exp_().masked_fill_(opened, 0.0)
    am, az, an = (workspace.buffers[k] for k in ("acc_m", "acc_z", "acc_n"))
    am.copy_(safe_m)
    az.copy_(mass.sum(-1))
    vf = value.float()
    for size, rows, local, global_, ids in buckets:
        mb = torch.zeros((size, rows, rn), device=dq.device, dtype=torch.float32)
        mb.view(-1, rn).index_copy_(0, local, mass.index_select(0, global_))
        vb = vf.index_select(0, ids)
        ob = torch.bmm(mb, vb)
        an.index_copy_(0, global_, ob.view(-1, d).index_select(0, local))
    return opened, tau


def _run_core(q, k, v, workspace, head_index, diagnostics):
    plan = workspace.plan
    geom = htrc.prepare_leaf_geometry(
        k.contiguous()[:, None, :], v.contiguous()[:, None, :],
        leaf_work_chunk=plan.config.leaf_work_chunk,
    )
    g = htrc.select_head_geometry(geom, 0)
    del geom
    centers = htrc.sampled_query_centers(q, count=128, clusters=32)
    forest = htrc.build_root_kmeans_implicit_forest(
        g["km"].detach().cpu(), roots=80, fanout=4, proj_dim=24,
        iters=8, seed=20260925, head_index=head_index,
    )
    c, n = centers.shape[0], int(forest["node_count"])
    workspace.require("nodes", n, plan.node_capacity)
    workspace.require("centroids", c, plan.centroid_capacity)
    dev = q.device
    forest_gpu = tuple(forest[name].to(dev, dtype=torch.int32).contiguous()
                       for name in ("leaf_order", "node_leaf_start", "node_leaf_count"))
    children = forest["child_idx"].to(dev, dtype=torch.int32).contiguous()
    child_count = forest["child_count"].to(dev, dtype=torch.int32).contiguous()
    roots = forest["root_ids"].to(dev, dtype=torch.long)
    workspace.require("roots", roots.numel(), plan.root_capacity)
    if plan.config.leaf_producer == "reuse":
        leaf = build_leaf_reuse(centers, g, workspace)
    else:
        leaf = build_leaf_operator_authority(htrc, centers, g, centroid_chunk=4)
    # Leaf outputs own the data needed below; discard unused geometry here.
    del g
    dq, centroid, order, buckets = _pack_queries(q, centers)
    root_keys = (torch.arange(c, device=dev, dtype=torch.long)[:, None]*n
                 + roots[None, :]).reshape(-1)
    state = build_state(root_keys, leaf, forest_gpu, workspace, node_count=n,
                        seed_base=20260927 + 100000*head_index)
    opened, tau = _root_route(dq, buckets, state, roots, child_count, workspace)
    # All root state consumers precede the next build on the same stream.
    root_trace = ({"opened": opened.clone(), "query_order": order.clone()}
                  if diagnostics == "trace" else None)
    del state, root_keys, buckets, centers
    fq, fn, fc = compact_roots(opened, roots, children, child_count, workspace)
    del opened, roots
    rows, unique_counts, route_trace = [], [], []
    side = "b"
    for level in range(int(forest["max_depth"])):
        count = fq.numel()
        if not count:
            break
        rows.append(count)
        row_keys = kernels.htrc_frontier_row_keys(fq, fn, centroid, node_count=n)
        keys, slots = build_state_queue(row_keys, c, n, workspace)
        unique_counts.append(keys.numel())
        state = build_state(
            keys, leaf, forest_gpu, workspace, node_count=n,
            seed_base=20260927 + 100000*head_index + 10000*(level+1),
        )
        plz, pg, pv, ll, uu, tr, ctr = state
        mass, opened, next_count = kernels.htrc_frontier_step_packed(
            dq, fq, fn, slots, fc, workspace.buffers["log_total"],
            logz=plz, grad=pg, lam=ll, U=uu, trace=tr, captured_trace=ctr,
            child_count=child_count, capacity=count, tau=tau,
        )
        if diagnostics == "trace":
            route_trace.append({
                "level": level, "query": fq.clone(), "node": fn.clone(),
                "keys": keys.clone(), "opened": opened.clone(),
            })
        kernels.htrc_frontier_level_merge_packed_(
            fq, slots, fc, mass, opened, value=pv,
            global_m=workspace.buffers["acc_m"], global_z=workspace.buffers["acc_z"],
            global_n=workspace.buffers["acc_n"],
        )
        fq, fn, fc = expand_frontier(fq, fn, fc, next_count, children, workspace, side)
        side = "a" if side == "b" else "b"
        del state, plz, pg, pv, ll, uu, tr, ctr, slots, mass, opened, next_count
        del row_keys, keys
    if fq.numel():
        raise RuntimeError("HTRC frontier remains active beyond forest depth")
    out = torch.empty_like(q, dtype=torch.float32)
    sorted_out = workspace.buffers["acc_n"] / workspace.buffers["acc_z"][:, None].clamp_min(1e-20)
    out.index_copy_(0, order, sorted_out)
    if not bool(torch.isfinite(out).all().item()):
        workspace.buffers["status"][0].fill_(2)
        raise RuntimeError("HTRC executor produced non-finite output")
    return out, {
        "format": "htrc-executor-e3-state-queue",
        "state_queue": plan.config.state_queue,
        "leaf_producer": plan.config.leaf_producer, "physical_head": head_index,
        "boundary": "htrc_geometry_and_prefix_core",
        "source_tail_count": plan.source_tail_count,
        "source_tail_merged": False,
        "tau": tau, "frontier_rows_by_level": rows,
        "frontier_unique_states_by_level": unique_counts,
        "diagnostics": diagnostics, "route_trace": route_trace, "root_trace": root_trace,
        "host_count_reads": True, "cpu_topology": True,
        "transient_allocations": True, "graph_capture_supported": False,
    }


@torch.inference_mode()
def execute_core(q, k, v, plan: ExecutorPlan, workspace: ExecutorWorkspace, *,
                 head_index=0, stream=None, diagnostics="boundary"):
    """Run geometry + prefix core for one head using [Q,128]/[K,128] tensors.

    Inputs are unscaled captured tensors, exactly as in the reference profiler.
    Non-multiple-of-16 source tails are explicitly excluded, as in that profiler.
    This opt-in research API must not be substituted for complete attention.
    The result owns its output/status; trace mode owns additional GPU snapshots.
    Exceptions publish no output. A capacity error is safe to replay from start.
    """
    if plan != workspace.plan:
        raise ValueError("workspace was allocated for a different plan")
    if diagnostics not in ("boundary", "trace"):
        raise ValueError("diagnostics must be boundary or trace")
    if type(head_index) is not int or head_index < 0:
        raise ValueError("head_index must be a nonnegative integer")
    for name, t, shape in (("q", q, (plan.query_count, 128)),
                           ("k", k, (plan.source_count, 128)),
                           ("v", v, (plan.source_count, 128))):
        if t.shape != shape or t.device != workspace.device:
            raise ValueError(f"{name} must have shape {shape} on {workspace.device}")
        if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError(f"{name} must have a floating attention dtype")
    if k.dtype != v.dtype:
        raise TypeError("k and v must have the same dtype")
    with torch.cuda.device(workspace.device):
        chosen = stream if stream is not None else torch.cuda.current_stream()
        if chosen.device != workspace.device:
            raise ValueError("stream belongs to another CUDA device")
        chosen.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(chosen):
            workspace.acquire()
            try:
                for buffer in workspace.buffers.values():
                    buffer.record_stream(chosen)
                for t in (q, k, v):
                    t.record_stream(chosen)
                kernels._ensure_windows_triton_dll_search()
                qf = q.float().contiguous()
                out, metadata = _run_core(qf, k, v, workspace, head_index, diagnostics)
                status = workspace.buffers["status"].clone()
            finally:
                completion = workspace.release(chosen)
    return ExecutorResult(out, completion, status, metadata)
