"""Live HTRC state builder for MiniMax-H3.

This module ports the accepted HTRC research mathematics into the runtime tree.
It deliberately contains no coarse-source surrogate.  The online evaluator is
kept separate so Triton traversal can consume the exact same flattened state.

Current authority:
- 16-token RRC3 leaf geometry;
- 32 query centroids;
- K-side roots=80, fanout=4;
- hierarchical signed rank-8 curvature;
- mass-weighted routing certificate;
- accepted mass = rank8 + residual-tail correction;
- node value = centroid monopole (the historical first HTRC value baseline).

The monopole value is intentionally the first runtime value law.  It does not
change routing/mass mathematics and can later be replaced by the validated
local dipole without changing the tree evaluator ABI.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
import torch

EPS = 1.0e-6
SEED_MOD = (1 << 63) - 1
_LOG = logging.getLogger("APL.H3HTRCLive")
_HTRC_MEMORY_PLAN_LOGGED = False
_NUMERIC_TELEMETRY = {
    "eigh_gpu_batch_failures": 0,
    "eigh_cpu_fp64_rescues": 0,
    "eigh_recursive_splits": 0,
    "chol_batches": 0,
    "chol_matrices": 0,
    "chol_fail_matrices": 0,
    "chol_qr_fallback_matrices": 0,
    "leaf_eigh_chunks": 0,
    "leaf_eigh_matrices": 0,
    "leaf_eigh_max_chunk": 0,
}


def numeric_telemetry_snapshot():
    return {k: int(v) for k, v in _NUMERIC_TELEMETRY.items()}


def numeric_telemetry_delta(before):
    now = numeric_telemetry_snapshot()
    return {
        k: int(now.get(k, 0) - int(before.get(k, 0)))
        for k in now
    }




def bounded_seed(seed: int) -> int:
    return int(seed % SEED_MOD)


def tree_kmeans(x: torch.Tensor, k: int, proj_dim: int, iters: int, seed: int):
    """Exact tree-construction k-means used by the research authority."""
    x = x.detach().float().cpu()
    n, d = x.shape
    k = max(1, min(int(k), int(n)))
    pd = min(int(proj_dim), int(d))
    g = torch.Generator(device="cpu")
    g.manual_seed(bounded_seed(seed))
    P = torch.randn((d, pd), generator=g) / math.sqrt(float(pd))
    y = x @ P
    order = (y * y).sum(1).argsort()
    pos = torch.linspace(0, n - 1, k).round().long()
    cc = y[order[pos]].clone()
    ones = torch.ones(n)
    for _ in range(max(1, int(iters))):
        dist = (
            (y * y).sum(1, keepdim=True)
            + (cc * cc).sum(1)[None, :]
            - 2.0 * y @ cc.T
        )
        a = dist.argmin(1)
        sums = torch.zeros_like(cc)
        cnt = torch.zeros(k)
        sums.index_add_(0, a, y)
        cnt.index_add_(0, a, ones)
        m = cnt > 0
        cc[m] = sums[m] / cnt[m, None]
        if (~m).any():
            miss = (~m).nonzero().flatten()
            repl = dist.min(1).values.topk(min(len(miss), n)).indices
            cc[miss[:len(repl)]] = y[repl]
    return a


def query_kmeans(q: torch.Tensor, k: int, iters: int = 30):
    """Deterministic farthest-first query centroids from research authority."""
    q = q.detach().float()
    n = len(q)
    k = min(int(k), int(n))
    first = int(torch.argmax((q * q).sum(-1)))
    centers = [q[first].clone()]
    chosen = {first}
    min_d2 = ((q - centers[0]) ** 2).sum(-1)
    for _ in range(1, k):
        order = torch.argsort(min_d2, descending=True)
        nxt = next(int(x) for x in order if int(x) not in chosen)
        chosen.add(nxt)
        centers.append(q[nxt].clone())
        min_d2 = torch.minimum(min_d2, ((q - q[nxt]) ** 2).sum(-1))
    centers = torch.stack(centers)
    assign = None
    for _ in range(max(1, int(iters))):
        new_assign = torch.cdist(q, centers).argmin(-1)
        new_centers = centers.clone()
        for j in range(k):
            mask = new_assign == j
            if bool(mask.any()):
                new_centers[j] = q[mask].mean(0)
        if (
            assign is not None
            and torch.equal(new_assign, assign)
            and torch.allclose(new_centers, centers, atol=1e-7, rtol=0)
        ):
            centers = new_centers
            assign = new_assign
            break
        centers = new_centers
        assign = new_assign
    return centers, assign


@dataclass
class _Node:
    id: int
    bids: torch.Tensor
    block: int | None = None
    children: list["_Node"] | None = None
    depth: int = 0
    subtree_end: int = -1

    def __post_init__(self):
        if self.children is None:
            self.children = []


def _build_tree(means, bids, fanout, proj, iters, seed, nodes):
    nid = len(nodes)
    if len(bids) == 1:
        n = _Node(nid, bids, int(bids[0]))
        nodes.append(n)
        n.subtree_end = len(nodes)
        return n
    n = _Node(nid, bids)
    nodes.append(n)
    a = tree_kmeans(means[bids], fanout, proj, iters, seed)
    groups = []
    for gi in range(int(a.max()) + 1):
        loc = (a == gi).nonzero().flatten()
        if len(loc):
            groups.append(bids[loc])
    if len(groups) <= 1 or max(len(x) for x in groups) >= len(bids):
        parts = min(int(fanout), len(bids))
        groups = [x for x in torch.tensor_split(bids, parts) if len(x)]
    for gi, cb in enumerate(groups):
        c = _build_tree(
            means,
            cb,
            fanout,
            proj,
            iters,
            bounded_seed(seed * 131 + gi + 7),
            nodes,
        )
        n.children.append(c)
        n.depth = max(n.depth, c.depth + 1)
    n.subtree_end = len(nodes)
    return n


def build_forest(
    means: torch.Tensor,
    *,
    roots: int = 80,
    fanout: int = 4,
    proj_dim: int = 24,
    iters: int = 8,
    seed: int = 20260925,
    head_index: int = 0,
):
    root_seed = int(seed) + 1000 * int(head_index)
    root_assign = tree_kmeans(means, roots, proj_dim, iters, root_seed)
    nodes = []
    root_nodes = []
    for ri in range(int(roots)):
        bids = (root_assign == ri).nonzero().flatten().cpu()
        if len(bids):
            root_nodes.append(
                _build_tree(
                    means,
                    bids,
                    fanout,
                    proj_dim,
                    iters,
                    int(seed) + 100000 * int(head_index) + ri,
                    nodes,
                )
            )
    max_depth = max((n.depth for n in nodes), default=0)
    levels = {}
    for n in nodes:
        levels.setdefault(n.depth, []).append(n.id)
    leaf_node_for_block = torch.empty(len(means), dtype=torch.long)
    for n in nodes:
        if n.block is not None:
            leaf_node_for_block[n.block] = n.id

    max_children = max((len(n.children) for n in nodes), default=0)
    child_idx = torch.full(
        (len(nodes), max(1, max_children)), -1, dtype=torch.int32
    )
    child_count = torch.zeros(len(nodes), dtype=torch.int16)
    for n in nodes:
        child_count[n.id] = len(n.children)
        for j, c in enumerate(n.children):
            child_idx[n.id, j] = c.id
    return {
        "nodes": nodes,
        "root_ids": torch.tensor([n.id for n in root_nodes], dtype=torch.int32),
        "leaf_node_for_block": leaf_node_for_block,
        "levels": levels,
        "max_depth": max_depth,
        "child_idx": child_idx,
        "child_count": child_count,
        "subtree_end": torch.tensor(
            [n.subtree_end for n in nodes], dtype=torch.int32
        ),
        "leaf_block": torch.tensor(
            [(-1 if n.block is None else int(n.block)) for n in nodes],
            dtype=torch.int32,
        ),
        "node_count": len(nodes),
    }


def build_root_kmeans_structural_forest(
    means: torch.Tensor,
    *,
    roots: int = 80,
    fanout: int = 4,
    proj_dim: int = 24,
    iters: int = 8,
    seed: int = 20260925,
    head_index: int = 0,
):
    """Keep semantic K-means roots and use deterministic positional subdivision.

    This preserves the content-dependent 80-root partition while removing the
    recursive per-node K-means hot path. Within each semantic root, blocks are
    sorted by source position and split evenly with the same fanout.
    """
    root_seed = int(seed) + 1000 * int(head_index)
    root_assign = tree_kmeans(means, roots, proj_dim, iters, root_seed)
    nodes = []
    root_nodes = []

    def rec(bids: torch.Tensor):
        nid = len(nodes)
        if len(bids) == 1:
            n = _Node(nid, bids, int(bids[0]))
            nodes.append(n)
            n.subtree_end = len(nodes)
            return n

        n = _Node(nid, bids)
        nodes.append(n)
        ordered = bids.sort().values
        parts = min(int(fanout), len(ordered))
        groups = [x for x in torch.tensor_split(ordered, parts) if len(x)]
        for cb in groups:
            child = rec(cb)
            n.children.append(child)
            n.depth = max(n.depth, child.depth + 1)
        n.subtree_end = len(nodes)
        return n

    for ri in range(int(roots)):
        bids = (root_assign == ri).nonzero().flatten().cpu()
        if len(bids):
            root_nodes.append(rec(bids))

    max_depth = max((n.depth for n in nodes), default=0)
    levels = {}
    for n in nodes:
        levels.setdefault(n.depth, []).append(n.id)
    leaf_node_for_block = torch.empty(len(means), dtype=torch.long)
    for n in nodes:
        if n.block is not None:
            leaf_node_for_block[n.block] = n.id

    max_children = max((len(n.children) for n in nodes), default=0)
    child_idx = torch.full(
        (len(nodes), max(1, max_children)), -1, dtype=torch.int32
    )
    child_count = torch.zeros(len(nodes), dtype=torch.int16)
    for n in nodes:
        child_count[n.id] = len(n.children)
        for j, child in enumerate(n.children):
            child_idx[n.id, j] = child.id
    return {
        "nodes": nodes,
        "root_ids": torch.tensor([n.id for n in root_nodes], dtype=torch.int32),
        "leaf_node_for_block": leaf_node_for_block,
        "levels": levels,
        "max_depth": max_depth,
        "child_idx": child_idx,
        "child_count": child_count,
        "subtree_end": torch.tensor(
            [n.subtree_end for n in nodes], dtype=torch.int32
        ),
        "leaf_block": torch.tensor(
            [(-1 if n.block is None else int(n.block)) for n in nodes],
            dtype=torch.int32,
        ),
    }



def _tensor_split_sizes(n: int, parts: int):
    """Return torch.tensor_split-equivalent positive segment lengths."""
    q, r = divmod(int(n), int(parts))
    return [q + 1] * r + [q] * (int(parts) - r)


def build_root_kmeans_implicit_forest(
    means: torch.Tensor,
    *,
    roots: int = 80,
    fanout: int = 4,
    proj_dim: int = 24,
    iters: int = 8,
    seed: int = 20260925,
    head_index: int = 0,
):
    """Flat equivalent of root_kmeans_structural without _Node/bids objects.

    Root membership remains the exact content-dependent K-means authority.
    Below each root, the historical structural topology is completely
    determined by source-position order plus tensor_split lengths.  Materialize
    only the compact traversal/state ABI required downstream.
    """
    root_seed = int(seed) + 1000 * int(head_index)
    root_assign = tree_kmeans(means, roots, proj_dim, iters, root_seed)

    root_ids = []
    leaf_node_for_block = [-1] * int(len(means))
    child_rows = []
    child_count = []
    subtree_end = []
    leaf_block = []
    node_depth = []
    node_leaf_start = []
    node_leaf_count = []
    leaf_order = []

    def emit(blocks, lo: int, hi: int, leaf_base: int):
        nid = len(child_rows)
        child_rows.append([])
        child_count.append(0)
        subtree_end.append(-1)
        leaf_block.append(-1)
        node_depth.append(0)
        node_leaf_start.append(int(leaf_base + lo))
        node_leaf_count.append(int(hi - lo))

        n = int(hi - lo)
        if n == 1:
            bid = int(blocks[lo])
            leaf_block[nid] = bid
            leaf_node_for_block[bid] = nid
            subtree_end[nid] = len(child_rows)
            return nid, 0

        parts = min(int(fanout), n)
        pos = int(lo)
        children = []
        max_child_depth = 0
        for size in _tensor_split_sizes(n, parts):
            cid, cdepth = emit(blocks, pos, pos + int(size), leaf_base)
            children.append(cid)
            max_child_depth = max(max_child_depth, int(cdepth))
            pos += int(size)

        child_rows[nid] = children
        child_count[nid] = len(children)
        node_depth[nid] = max_child_depth + 1
        subtree_end[nid] = len(child_rows)
        return nid, node_depth[nid]

    for ri in range(int(roots)):
        # nonzero is already in ascending source-position order, so there is
        # no need to materialize/sort one bids tensor at every descendant node.
        bids = torch.nonzero(root_assign == ri, as_tuple=False).flatten()
        if not int(bids.numel()):
            continue
        blocks = [int(x) for x in bids.tolist()]
        leaf_base = len(leaf_order)
        leaf_order.extend(blocks)
        rid, _ = emit(blocks, 0, len(blocks), leaf_base)
        root_ids.append(rid)

    max_children = max(child_count or [0])
    child_idx = torch.full(
        (len(child_rows), max(1, max_children)), -1, dtype=torch.int32
    )
    for nid, row in enumerate(child_rows):
        if row:
            child_idx[nid, :len(row)] = torch.tensor(row, dtype=torch.int32)

    return {
        "root_ids": torch.tensor(root_ids, dtype=torch.int32),
        "leaf_node_for_block": torch.tensor(
            leaf_node_for_block, dtype=torch.long
        ),
        "node_depth": torch.tensor(node_depth, dtype=torch.int16),
        "max_depth": max(node_depth or [0]),
        "child_idx": child_idx,
        "child_count": torch.tensor(child_count, dtype=torch.int16),
        "subtree_end": torch.tensor(subtree_end, dtype=torch.int32),
        "leaf_block": torch.tensor(leaf_block, dtype=torch.int32),
        "leaf_order": torch.tensor(leaf_order, dtype=torch.int32),
        "node_leaf_start": torch.tensor(node_leaf_start, dtype=torch.int32),
        "node_leaf_count": torch.tensor(node_leaf_count, dtype=torch.int32),
        "node_count": len(child_rows),
        "implicit_layout": True,
    }


def prepare_leaf_geometry(
    kh: torch.Tensor,
    vh: torch.Tensor,
    *,
    out: dict[str, torch.Tensor] | None = None,
    out_start: int = 0,
    leaf_work_chunk: int = 32,
):
    """Prepare accepted 16-token RRC3 leaf geometry with bounded FP32 work.

    When out is supplied, leaf geometry is written directly into caller-owned
    final buffers and no full-chunk FP32 geometry result is retained.
    """
    block = 16
    kred = 8
    prefix = (int(kh.shape[0]) // block) * block
    if prefix <= 0:
        raise RuntimeError("HTRC needs at least one complete 16-token block")
    hn = int(kh.shape[1])
    dim = int(kh.shape[2])
    nb = prefix // block
    work = max(1, min(nb, int(leaf_work_chunk)))

    names = ("km", "vm", "basis", "amap", "m", "d")
    if out is None:
        result = {
            "km": torch.empty((nb, hn, dim), device=kh.device, dtype=kh.dtype),
            "vm": torch.empty((nb, hn, dim), device=kh.device, dtype=kh.dtype),
            "basis": torch.empty((nb, hn, dim, kred), device=kh.device, dtype=kh.dtype),
            "amap": torch.empty((nb, hn, kred, dim), device=kh.device, dtype=kh.dtype),
            "m": torch.empty((nb, hn, kred, kred), device=kh.device, dtype=kh.dtype),
            "d": torch.empty((nb, hn, kred, kred), device=kh.device, dtype=kh.dtype),
        }
        base = 0
    else:
        result = out
        base = int(out_start)
        for name in names:
            if name not in result:
                raise ValueError(f"HTRC geometry output missing {name}")

    for leaf0 in range(0, nb, work):
        leaf1 = min(nb, leaf0 + work)
        row0 = leaf0 * block
        row1 = leaf1 * block
        cb = leaf1 - leaf0

        kb = (
            kh[row0:row1].float().reshape(cb, block, hn, dim)
            .permute(0, 2, 1, 3).contiguous()
        )
        vb = (
            vh[row0:row1].float().reshape(cb, block, hn, dim)
            .permute(0, 2, 1, 3).contiguous()
        )
        km = kb.mean(dim=2)
        vm = vb.mean(dim=2)
        x = kb - km[:, :, None, :]
        del kb
        gram = x @ x.transpose(-1, -2)
        evals, u = torch.linalg.eigh(gram)
        del gram, evals
        u = u[:, :, :, -kred:].contiguous()
        z = 4.0 * u
        basis = torch.einsum("bhtd,bhtk->bhdk", x, u).contiguous()
        del x, u
        amap = (
            torch.einsum("bhtk,bhtd->bhkd", z, vb - vm[:, :, None, :])
            / float(block)
        ).contiguous()
        del vb

        cost = ((z[:, :, :, None, :] + z[:, :, None, :, :]) ** 2).sum(-1)
        used = torch.zeros((cb, hn, block), dtype=torch.bool, device=kh.device)
        idx = torch.arange(block, device=kh.device)[None, None, :].expand(cb, hn, -1)
        bi = torch.arange(cb, device=kh.device)[:, None].expand(cb, hn)
        hi = torch.arange(hn, device=kh.device)[None, :].expand(cb, hn)
        inf = torch.tensor(float("inf"), device=kh.device)
        pair_i = []
        pair_j = []
        for _ in range(block // 2):
            ii = torch.where(~used, idx, block).amin(dim=-1)
            ci = cost[bi, hi, ii].masked_fill(used, inf)
            ci[bi, hi, ii] = inf
            jj = ci.argmin(dim=-1)
            used[bi, hi, ii] = True
            used[bi, hi, jj] = True
            pair_i.append(ii)
            pair_j.append(jj)
        pair_i = torch.stack(pair_i, dim=-1)
        pair_j = torch.stack(pair_j, dim=-1)
        zp = z[bi[:, :, None], hi[:, :, None], pair_i]
        zm = z[bi[:, :, None], hi[:, :, None], pair_j]
        m = ((zp + zm) * 0.5).contiguous()
        d = ((zp - zm) * 0.5).contiguous()

        dst0 = base + leaf0
        dst1 = base + leaf1
        result["km"][dst0:dst1].copy_(km)
        result["vm"][dst0:dst1].copy_(vm)
        result["basis"][dst0:dst1].copy_(basis)
        result["amap"][dst0:dst1].copy_(amap)
        result["m"][dst0:dst1].copy_(m)
        result["d"][dst0:dst1].copy_(d)

        del km, vm, z, basis, amap, cost, used, idx, bi, hi, inf
        del pair_i, pair_j, zp, zm, m, d

    if out is not None:
        return {"prefix": prefix, "leaves": nb}
    return {"prefix": prefix, **result}


def prepare_leaf_routing_geometry(
    kh: torch.Tensor,
    *,
    leaf_work_chunk: int = 32,
):
    """Prepare only K-side leaf geometry required by recursive routing.

    This is mathematically identical to the K-derived subset of
    prepare_leaf_geometry(), but intentionally omits every V-dependent field:
    vm, amap, M and Dv.  It is the correct broad state for split r6 routing;
    value-side state can be treated as lazy extra work instead of a fixed
    routing cost.
    """
    block = 16
    kred = 8
    prefix = (int(kh.shape[0]) // block) * block
    if prefix <= 0:
        raise RuntimeError("HTRC needs at least one complete 16-token block")
    hn = int(kh.shape[1])
    dim = int(kh.shape[2])
    nb = prefix // block
    work = max(1, min(nb, int(leaf_work_chunk)))

    result = {
        "km": torch.empty((nb, hn, dim), device=kh.device, dtype=kh.dtype),
        "basis": torch.empty(
            (nb, hn, dim, kred), device=kh.device, dtype=kh.dtype
        ),
        "m": torch.empty(
            (nb, hn, block // 2, kred), device=kh.device, dtype=kh.dtype
        ),
        "d": torch.empty(
            (nb, hn, block // 2, kred), device=kh.device, dtype=kh.dtype
        ),
    }

    for leaf0 in range(0, nb, work):
        leaf1 = min(nb, leaf0 + work)
        row0 = leaf0 * block
        row1 = leaf1 * block
        cb = leaf1 - leaf0

        kb = (
            kh[row0:row1].float().reshape(cb, block, hn, dim)
            .permute(0, 2, 1, 3).contiguous()
        )
        km = kb.mean(dim=2)
        x = kb - km[:, :, None, :]
        del kb

        gram = x @ x.transpose(-1, -2)
        evals, u = torch.linalg.eigh(gram)
        del gram, evals
        u = u[:, :, :, -kred:].contiguous()
        z = 4.0 * u
        basis = torch.einsum("bhtd,bhtk->bhdk", x, u).contiguous()
        del x, u

        cost = ((z[:, :, :, None, :] + z[:, :, None, :, :]) ** 2).sum(-1)
        used = torch.zeros(
            (cb, hn, block), dtype=torch.bool, device=kh.device
        )
        idx = torch.arange(block, device=kh.device)[None, None, :].expand(
            cb, hn, -1
        )
        bi = torch.arange(cb, device=kh.device)[:, None].expand(cb, hn)
        hi = torch.arange(hn, device=kh.device)[None, :].expand(cb, hn)
        inf = torch.tensor(float("inf"), device=kh.device)
        pair_i = []
        pair_j = []
        for _ in range(block // 2):
            ii = torch.where(~used, idx, block).amin(dim=-1)
            ci = cost[bi, hi, ii].masked_fill(used, inf)
            ci[bi, hi, ii] = inf
            jj = ci.argmin(dim=-1)
            used[bi, hi, ii] = True
            used[bi, hi, jj] = True
            pair_i.append(ii)
            pair_j.append(jj)
        pair_i = torch.stack(pair_i, dim=-1)
        pair_j = torch.stack(pair_j, dim=-1)
        zp = z[bi[:, :, None], hi[:, :, None], pair_i]
        zm = z[bi[:, :, None], hi[:, :, None], pair_j]
        m = ((zp + zm) * 0.5).contiguous()
        d = ((zp - zm) * 0.5).contiguous()

        result["km"][leaf0:leaf1].copy_(km)
        result["basis"][leaf0:leaf1].copy_(basis)
        result["m"][leaf0:leaf1].copy_(m)
        result["d"][leaf0:leaf1].copy_(d)

        del km, z, basis, cost, used, idx, bi, hi, inf
        del pair_i, pair_j, zp, zm, m, d

    return {"prefix": prefix, **result}


def select_head_routing_geometry(g, head: int):
    """Materialize the K-only per-head geometry required by r6 routing."""
    basis = g["basis"][:, head].float()
    m = g["m"][:, head].float()
    d = g["d"][:, head].float()
    km = g["km"][:, head].float()
    dim = int(km.shape[-1])
    scale = math.sqrt(16.0 * float(dim))

    basis_norm = basis.square().sum(dim=1).sqrt()
    nonzero = basis_norm > 0
    safe_norm = torch.where(nonzero, basis_norm, torch.ones_like(basis_norm))
    q8 = basis / safe_norm[:, None, :]
    coord_scale = basis_norm / scale
    c_coord8 = m * coord_scale[:, None, :]
    r_coord8 = d * coord_scale[:, None, :]
    leaf_coord8 = torch.cat([c_coord8, r_coord8], dim=1).contiguous()

    c = torch.einsum("bdk,bpk->bpd", basis, m) / scale
    r = torch.einsum("bdk,bpk->bpd", basis, d) / scale
    kappa = km / math.sqrt(float(dim))
    return {
        "km": km,
        "c": c.contiguous(),
        "r": r.contiguous(),
        "kappa": kappa.contiguous(),
        "leaf_q8": q8.contiguous(),
        "leaf_coord8": leaf_coord8,
    }

def select_head_geometry(g, head: int):
    """Materialize one head's HTRC leaf state from compact streamed geometry."""
    basis = g["basis"][:, head].float()
    amap = g["amap"][:, head].float()
    m = g["m"][:, head].float()
    d = g["d"][:, head].float()
    km = g["km"][:, head].float()
    vm = g["vm"][:, head].float()
    dim = int(km.shape[-1])
    scale = math.sqrt(16.0 * float(dim))
    # basis = X^T U where U diagonalizes X X^T.  Therefore
    # basis^T basis = U^T X X^T U is diagonal: these eight columns are already
    # mutually orthogonal by construction.  A per-leaf QR would only
    # renormalize them and is unnecessary.
    basis_norm = basis.square().sum(dim=1).sqrt()
    nonzero = basis_norm > 0
    safe_norm = torch.where(nonzero, basis_norm, torch.ones_like(basis_norm))
    q8 = basis / safe_norm[:, None, :]
    coord_scale = basis_norm / scale
    c_coord8 = m * coord_scale[:, None, :]
    r_coord8 = d * coord_scale[:, None, :]
    leaf_coord8 = torch.cat([c_coord8, r_coord8], dim=1).contiguous()

    c = torch.einsum("bdk,bpk->bpd", basis, m) / scale
    r = torch.einsum("bdk,bpk->bpd", basis, d) / scale
    kappa = km / math.sqrt(float(dim))
    M = torch.einsum("bpk,bkd->bpd", m, amap)
    Dv = torch.einsum("bpk,bkd->bpd", d, amap)
    return {
        "km": km,
        "vm": vm,
        "c": c.contiguous(),
        "r": r.contiguous(),
        "kappa": kappa.contiguous(),
        "M": M.contiguous(),
        "Dv": Dv.contiguous(),
        "leaf_q8": q8.contiguous(),
        "leaf_coord8": leaf_coord8,
    }


def centroid_summary_batch(centers: torch.Tensor, g):
    """Exact block mass derivatives + monopole value without CxBxPxD temporaries."""
    def _require_finite_value_stage(name, tensor):
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            return
        bad = int((~finite).sum().item())
        safe = torch.where(finite, tensor.float(), torch.zeros_like(tensor.float()))
        raise RuntimeError(
            "HTRC value-stage non-finite tensor: "
            f"name={name} bad={bad}/{int(tensor.numel())} "
            f"finite_abs_max={float(safe.abs().amax().item()):.6e}"
        )

    a = torch.einsum("cd,bpd->cbp", centers, g["c"])
    h = torch.einsum("cd,bpd->cbp", centers, g["r"])
    t = torch.einsum("cd,bd->cb", centers, g["kappa"])
    ch = torch.cosh(h)
    sh = torch.sinh(h)
    th = torch.tanh(h)
    e = (1.0 + 0.5 * a).square() + 1e-8
    de = 1.0 + 0.5 * a
    et = 2.0 * torch.exp(t[..., None])
    pair_mass = et * e * ch
    _require_finite_value_stage("pair_mass", pair_mass)
    Z = pair_mass.sum(-1).clamp_min(1e-20)
    _require_finite_value_stage("Z", Z)
    pi = pair_mass / Z[..., None]
    _require_finite_value_stage("pi", pi)
    alpha = de / e
    alpha_prime = (0.5 * e - de.square()) / e.square()
    sech2 = 1.0 / ch.square().clamp_min(1e-20)

    # sum_p pi_p * (kappa + alpha_p*c_p + tanh(h_p)*r_p)
    # avoids materializing pair_grad[C,B,P,D].
    grad = (
        g["kappa"][None]
        + torch.einsum("cbp,bpd->cbd", pi * alpha, g["c"])
        + torch.einsum("cbp,bpd->cbd", pi * th, g["r"])
    )

    # Algebraically identical to the old F[C,B,P,D] construction, but reduce
    # each coefficient field against its BxPxD value basis directly.
    base = g["vm"][:, None, :] + g["M"]
    w_base = et * e * ch
    w_dv = et * e * sh
    w_m = et * de * ch
    N = (
        torch.einsum("cbp,bpd->cbd", w_base, base)
        + torch.einsum("cbp,bpd->cbd", w_dv, g["Dv"])
        + torch.einsum("cbp,bpd->cbd", w_m, g["M"])
    )
    _require_finite_value_stage("N", N)
    V0 = N / Z[..., None]
    _require_finite_value_stage("V0", V0)
    return {
        "logZ": torch.log(Z),
        "Z": Z,
        "grad": grad,
        "V0": V0,
        "pi": pi,
        "alpha_prime": alpha_prime,
        "sech2": sech2,
    }


def _level_plan(forest, device):
    """Build per-depth gathers from flat integer topology only.

    Active implicit forests carry one int16 node_depth vector instead of the
    historical Python levels dict.  Legacy explicit forests still use the dict
    as a research reference.
    """
    plan = {}
    child_idx = forest["child_idx"]
    child_count = forest["child_count"]
    node_depth = forest.get("node_depth")

    for depth in range(1, forest["max_depth"] + 1):
        if node_depth is not None:
            tids_cpu = torch.nonzero(
                node_depth == int(depth), as_tuple=False
            ).flatten().to(torch.long)
        else:
            ids = forest["levels"].get(depth, [])
            if not ids:
                continue
            tids_cpu = torch.tensor(ids, dtype=torch.long)

        if int(tids_cpu.numel()) == 0:
            continue

        counts = child_count.index_select(0, tids_cpu).to(torch.long)
        F = int(counts.max().item())
        raw = child_idx.index_select(0, tids_cpu)[:, :F].to(torch.long)
        if bool((counts <= 0).any().item()):
            raise RuntimeError(
                "HTRC internal level plan contains a childless node: "
                f"depth={int(depth)}"
            )

        cols = torch.arange(F, dtype=torch.long)[None, :]
        mask_cpu = cols < counts[:, None]
        first = raw[:, :1].expand(-1, F)
        cidx_cpu = torch.where(mask_cpu, raw, first)

        plan[depth] = (
            tids_cpu.to(device),
            cidx_cpu.to(device),
            mask_cpu.to(device),
        )
    return plan

def node_base_and_value(
    summary,
    forest,
    device,
    plan=None,
    *,
    return_level_weights: bool = False,
):
    """Bottom-up baseline logZ/grad and centroid monopole value.

    Optionally retain only scalar child-mixture weights per level so the
    curvature pass can reuse them without repeating logz gather/mask/softmax.
    """
    C, B, D = summary["grad"].shape
    N = int(forest.get("node_count", int(forest["child_count"].numel())))
    lids = forest["leaf_node_for_block"].to(device)
    logz = torch.empty((C, N), device=device)
    grad = torch.empty((C, N, D), device=device)
    value = torch.empty((C, N, D), device=device)
    logz[:, lids] = summary["logZ"]
    grad[:, lids] = summary["grad"]
    value[:, lids] = summary["V0"]
    if plan is None:
        plan = _level_plan(forest, device)
    level_weights = {} if return_level_weights else None
    for depth in range(1, forest["max_depth"] + 1):
        if depth not in plan:
            continue
        tids, cidx, mask = plan[depth]
        clz = logz[:, cidx]
        clz = torch.where(mask[None], clz, torch.full_like(clz, -torch.inf))
        plz = torch.logsumexp(clz, 2)
        w = torch.softmax(clz, 2) * mask[None]
        if level_weights is not None:
            level_weights[depth] = w
        logz[:, tids] = plz
        grad[:, tids] = (w[..., None] * grad[:, cidx]).sum(2)
        value[:, tids] = (w[..., None] * value[:, cidx]).sum(2)
    if return_level_weights:
        return logz, grad, value, plan, level_weights
    return logz, grad, value, plan


def _stable_eigh(a):
    s = 0.5 * (a + a.transpose(-1, -2))
    scale = s.abs().amax(dim=(-2, -1)).clamp_min(1e-12)
    sn = torch.nan_to_num(s / scale[..., None, None])

    try:
        ev, vec = torch.linalg.eigh(sn)
    except torch._C._LinAlgError:
        _NUMERIC_TELEMETRY["eigh_gpu_batch_failures"] += 1
        # cuSOLVER can occasionally fail on a single ill-conditioned member of
        # an otherwise healthy small symmetric batch. Keep the common path on
        # GPU, recursively isolate only the failing members, and solve those in
        # CPU float64. This preserves the original symmetric eigensystem rather
        # than perturbing it with a diagonal jitter.
        n = int(sn.shape[-1])
        flat = sn.reshape(-1, n, n)

        def solve_part(x):
            try:
                return torch.linalg.eigh(x)
            except torch._C._LinAlgError:
                if int(x.shape[0]) == 1:
                    _NUMERIC_TELEMETRY["eigh_cpu_fp64_rescues"] += 1
                    _LOG.warning(
                        "HTRC eig CPU-fp64 rescue: matrix=%dx%d device=%s",
                        int(x.shape[-2]),
                        int(x.shape[-1]),
                        str(x.device),
                    )
                    ec, vc = torch.linalg.eigh(
                        x[0].double().cpu()
                    )
                    return (
                        ec.to(device=x.device, dtype=x.dtype)[None],
                        vc.to(device=x.device, dtype=x.dtype)[None],
                    )
                _NUMERIC_TELEMETRY["eigh_recursive_splits"] += 1
                mid = int(x.shape[0]) // 2
                e0, v0 = solve_part(x[:mid])
                e1, v1 = solve_part(x[mid:])
                return torch.cat([e0, e1], 0), torch.cat([v0, v1], 0)

        ev, vec = solve_part(flat)
        ev = ev.reshape(*sn.shape[:-2], n)
        vec = vec.reshape(*sn.shape[:-2], n, n)

    return ev * scale[..., None], vec


def _stable_eigh_chunked(a, max_matrices: int | None):
    """Apply the identical symmetric eigensolver in bounded independent batches."""
    n = int(a.shape[-1])
    flat = a.reshape(-1, n, n)
    total = int(flat.shape[0])
    chunk = (
        total
        if max_matrices is None
        else max(1, min(int(max_matrices), total))
    )
    if chunk >= total:
        ev, vec = _stable_eigh(flat)
    else:
        ev = torch.empty(
            (total, n), device=flat.device, dtype=flat.dtype
        )
        vec = torch.empty(
            (total, n, n), device=flat.device, dtype=flat.dtype
        )
        for start in range(0, total, chunk):
            stop = min(total, start + chunk)
            evp, vecp = _stable_eigh(flat[start:stop])
            ev[start:stop].copy_(evp)
            vec[start:stop].copy_(vecp)
            del evp, vecp
    return (
        ev.reshape(*a.shape[:-2], n),
        vec.reshape(*a.shape[:-2], n, n),
    )


def prepare_leaf_curvature_basis(g):
    """Return the exact centroid-invariant <=8D physical leaf basis.

    By construction c_p = B m_p / s and r_p = B d_p / s with B having eight
    columns.  select_head_geometry already factors B/s = Q8 R8 and stores the
    16 c/r vectors as coordinates in Q8.  No rank-deficient 16-column QR is
    needed here.
    """
    return g["leaf_q8"], g["leaf_coord8"]


def leaf_curvature(
    centers,
    g,
    summary,
    leaf_qr=None,
    *,
    eigh_chunk_matrices: int | None = None,
):
    """Exact leaf Hessian eigensystems in the common <=8D physical span."""
    if leaf_qr is None:
        qleaf, coord8 = prepare_leaf_curvature_basis(g)
    else:
        qleaf, coord8 = leaf_qr
    a = torch.einsum("cd,bpd->cbp", centers, g["c"])
    h = torch.einsum("cd,bpd->cbp", centers, g["r"])
    e = (1.0 + 0.5 * a).square() + 1e-8
    de = 1.0 + 0.5 * a
    alpha = de / e
    th = torch.tanh(h)
    C, B, P = a.shape
    pi = summary["pi"]

    # Each support vector z_p has only two nonzero coordinates:
    #   z_p[p] = alpha_p
    #   z_p[P+p] = tanh(h_p)
    # Therefore E[z z^T] is eight independent 2x2 contributions.  Build the
    # exact covariance directly instead of materializing z[C,B,P,2P] and a
    # generic outer-product einsum.
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
    # diagonal(offset=+P/-P) returns writable views for the paired
    # (p,P+p)/(P+p,p) entries.  Avoid advanced-index in-place updates here:
    # advanced indexing may operate on a copy rather than mutate A.
    A.diagonal(offset=P, dim1=-2, dim2=-1).add_(cross)
    A.diagonal(offset=-P, dim1=-2, dim2=-1).add_(cross)
    # If X=[c_1..c_P,r_1..r_P]=Q8*T^T, then
    #   H = X A X^T = Q8 (T^T A T) Q8^T.
    # coord8 stores T row-wise as [B,16,8], so the exact spectral problem is
    # only 8x8.  No assumption about centroid-to-centroid eigenvector alignment
    # is involved; eigenvectors may rotate freely inside this common space.
    small = torch.einsum(
        "bxi,cbxy,byj->cbij", coord8, A, coord8
    )
    sf = small.reshape(C * B, small.shape[-2], small.shape[-1])
    total_mats = int(sf.shape[0])
    chunk_mats = (
        total_mats
        if eigh_chunk_matrices is None
        else max(1, min(int(eigh_chunk_matrices), total_mats))
    )
    evals = torch.empty(
        (total_mats, sf.shape[-1]), device=sf.device, dtype=sf.dtype
    )
    evecs = torch.empty(
        (total_mats, sf.shape[-1], sf.shape[-1]),
        device=sf.device,
        dtype=sf.dtype,
    )
    for m0 in range(0, total_mats, chunk_mats):
        m1 = min(total_mats, m0 + chunk_mats)
        _NUMERIC_TELEMETRY["leaf_eigh_chunks"] += 1
        _NUMERIC_TELEMETRY["leaf_eigh_matrices"] += int(m1 - m0)
        _NUMERIC_TELEMETRY["leaf_eigh_max_chunk"] = max(
            int(_NUMERIC_TELEMETRY["leaf_eigh_max_chunk"]),
            int(m1 - m0),
        )
        evp, vep = _stable_eigh(sf[m0:m1])
        evals[m0:m1].copy_(evp)
        evecs[m0:m1].copy_(vep)
        del evp, vep
    # The exact physical leaf space is already at most rank 8, so all
    # eigenpairs are retained.  There is no leaf-level truncation here.
    leaf_trace = evals.sum(-1)
    modes = (
        qleaf[None]
        .expand(C, -1, -1, -1)
        .reshape(C * B, qleaf.shape[1], qleaf.shape[2])
        @ evecs
    )
    retained_rank = int(evals.shape[-1])
    return (
        evals.reshape(C, B, retained_rank),
        modes.reshape(C, B, qleaf.shape[1], retained_rank),
        leaf_trace.reshape(C, B),
        small.reshape(C, B, retained_rank, retained_rank),
    )


def compress_rank8_and_trace(
    logz,
    grad,
    leaf_evals,
    leaf_modes,
    leaf_trace,
    leaf_coeff8,
    forest,
    plan,
    *,
    level_weights=None,
    depth1_structural=None,
    factor_mode: str = "qr",
    hierarchical_eigh_chunk_matrices: int | None = 4096,
    profile: dict | None = None,
):
    """Accepted hierarchical signed-rank8 compression + exact trace.

    When shared_depth1_basis is provided, depth-1 reuses one centroid-shared
    physical QR basis per node and keeps centroid-specific signed eigensystems.
    Deeper levels use either the original QR path or the numerically gated
    Gram+Cholesky factorization. Gram+Cholesky reconstructs the same physical
    subspace from Y^T Y and falls back per failing matrix to QR.
    """
    device = logz.device
    C, N, D = grad.shape
    rank = 8
    lam = torch.zeros((C, N, rank), device=device)
    U = torch.zeros((C, N, D, rank), device=device)
    trace = torch.empty((C, N), device=device)
    lids = forest["leaf_node_for_block"].to(device)
    lam[:, lids] = leaf_evals
    U[:, lids] = leaf_modes
    trace[:, lids] = leaf_trace

    for depth in range(1, forest["max_depth"] + 1):
        if depth not in plan:
            continue
        if profile is not None:
            _depth_t0 = time.perf_counter()
        tids, cidx, mask = plan[depth]
        M = len(tids)
        F = cidx.shape[1]
        if level_weights is None:
            clz = logz[:, cidx]
            clz = torch.where(
                mask[None], clz, torch.full_like(clz, -torch.inf)
            )
            w = torch.softmax(clz, 2) * mask[None]
        else:
            w = level_weights[depth]
        cg = grad[:, cidx]
        dev = cg - grad[:, tids, None, :]

        if depth == 1 and depth1_structural is not None:
            evr_all = torch.empty(
                (C, M, rank), device=device, dtype=torch.float32
            )
            Ur_all = torch.empty(
                (C, M, D, rank), device=device, dtype=torch.float32
            )

            for grp in depth1_structural:
                rows = grp["rows"]
                Fg = int(grp["child_count"])
                Mg = int(rows.numel())
                Kg = int(grp["Q"].shape[-1])
                wg = w.index_select(1, rows)[:, :, :Fg]
                devg = dev.index_select(1, rows)[:, :, :Fg, :]
                child_A = leaf_coeff8[:, grp["child_blocks"]]

                P = grp["leaf_proj"]
                small_leaf = torch.einsum(
                    "mfki,cmfij,mflj,cmf->cmkl",
                    P,
                    child_A,
                    P,
                    wg,
                )
                qdev = torch.einsum(
                    "mdk,cmfd->cmfk", grp["Q"], devg
                )
                small_dev = torch.einsum(
                    "cmf,cmfk,cmfl->cmkl", wg, qdev, qdev
                )
                small = small_leaf + small_dev
                sf = small.reshape(C * Mg, Kg, Kg)

                hier_chunk = (
                    Mg
                    if hierarchical_eigh_chunk_matrices is None
                    else min(
                        int(hierarchical_eigh_chunk_matrices), Mg
                    )
                )
                ev, V = _stable_eigh_chunked(sf, hier_chunk)
                _mag, order = torch.topk(
                    ev.abs(),
                    k=rank,
                    dim=-1,
                    largest=True,
                    sorted=True,
                )
                del _mag
                evr = torch.gather(ev, 1, order)
                Vr = torch.gather(
                    V,
                    2,
                    order[:, None, :].expand(-1, V.shape[1], -1),
                )

                Qexp = (
                    grp["Q"][None]
                    .expand(C, Mg, D, Kg)
                    .reshape(C * Mg, D, Kg)
                )
                Ur = Qexp @ Vr

                evr_all[:, rows] = evr.reshape(C, Mg, rank)
                Ur_all[:, rows] = Ur.reshape(C, Mg, D, rank)

            evr = evr_all.reshape(C * M, rank)
            Ur = Ur_all.reshape(C * M, D, rank)
        else:
            childU = U[:, cidx]
            childL = lam[:, cidx]
            Ym = childU.permute(0, 1, 3, 2, 4).reshape(
                C, M, D, F * rank
            )
            Ya = (w[..., None] * childL).reshape(C, M, F * rank)
            Yd = dev.permute(0, 1, 3, 2)
            Y = torch.cat([Ym, Yd], dim=-1)
            coeff = torch.cat([Ya, w], dim=-1)
            Yf = Y.reshape(C * M, D, Y.shape[-1])
            cf = coeff.reshape(C * M, coeff.shape[-1])
            fmode = str(factor_mode).strip().lower()
            if fmode == "qr":
                Qm, Rm = torch.linalg.qr(Yf, mode="reduced")
            elif fmode == "gram_cholesky":
                _NUMERIC_TELEMETRY["chol_batches"] += 1
                _NUMERIC_TELEMETRY["chol_matrices"] += int(Yf.shape[0])
                gram = Yf.transpose(-1, -2) @ Yf
                gram = 0.5 * (gram + gram.transpose(-1, -2))
                scale = (
                    gram.diagonal(dim1=-2, dim2=-1)
                    .abs()
                    .amax(-1)
                )
                jitter = (scale * 1.0e-6).clamp_min(1.0e-10)
                eye = torch.eye(
                    gram.shape[-1],
                    device=device,
                    dtype=gram.dtype,
                )[None]
                L, info = torch.linalg.cholesky_ex(
                    gram + jitter[:, None, None] * eye
                )
                Rm = L.transpose(-1, -2)

                # Do not reconstruct the full Q = Y R^-1.  The only consumer
                # is the final rank-8 product Q @ Vr, which is exactly
                # Y @ (R^-1 @ Vr).  Deferring the triangular solve until Vr is
                # known reduces it from D right-hand sides to rank=8 and avoids
                # materializing the full [batch,D,K] Q tensor.
                Qm = None

                bad = info != 0
                if bool(bad.any()):
                    bad_idx = bad.nonzero(as_tuple=False).flatten()
                    nbad = int(bad_idx.numel())
                    _NUMERIC_TELEMETRY["chol_fail_matrices"] += nbad
                    _NUMERIC_TELEMETRY[
                        "chol_qr_fallback_matrices"
                    ] += nbad
                    _LOG.warning(
                        "HTRC Gram+Cholesky fallback to QR: "
                        "depth=%d failed=%d/%d",
                        int(depth),
                        nbad,
                        int(Yf.shape[0]),
                    )
                    Ybad = Yf.index_select(0, bad_idx)
                    _Qbad, Rbad = torch.linalg.qr(
                        Ybad, mode="reduced"
                    )
                    Rm.index_copy_(0, bad_idx, Rbad)
                    del _Qbad, Rbad, Ybad
            else:
                raise ValueError(
                    "HTRC factor_mode must be 'qr' or "
                    "'gram_cholesky'"
                )

            # Exact diagonal action without a dense diag_embed allocation.
            small = (
                Rm * cf.unsqueeze(-2)
            ) @ Rm.transpose(-1, -2)
            hier_chunk = (
                M
                if hierarchical_eigh_chunk_matrices is None
                else min(int(hierarchical_eigh_chunk_matrices), M)
            )
            ev, V = _stable_eigh_chunked(small, hier_chunk)
            _mag, order = torch.topk(
                ev.abs(), k=rank, dim=-1, largest=True, sorted=True
            )
            del _mag
            evr = torch.gather(ev, 1, order)
            Vr = torch.gather(
                V, 2, order[:, None, :].expand(-1, V.shape[1], -1)
            )
            if fmode == "gram_cholesky":
                # Q @ Vr == Y @ (R^-1 @ Vr), with the same R used to build
                # the small eigensystem.  This preserves the accepted
                # eigenspace while avoiding full-Q reconstruction.
                RV = torch.linalg.solve_triangular(
                    Rm, Vr, upper=True
                )
                Ur = Yf @ RV
                del RV
            else:
                Ur = Qm @ Vr
        lam[:, tids] = evr.reshape(C, M, rank)
        U[:, tids] = Ur.reshape(C, M, D, rank)

        dev2 = dev.square().sum(-1)
        trace[:, tids] = (w * (trace[:, cidx] + dev2)).sum(2)
        if profile is not None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            key = f"hierarchy_depth_{int(depth)}"
            profile[key] = float(
                profile.get(key, 0.0) + time.perf_counter() - _depth_t0
            )
    return lam, U, trace


def prepare_depth1_structural_basis(g, forest, plan):
    """Prepare a stable K-side depth-1 basis by small-block orthogonalization.

    Each leaf Q8 block is already orthonormal.  Instead of QR(S) for the full
    D x K structural matrix or normal equations S^T S, extend the shared basis
    incrementally:
      1. start from child 0 Q8;
      2. project each subsequent Q8 against the accumulated basis twice;
      3. QR only the residual D x 8 block;
      4. project each kappa_i-kappa_0 direction twice and normalize it.

    For the observed full structural ranks this spans exactly the same
    K=8F+(F-1) space while avoiding both full D x K QR and conditioning loss
    from normal equations.
    """
    device = g["kappa"].device
    _tids, cidx, mask = plan[1]
    leaf_block = forest["leaf_block"].to(device)
    child_count = mask.sum(-1).to(torch.long)
    groups = []

    for f in sorted(int(x) for x in torch.unique(child_count).tolist()):
        rows = torch.nonzero(
            child_count == int(f), as_tuple=False
        ).flatten()
        if not int(rows.numel()):
            continue
        cidx_g = cidx.index_select(0, rows)[:, :f]
        child_blocks = leaf_block[cidx_g]
        if bool((child_blocks < 0).any().item()):
            raise RuntimeError(
                "HTRC structural depth1 basis contains a non-leaf child"
            )

        child_q8 = g["leaf_q8"][child_blocks]  # [M,F,D,8]
        child_kappa = g["kappa"][child_blocks]  # [M,F,D]
        M, F, D, RANK = child_q8.shape

        # leaf_q8 is analytically orthogonal, but its stored FP32 columns are
        # not accurate enough to serve as the root of a larger coefficient
        # basis without correction.  Orthonormalize the first D x 8 block via
        # its own 8 x 8 Gram/Cholesky, avoiding a D x 8 QR.
        q0 = child_q8[:, 0]
        gram0 = q0.transpose(-1, -2) @ q0
        gram0 = 0.5 * (gram0 + gram0.transpose(-1, -2))
        L0, info0 = torch.linalg.cholesky_ex(gram0)
        if bool((info0 != 0).any().item()):
            raise RuntimeError(
                "HTRC depth1 first-Q8 8x8 Cholesky lost structural rank"
            )
        R0 = L0.transpose(-1, -2)
        eye8 = torch.eye(
            RANK, device=device, dtype=R0.dtype
        )[None].expand(M, -1, -1)
        R0inv = torch.linalg.solve_triangular(
            R0, eye8, upper=True
        )
        Q = (q0 @ R0inv).contiguous()

        for fi in range(1, F):
            z = child_q8[:, fi]
            z = z - Q @ (Q.transpose(-1, -2) @ z)
            z = z - Q @ (Q.transpose(-1, -2) @ z)

            # Stable small residual factorization:
            #   z = qz R,  R^T R = z^T z.
            # Normal equations are confined to 8x8 residual blocks rather
            # than the full 17/26/35D structural span.
            gram8 = z.transpose(-1, -2) @ z
            gram8 = 0.5 * (gram8 + gram8.transpose(-1, -2))
            L8, info8 = torch.linalg.cholesky_ex(gram8)
            if bool((info8 != 0).any().item()):
                raise RuntimeError(
                    "HTRC depth1 residual 8x8 Cholesky lost structural rank"
                )
            R8 = L8.transpose(-1, -2)
            R8inv = torch.linalg.solve_triangular(
                R8, eye8, upper=True
            )
            qz = z @ R8inv

            # One cheap post-projection repairs FP32 overlap without another QR.
            qz = qz - Q @ (Q.transpose(-1, -2) @ qz)
            gramq = qz.transpose(-1, -2) @ qz
            gramq = 0.5 * (gramq + gramq.transpose(-1, -2))
            Lq, infoq = torch.linalg.cholesky_ex(gramq)
            if bool((infoq != 0).any().item()):
                raise RuntimeError(
                    "HTRC depth1 residual reorthogonalization lost rank"
                )
            Rq = Lq.transpose(-1, -2)
            Rqinv = torch.linalg.solve_triangular(
                Rq, eye8, upper=True
            )
            qz = qz @ Rqinv

            Q = torch.cat([Q, qz], dim=-1)

        for fi in range(1, F):
            z = child_kappa[:, fi] - child_kappa[:, 0]
            z = z - torch.einsum(
                "mdk,mk->md", Q, torch.einsum("mdk,md->mk", Q, z)
            )
            z = z - torch.einsum(
                "mdk,mk->md", Q, torch.einsum("mdk,md->mk", Q, z)
            )
            zn = z.square().sum(-1).sqrt()
            if bool((zn == 0).any().item()):
                raise RuntimeError(
                    "HTRC depth1 kappa residual lost structural rank"
                )
            z = z / zn[:, None]

            # Normalization can expose residual overlap at FP32.  One more
            # two-pass projection + renormalization keeps the accumulated Q
            # genuinely orthonormal instead of merely spanning the right space.
            z = z - torch.einsum(
                "mdk,mk->md", Q, torch.einsum("mdk,md->mk", Q, z)
            )
            z = z - torch.einsum(
                "mdk,mk->md", Q, torch.einsum("mdk,md->mk", Q, z)
            )
            zn2 = z.square().sum(-1).sqrt()
            if bool((zn2 == 0).any().item()):
                raise RuntimeError(
                    "HTRC depth1 kappa reorthogonalization lost rank"
                )
            Q = torch.cat(
                [Q, (z / zn2[:, None])[:, :, None]], dim=-1
            )

        leaf_proj = torch.einsum(
            "mdk,mfdj->mfkj", Q, child_q8
        ).contiguous()

        groups.append(
            {
                "child_count": int(F),
                "rows": rows,
                "child_blocks": child_blocks,
                "Q": Q.contiguous(),
                "leaf_proj": leaf_proj,
            }
        )

    return groups


def prepare_shared_depth1_basis_from_centroid_state(
    logz,
    grad,
    leaf_evals,
    leaf_modes,
    forest,
    plan,
):
    """Historical centroid-derived depth-1 basis reference path.

    The historical implementation recomputed centroid 0 solely to obtain this
    basis and then recomputed the same centroid again in the main state loop.
    Accept already-computed centroid-0 tensors instead so the accepted
    mathematics is unchanged while one full centroid state pass per head is
    eliminated.
    """
    device = grad.device
    C, N, D = grad.shape
    if C != 1:
        raise ValueError(
            "HTRC shared depth1 basis requires exactly one centroid state"
        )
    rank = 8
    tids, cidx, mask = plan[1]
    M = len(tids)
    F = cidx.shape[1]
    clz = logz[:, cidx]
    clz = torch.where(mask[None], clz, torch.full_like(clz, -torch.inf))
    w = torch.softmax(clz, 2) * mask[None]
    dev = grad[:, cidx] - grad[:, tids, None, :]

    # Depth-1 children are leaves.  Gather their retained eigensystems
    # directly from leaf state instead of allocating/scattering full
    # [1,N,D,rank] and [1,N,rank] temporary tensors solely for this QR.
    leaf_block = forest["leaf_block"].to(device)
    child_blocks = leaf_block[cidx]
    if bool((child_blocks < 0).any().item()):
        raise RuntimeError(
            "HTRC depth1 shared-basis plan contains a non-leaf child"
        )
    childU = leaf_modes[:, child_blocks]
    childL = leaf_evals[:, child_blocks]
    Ym = childU.permute(0, 1, 3, 2, 4).reshape(C, M, D, F * rank)
    Ya = (w[..., None] * childL).reshape(C, M, F * rank)
    Yd = dev.permute(0, 1, 3, 2)
    Y = torch.cat([Ym, Yd], dim=-1)
    Qshared, _ = torch.linalg.qr(Y[0], mode="reduced")
    return Qshared.contiguous()


def sampled_query_centers(q: torch.Tensor, count: int = 128, clusters: int = 32):
    """Match the research 64-train/64-eval scale from a full live Q sequence."""
    n = int(q.shape[0])
    take = min(int(count), n)
    if take <= 1:
        sample = q
    else:
        idx = torch.linspace(0, n - 1, take, device=q.device).round().long()
        sample = q.index_select(0, idx)
    train = sample[::2].contiguous()
    return query_kmeans(train, min(int(clusters), len(train)))[0]


def build_head_state(
    q: torch.Tensor,
    g,
    *,
    roots: int = 80,
    fanout: int = 4,
    query_clusters: int = 32,
    query_sample: int = 128,
    proj_dim: int = 24,
    kmeans_iters: int = 8,
    seed: int = 20260925,
    head_index: int = 0,
    centroid_chunk: int = 1,
    forest_mode: str = "kmeans",
    factor_mode: str = "qr",
    leaf_eigh_chunk_matrices: int | None = 4096,
    vector_dtype=torch.bfloat16,
    profile: dict | None = None,
):
    """Build HTRC rank8 state with bounded centroid working memory.

    Tree topology is shared across centroids. Each centroid chunk performs the
    exact FP32 summary/curvature/hierarchical compression independently, then
    writes the large grad/value/U state into compact vector_dtype storage.
    This removes the historical CxBxPxD and CxBxDxR peak allocations.
    """
    device = q.device
    def _profile_add(name, seconds):
        if profile is not None:
            profile[name] = float(profile.get(name, 0.0) + float(seconds))

    _t = None
    centers = sampled_query_centers(
        q, count=query_sample, clusters=query_clusters
    )
    if profile is not None:
        _t = time.perf_counter()
    mode = str(forest_mode).strip().lower()
    if mode == "kmeans":
        forest = build_forest(
            g["km"].detach().cpu(),
            roots=roots,
            fanout=fanout,
            proj_dim=proj_dim,
            iters=kmeans_iters,
            seed=seed,
            head_index=head_index,
        )
    elif mode == "root_kmeans_structural":
        # Active structural mode: exact root K-means semantics with the
        # descendant topology emitted directly as a flat implicit layout.
        forest = build_root_kmeans_implicit_forest(
            g["km"].detach().cpu(),
            roots=roots,
            fanout=fanout,
            proj_dim=proj_dim,
            iters=kmeans_iters,
            seed=seed,
            head_index=head_index,
        )
    elif mode == "root_kmeans_structural_explicit":
        # Legacy reference retained only for research equivalence gates.
        forest = build_root_kmeans_structural_forest(
            g["km"].detach().cpu(),
            roots=roots,
            fanout=fanout,
            proj_dim=proj_dim,
            iters=kmeans_iters,
            seed=seed,
            head_index=head_index,
        )
    elif mode == "root_kmeans_implicit":
        # Compatibility alias for experiments created before implicit layout
        # became the active structural mode.
        forest = build_root_kmeans_implicit_forest(
            g["km"].detach().cpu(),
            roots=roots,
            fanout=fanout,
            proj_dim=proj_dim,
            iters=kmeans_iters,
            seed=seed,
            head_index=head_index,
        )
    else:
        raise ValueError(
            "HTRC forest_mode must be 'kmeans', 'root_kmeans_structural', "
            "'root_kmeans_structural_explicit', or 'root_kmeans_implicit'"
        )
    if profile is not None:
        _profile_add("forest", time.perf_counter() - _t)

    # km is needed only to build the per-head forest. Drop its FP32
    # materialization before allocating centroid/node state.
    g.pop("km", None)
    C = int(centers.shape[0])
    N = int(forest.get("node_count", int(forest["child_count"].numel())))
    D = int(q.shape[-1])
    # Final state retains all query centroids. Distinct centroid algebra
    # is independent, so batch the configured number together to amortize
    # summary/eigh/compression launch overhead while preserving identical
    # per-centroid mathematics. MemoryBudget accounts this exact work chunk.
    work_chunk = max(1, min(C, int(centroid_chunk)))

    logz_all = torch.empty((C, N), device=device, dtype=torch.float32)
    grad_all = torch.empty((C, N, D), device=device, dtype=vector_dtype)
    value_all = torch.empty((C, N, D), device=device, dtype=vector_dtype)
    lam_all = torch.empty((C, N, 8), device=device, dtype=torch.float32)
    U_all = torch.empty((C, N, D, 8), device=device, dtype=vector_dtype)
    trace_all = torch.empty((C, N), device=device, dtype=torch.float32)

    # These are invariant across query centroids; compute once per head.
    if profile is not None:
        _t = time.perf_counter()
    plan = _level_plan(forest, device)
    leaf_qr = prepare_leaf_curvature_basis(g)
    if profile is not None:
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        _profile_add("plan_leaf_qr", time.perf_counter() - _t)

    if profile is not None:
        _t = time.perf_counter()
    depth1_structural = prepare_depth1_structural_basis(
        g, forest, plan
    )
    if profile is not None:
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        _profile_add("shared_depth1_basis", time.perf_counter() - _t)

    def build_centroid_slice(c0, c1):
        cc = centers[c0:c1]
        if profile is not None:
            _t0 = time.perf_counter()
        summary = centroid_summary_batch(cc, g)
        logz, grad, value, _plan, level_weights = node_base_and_value(
            summary,
            forest,
            device,
            plan=plan,
            return_level_weights=True,
        )
        if profile is not None:
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            _profile_add("summary_treebase", time.perf_counter() - _t0)
        value_finite = torch.isfinite(value)
        if not bool(value_finite.all().item()):
            bad = int((~value_finite).sum().item())
            safe = torch.where(
                value_finite, value.float(), torch.zeros_like(value.float())
            )
            raise RuntimeError(
                "HTRC value-stage non-finite tensor: "
                f"name=tree_value bad={bad}/{int(value.numel())} "
                f"finite_abs_max={float(safe.abs().amax().item()):.6e}"
            )
        if profile is not None:
            _t0 = time.perf_counter()
        le, lm, lt, lc = leaf_curvature(
            cc,
            g,
            summary,
            leaf_qr=leaf_qr,
            eigh_chunk_matrices=leaf_eigh_chunk_matrices,
        )
        if profile is not None:
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            _profile_add("leaf_curvature", time.perf_counter() - _t0)
        return (
            summary,
            logz,
            grad,
            value,
            _plan,
            level_weights,
            le,
            lm,
            lt,
            lc,
        )

    # Build the first configured centroid batch once.  The depth-1 physical
    # basis is already fixed from K-side geometry above and is independent of
    # query centroids.
    first_c1 = min(C, work_chunk)
    (
        summary,
        logz,
        grad,
        value,
        _plan,
        level_weights,
        le,
        lm,
        lt,
        lc,
    ) = build_centroid_slice(0, first_c1)
    if profile is not None:
        _t = time.perf_counter()
    lam, U, trace = compress_rank8_and_trace(
        logz,
        grad,
        le,
        lm,
        lt,
        lc,
        forest,
        plan,
        level_weights=level_weights,
        depth1_structural=depth1_structural,
        factor_mode=factor_mode,
        hierarchical_eigh_chunk_matrices=leaf_eigh_chunk_matrices,
        profile=profile,
    )
    if profile is not None:
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        _profile_add("hierarchy", time.perf_counter() - _t)
    logz_all[0:first_c1].copy_(logz)
    grad_all[0:first_c1].copy_(grad)
    value_all[0:first_c1].copy_(value)
    if not bool(torch.isfinite(value_all[0:first_c1]).all().item()):
        raise RuntimeError(
            "HTRC value-stage non-finite tensor: name=value_all_after_copy"
        )
    lam_all[0:first_c1].copy_(lam)
    U_all[0:first_c1].copy_(U)
    trace_all[0:first_c1].copy_(trace)
    del summary, logz, grad, value, _plan, level_weights, le, lm, lt, lc, lam, U, trace

    # Remaining centroids retain the configured batching policy.
    for c0 in range(first_c1, C, work_chunk):
        c1 = min(C, c0 + work_chunk)
        (
            summary,
            logz,
            grad,
            value,
            _plan,
            level_weights,
            le,
            lm,
            lt,
            lc,
        ) = build_centroid_slice(c0, c1)
        if profile is not None:
            _t = time.perf_counter()
        lam, U, trace = compress_rank8_and_trace(
            logz,
            grad,
            le,
            lm,
            lt,
            lc,
            forest,
            plan,
            level_weights=level_weights,
            depth1_structural=depth1_structural,
            factor_mode=factor_mode,
            hierarchical_eigh_chunk_matrices=leaf_eigh_chunk_matrices,
            profile=profile,
        )
        if profile is not None:
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            _profile_add("hierarchy", time.perf_counter() - _t)

        logz_all[c0:c1].copy_(logz)
        grad_all[c0:c1].copy_(grad)
        value_all[c0:c1].copy_(value)
        if not bool(torch.isfinite(value_all[c0:c1]).all().item()):
            raise RuntimeError(
                "HTRC value-stage non-finite tensor: name=value_all_after_copy"
            )
        lam_all[c0:c1].copy_(lam)
        U_all[c0:c1].copy_(U)
        trace_all[c0:c1].copy_(trace)

        del summary, logz, grad, value, _plan, level_weights, le, lm, lt, lc, lam, U, trace

    captured_trace = lam_all.sum(-1)
    return {
        "centers": centers,
        "forest": forest,
        "logz": logz_all,
        "grad": grad_all,
        "value": value_all,
        "lam": lam_all,
        "U": U_all,
        "trace": trace_all,
        "captured_trace": captured_trace,
    }


def sampled_query_train(q: torch.Tensor, count: int = 128):
    n = int(q.shape[0])
    take = min(int(count), n)
    if take <= 1:
        sample = q
    else:
        idx = torch.linspace(0, n - 1, take, device=q.device).round().long()
        sample = q.index_select(0, idx)
    return sample[::2].contiguous()


def assign_queries(q: torch.Tensor, centers: torch.Tensor):
    # Q x C is tiny compared with attention and avoids a materialized Q x C x D.
    q2 = (q.float() * q.float()).sum(-1, keepdim=True)
    c2 = (centers.float() * centers.float()).sum(-1)[None, :]
    dist = q2 + c2 - 2.0 * q.float() @ centers.float().T
    return dist.argmin(-1)


def node_rank8_terms(
    q: torch.Tensor,
    assign: torch.Tensor,
    state,
    node_ids: torch.Tensor,
):
    """Reference vectorized rank8 terms for a small node set (roots/calibration)."""
    ids = node_ids.to(q.device, dtype=torch.long)
    C = state["centers"]
    out_pred = torch.empty((len(q), len(ids)), device=q.device, dtype=torch.float32)
    out_cert = torch.empty_like(out_pred)
    for cj in range(len(C)):
        rows = torch.nonzero(assign == cj, as_tuple=False).flatten()
        if not len(rows):
            continue
        dq = q[rows].float() - C[cj][None]
        logz = state["logz"][cj, ids]
        grad = state["grad"][cj, ids].float()
        lam = state["lam"][cj, ids]
        U = state["U"][cj, ids].float()
        linear = dq @ grad.T
        proj = torch.einsum("qd,ndr->qnr", dq, U)
        q2 = (proj.square() * lam[None]).sum(-1)
        out_pred[rows] = logz[None] + linear + 0.5 * q2
        out_cert[rows] = (0.5 * q2).abs()
    return out_pred, out_cert


def log_expm1_pos(x: torch.Tensor):
    return torch.where(
        x > 20.0,
        x,
        torch.log(torch.expm1(x).clamp_min(1e-30)),
    )


def calibrate_tau(
    q: torch.Tensor,
    state,
    *,
    open_fraction: float = 0.02,
    query_sample: int = 128,
):
    """Train-only root-score threshold, matching HTRC research protocol."""
    train = sampled_query_train(q, count=query_sample)
    assign = assign_queries(train, state["centers"])
    roots = state["forest"]["root_ids"].to(q.device)
    pred, cert = node_rank8_terms(train, assign, state, roots)
    total = torch.logsumexp(pred, dim=1, keepdim=True)
    score = pred - total + log_expm1_pos(cert)
    frac = min(max(float(open_fraction), 1.0e-6), 1.0 - 1.0e-6)
    return float(torch.quantile(score.flatten(), 1.0 - frac).item())


def compact_state_for_eval(state, *, vector_dtype=None):
    """Return contiguous GPU tensors consumed by the online evaluator.

    The large vector state (grad/value/U) may be stored in bf16 while scalar
    routing terms stay fp32. Triton widens vector loads to fp32 in registers.
    """
    dev = state["centers"].device
    forest = state["forest"]
    vd = vector_dtype
    def vec(x):
        return x.contiguous() if vd is None else x.to(dtype=vd).contiguous()
    return {
        "centers": state["centers"].contiguous(),
        "logz": state["logz"].contiguous(),
        "grad": vec(state["grad"]),
        "value": vec(state["value"]),
        "lam": state["lam"].contiguous(),
        "U": vec(state["U"]),
        "trace": state["trace"].contiguous(),
        "captured_trace": state["captured_trace"].contiguous(),
        "root_ids": forest["root_ids"].to(dev).contiguous(),
        "child_idx": forest["child_idx"].to(dev).contiguous(),
        "child_count": forest["child_count"].to(dev).contiguous(),
        "max_depth": int(forest["max_depth"]),
    }


def _next_pow2_capacity(x: float, floor: int = 1) -> int:
    n = max(int(math.ceil(float(x))), int(floor), 1)
    return 1 << (n - 1).bit_length()


@torch.inference_mode()
def bucketed_attention(
    q: torch.Tensor,
    assign: torch.Tensor,
    estate,
    kernels,
    *,
    open_fraction: float,
    exact_k: torch.Tensor | None = None,
    exact_v: torch.Tensor | None = None,
    exact_chunk_rows: int = 128,
    root_capacity_override: int | None = None,
    frontier_trace: list[dict] | None = None,
    frontier_reorder_by_state: bool = False,
):
    """Full HTRC attention with tight static frontier buckets.

    This is the production form of the accepted launch architecture:
    root batched projection/decision followed by three compact frontier levels.
    Accepted node values are accumulated in log-sum-exp form; any capacity
    overflow or residual fourth level is reported to the caller, which fails
    closed rather than silently truncating the tree.
    """
    if q.ndim != 2:
        raise ValueError("HTRC bucketed evaluator expects q=[Q,D]")
    qn, d = (int(v) for v in q.shape)
    centers = estate["centers"].float()
    ccount = int(centers.shape[0])
    roots = estate["root_ids"].to(q.device, dtype=torch.long)
    rcount = int(roots.numel())
    if assign.shape != (qn,):
        raise ValueError("HTRC bucketed assign must be [Q]")

    order = torch.argsort(assign.to(torch.int64)).contiguous()
    counts_t = torch.bincount(
        assign.to(torch.int64), minlength=ccount
    )
    counts = [int(v) for v in counts_t.cpu().tolist()]
    max_rows = max(counts) if counts else 0
    if max_rows <= 0:
        raise RuntimeError("HTRC bucketed evaluator has no query rows")

    qsorted = q.index_select(0, order).float()
    qbatch = torch.zeros(
        (ccount, max_rows, d), device=q.device, dtype=torch.float32
    )
    valid = torch.zeros(
        (ccount, max_rows), device=q.device, dtype=torch.bool
    )
    start = 0
    for cid, count in enumerate(counts):
        stop = start + int(count)
        if count:
            qbatch[cid, :count] = qsorted[start:stop]
            valid[cid, :count] = True
        start = stop

    root_grad = estate["grad"].index_select(1, roots).float()
    root_u = estate["U"].index_select(1, roots).float()

    def _finite_stats_bounded(tensor, *, chunk_elems: int = 1 << 20):
        """Return (bad_count, finite_abs_max) without a full-size bool mask."""
        flat = tensor.reshape(-1)
        total = int(flat.numel())
        bad_total = 0
        finite_abs_max = 0.0
        for start in range(0, total, int(chunk_elems)):
            part = flat[start:min(total, start + int(chunk_elems))]
            finite = torch.isfinite(part)
            if bool(finite.all().item()):
                continue
            bad_total += int((~finite).sum().item())
        if bad_total:
            for start in range(0, total, int(chunk_elems)):
                part = flat[start:min(total, start + int(chunk_elems))]
                finite = torch.isfinite(part)
                if bool(finite.any().item()):
                    # Index only the bounded chunk; never materialize a
                    # full-tensor FP32/safe copy for diagnostics.
                    local = part[finite].float().abs().amax()
                    finite_abs_max = max(
                        finite_abs_max, float(local.item())
                    )
        return int(bad_total), float(finite_abs_max)

    def _require_finite_bmm_input(name, tensor):
        bad, finite_abs_max = _finite_stats_bounded(tensor)
        if bad == 0:
            return
        raise RuntimeError(
            "HTRC root-bmm non-finite input: "
            f"name={name} bad={bad}/{int(tensor.numel())} "
            f"finite_abs_max={finite_abs_max:.6e}"
        )

    _require_finite_bmm_input("centers", centers)
    _require_finite_bmm_input("qbatch", qbatch)
    _require_finite_bmm_input("root_grad", root_grad)
    _require_finite_bmm_input("root_U", root_u)

    # qbatch is no longer needed as an absolute padded-Q tensor after
    # packing: exact-source work uses qsorted real rows below. Reuse its storage
    # for dq instead of retaining two [Qpad,D] FP32 buffers.
    qbatch.sub_(centers[:, None, :])
    dq = qbatch
    _require_finite_bmm_input("dq", dq)

    # Keep the two accepted root contractions separate. The old combined
    # [Qpad,R*9] projection remained live while the root-kernel wrapper made
    # contiguous linear/proj copies of its strided views, transiently doubling
    # the dominant root projection storage. Two BMMs produce the exact same
    # dq·grad and dq·U contractions directly in their final contiguous layouts.
    linear = torch.bmm(
        dq, root_grad.transpose(1, 2).contiguous()
    )
    root_u_weights = (
        root_u.permute(0, 2, 1, 3)
        .reshape(ccount, d, rcount * 8)
        .contiguous()
    )
    proj = torch.bmm(dq, root_u_weights).view(
        ccount, max_rows, rcount, 8
    )
    del root_u_weights
    dq_norm2 = dq.square().sum(-1)
    qpad = ccount * max_rows
    cid = (
        torch.arange(ccount, device=q.device, dtype=torch.int32)[:, None]
        .expand(ccount, max_rows)
        .reshape(-1)
    )

    root_lam = estate["lam"].index_select(1, roots)
    root_logz = estate["logz"].index_select(1, roots)
    root_trace = estate["trace"].index_select(1, roots)
    root_captured_trace = estate["captured_trace"].index_select(1, roots)

    def _require_finite_root_stage(name, tensor):
        bad, finite_abs_max = _finite_stats_bounded(tensor)
        if bad == 0:
            return
        raise RuntimeError(
            "HTRC root-stage non-finite tensor: "
            f"name={name} bad={bad}/{int(tensor.numel())} "
            f"finite_abs_max={finite_abs_max:.6e}"
        )

    for _name, _tensor in (
        ("linear", linear),
        ("proj", proj),
        ("dq_norm2", dq_norm2),
        ("root_lam", root_lam),
        ("root_logz", root_logz),
        ("root_trace", root_trace),
        ("root_captured_trace", root_captured_trace),
    ):
        _require_finite_root_stage(_name, _tensor)

    accepted, score, _opened_unused, log_total = kernels.htrc_root_decision_global(
        linear.reshape(qpad, rcount),
        proj.reshape(qpad, rcount, 8),
        dq_norm2.reshape(qpad),
        cid,
        lam=root_lam,
        logz=root_logz,
        trace=root_trace,
        captured_trace=root_captured_trace,
        dim=d,
        tau=0.0,
    )

    for _name, _tensor in (
        ("accepted", accepted),
        ("score", score),
        ("log_total", log_total),
    ):
        _require_finite_root_stage(_name, _tensor)

    # Root decision outputs own everything needed below. Release the dominant
    # projection tensors and selected root-state FP32 copies before tau/value
    # accumulation rather than carrying them through the whole traversal.
    del (
        linear,
        proj,
        dq_norm2,
        cid,
        root_lam,
        root_logz,
        root_trace,
        root_captured_trace,
        root_grad,
        root_u,
    )

    valid_flat = valid.reshape(-1)
    root_child_count = estate["child_count"].index_select(
        0, roots
    ).to(device=q.device)
    root_can_open = root_child_count > 0
    frac = min(max(float(open_fraction), 1.0e-6), 1.0 - 1.0e-6)
    openable_scores = score[valid_flat][:, root_can_open]
    if openable_scores.numel() == 0:
        tau = float("inf")
    else:
        tau = float(
            torch.quantile(
                openable_scores.flatten(), 1.0 - frac
            ).item()
        )
    opened = (score > tau) & root_can_open[None, :]
    open_valid = opened & valid_flat[:, None]
    root_open_fraction = float(
        open_valid.sum().item()
        / max(
            int(valid_flat.sum().item()) * int(root_can_open.sum().item()),
            1,
        )
    )
    root_mass = accepted.view(ccount, max_rows, rcount)
    accept_root = valid[:, :, None] & (~opened.view(
        ccount, max_rows, rcount
    ))
    del score, _opened_unused, root_child_count, root_can_open
    # accepted masses for opened/padded roots are dead once routing tau is
    # fixed. Convert the accepted tensor itself into normalized root weights:
    # this removes neg_inf, masked_mass and a separate root_w allocation.
    root_mass.masked_fill_(~accept_root, -torch.inf)
    root_m = root_mass.amax(dim=-1)
    root_has = accept_root.any(dim=-1)
    root_m_safe = torch.where(root_has, root_m, torch.zeros_like(root_m))
    root_mass.sub_(root_m_safe[:, :, None]).exp_()
    root_mass.masked_fill_(~accept_root, 0.0)
    root_w = root_mass
    root_z = root_w.sum(dim=-1)
    root_value = estate["value"].index_select(1, roots).float()
    _require_finite_root_stage("root_value", root_value)
    _require_finite_root_stage("root_w", root_w)
    root_n = torch.bmm(root_w, root_value)
    _require_finite_root_stage("root_n", root_n)
    _require_finite_root_stage("root_z", root_z)

    acc_m = root_m.reshape(qpad)
    acc_z = root_z.reshape(qpad)
    acc_n = root_n.reshape(qpad, d)
    del root_w, root_value, root_m_safe, root_has, accept_root, valid

    # Frozen tight launch policy from the runtime benchmark. Capacity scales
    # with real queries, not padded centroid rows: padded rows can never open.
    # Real 243-frame topology is broader below the roots than the
    # synthetic launch matrix: the observed first expansion was 86,200 rows
    # for ~76k queries, exceeding the qn*0.8 bucket. Keep the root bucket from
    # the validated tight policy, then use conservative geometric headroom for
    # deeper real-tree levels. These remain static device-side capacities.
    # Root frontier follows the validated 2%-open policy:
    # Q * roots(80) * 0.02 * fanout(4) = 6.4 * Q. Deeper frontier sizes are
    # determined exactly from next_count before expansion, avoiding synthetic
    # topology assumptions while reusing the host synchronization that the
    # old overflow check already required.
    root_cap = (
        int(root_capacity_override)
        if root_capacity_override is not None
        else _next_pow2_capacity(qn * 6.4, 2048)
    )
    if root_cap < 1:
        raise ValueError("HTRC root_capacity_override must be positive")
    frontier_levels = max(1, int(estate.get("max_depth", 1)))

    # Allocation-free analytical lifetime plan for traversal-local tensors.
    # This deliberately excludes externally-owned full-head geometry/state and
    # reports only buffers created by bucketed_attention itself.
    global _HTRC_MEMORY_PLAN_LOGGED
    if not _HTRC_MEMORY_PLAN_LOGGED:
        _HTRC_MEMORY_PLAN_LOGGED = True
        q_sorted_b = qn * d * 4
        qpad_vec_b = qpad * d * 4
        root_grad_b = ccount * rcount * d * 4
        root_u_b = ccount * rcount * d * 8 * 4
        linear_b = qpad * rcount * 4
        root_u_weights_b = ccount * d * rcount * 8 * 4
        proj_b = qpad * rcount * 8 * 4
        root_matrix_b = qpad * rcount * 4
        root_mask_b = qpad * rcount
        root_value_b = ccount * rcount * d * 4
        root_n_b = qpad_vec_b
        root_projection_local = (
            q_sorted_b + qpad_vec_b + qpad
            + root_grad_b + root_u_b + linear_b
            + root_u_weights_b + proj_b
        )
        root_decision_local = (
            q_sorted_b + qpad_vec_b + qpad
            + root_grad_b + root_u_b + linear_b + proj_b
            + 2 * root_matrix_b + root_mask_b + qpad * 4
        )
        root_value_local = (
            q_sorted_b + qpad_vec_b
            + root_matrix_b + root_mask_b + qpad * 4
            + root_value_b + root_n_b
        )
        frontier_level_local = (
            q_sorted_b
            + qpad_vec_b
            + 2 * qpad_vec_b
            + 5 * qpad * 4
            + root_cap * (4 + 4 + 4 + 1 + 4)
        )
        exact_rows_local = max(counts) if counts else 0
        exact_kv_rows_local = min(
            max(1, int(exact_chunk_rows)),
            max(
                1,
                int(exact_k.shape[0])
                if exact_k is not None
                else 1,
            ),
        )
        exact_local = (
            q_sorted_b
            + qpad_vec_b
            + 2 * qpad * 4
            + exact_rows_local * exact_kv_rows_local * 4
            + 2 * exact_rows_local * d * 4
            + 8 * exact_rows_local * 4
        )
        final_local = (
            qpad_vec_b
            + qpad * 4
            + qn * d * 4
            + max_rows * d * 4
        )
        _LOG.warning(
            "HTRC analytic traversal plan: q=%d qpad=%d max_rows=%d "
            "root_cap=%d local_mib(root_proj=%.1f root_decision=%.1f "
            "root_value=%.1f frontier=%.1f exact=%.1f final=%.1f)",
            int(qn),
            int(qpad),
            int(max_rows),
            int(root_cap),
            root_projection_local / (1024.0 ** 2),
            root_decision_local / (1024.0 ** 2),
            root_value_local / (1024.0 ** 2),
            frontier_level_local / (1024.0 ** 2),
            exact_local / (1024.0 ** 2),
            final_local / (1024.0 ** 2),
        )

    fq, fn, fc = kernels.htrc_frontier_compact(
        open_valid,
        estate["root_ids"],
        estate["child_idx"],
        estate["child_count"],
        capacity=root_cap,
    )

    def _append_frontier_trace(level_index: int, active_count: int) -> None:
        if frontier_trace is None or active_count <= 0:
            return
        count = min(int(active_count), int(fq.numel()))
        qidx = fq[:count].to(torch.int64)
        nidx = fn[:count].to(torch.int64)
        # Frontier q-indices are packed qpad row IDs.  Since qpad is grouped by
        # query centroid, cid=fq//max_rows identifies the actual state slab read
        # by this row.  The online state key is therefore (cid,node), not node
        # alone.
        cid_idx = torch.div(qidx, int(max_rows), rounding_mode="floor")
        ncount = int(estate["logz"].shape[1])
        state_key = cid_idx * ncount + nidx
        unique_keys, key_counts = torch.unique(
            state_key, return_counts=True
        )
        unique_nodes = torch.unique(nidx)
        multiplicity = key_counts.float()
        # Current frontier order locality: count transitions between distinct
        # state keys.  A perfect node-major schedule would approach one segment
        # per unique state key.
        if count > 1:
            segments = int(
                1 + (state_key[1:] != state_key[:-1]).sum().item()
            )
        else:
            segments = 1
        frontier_trace.append(
            {
                "level": int(level_index),
                "frontier_rows": int(count),
                "unique_nodes": int(unique_nodes.numel()),
                "unique_centroid_node_states": int(unique_keys.numel()),
                "current_state_key_segments": int(segments),
                "mean_queries_per_state": float(multiplicity.mean().item()),
                "max_queries_per_state": int(key_counts.max().item()),
                "state_row_reuse_factor": float(
                    count / max(int(unique_keys.numel()), 1)
                ),
                "segment_reduction_upper_bound": float(
                    segments / max(int(unique_keys.numel()), 1)
                ),
            }
        )
    caps_used = [int(root_cap)]
    first_required = int(fc.item())
    required = [int(first_required)]
    current_count = int(first_required)
    _append_frontier_trace(0, current_count)
    del open_valid, opened, valid_flat
    if first_required > int(root_cap):
        raise RuntimeError(
            "HTRC bucketed traversal initial frontier overflow: "
            f"required={first_required} capacity={int(root_cap)} "
            f"root_open_fraction={root_open_fraction:.6f}"
        )

    def merge_level(lm, lz, ln):
        nonlocal acc_m, acc_z, acc_n
        has_old = acc_z > 0
        has_new = lz > 0
        both = has_old & has_new
        nm = torch.where(
            both,
            torch.maximum(acc_m, lm),
            torch.where(has_new, lm, acc_m),
        )
        old_scale = torch.where(
            has_old, torch.exp(acc_m - nm), torch.zeros_like(acc_m)
        )
        new_scale = torch.where(
            has_new, torch.exp(lm - nm), torch.zeros_like(lm)
        )
        # Merge numerators in-place. The previous expression materialized
        # two additional [qpad,D] FP32 products plus the summed result while
        # acc_n and ln were already live. ln is dead after this merge, so it is
        # safe to consume it as the scaled-new workspace.
        acc_n.mul_(old_scale[:, None])
        ln.mul_(new_scale[:, None])
        acc_n.add_(ln)
        acc_z.mul_(old_scale)
        lz.mul_(new_scale)
        acc_z.add_(lz)
        acc_m = nm

    for level in range(frontier_levels):
        cap = int(fq.numel())
        if frontier_reorder_by_state and current_count > 1:
            active_count = min(int(current_count), cap)
            q_active = fq[:active_count].to(torch.int64)
            n_active = fn[:active_count].to(torch.int64)
            cid_active = torch.div(
                q_active, int(max_rows), rounding_mode="floor"
            )
            state_key = (
                cid_active * int(estate["logz"].shape[1]) + n_active
            )
            order_state = torch.argsort(state_key)
            fq_active = fq[:active_count].index_select(0, order_state)
            fn_active = fn[:active_count].index_select(0, order_state)
            fq = torch.cat([fq_active, fq[active_count:]], dim=0)
            fn = torch.cat([fn_active, fn[active_count:]], dim=0)
        mass_l, open_l, next_l = kernels.htrc_frontier_step_grouped(
            dq.reshape(qpad, d).contiguous(),
            fq,
            fn,
            fc,
            log_total,
            logz=estate["logz"],
            grad=estate["grad"],
            lam=estate["lam"],
            U=estate["U"],
            trace=estate["trace"],
            captured_trace=estate["captured_trace"],
            child_count=estate["child_count"],
            rows_per_centroid=max_rows,
            capacity=cap,
            tau=float(tau),
        )
        lm, lz, ln = kernels.htrc_frontier_level_value(
            fq,
            fn,
            fc,
            mass_l,
            open_l,
            value=estate["value"],
            rows_per_centroid=max_rows,
            qpad=qpad,
        )
        merge_level(lm, lz, ln)
        del mass_l, open_l, lm, lz, ln

        active_count = min(int(current_count), cap)
        next_required = int(
            next_l[:active_count].sum(dtype=torch.int64).item()
        )
        required.append(int(next_required))
        if next_required == 0:
            current_count = 0
            break

        next_cap = _next_pow2_capacity(next_required, 2048)
        fq, fn, fc = kernels.htrc_frontier_expand(
            fq,
            fn,
            fc,
            next_l,
            estate["child_idx"],
            capacity=next_cap,
        )
        caps_used.append(int(next_cap))
        current_count = int(next_required)
        _append_frontier_trace(level + 1, current_count)

    # One synchronization for all capacity/depth and accumulator guards.
    # Bounded kernels suppress OOB writes; the caller fails closed on violation.
    req = tuple(int(x) for x in required)
    overflow = False
    residual = int(req[-1]) > 0
    if residual:
        raise RuntimeError(
            "HTRC bucketed traversal exceeded forest max_depth: "
            f"max_depth={frontier_levels} "
            f"required={tuple(int(x) for x in req)}"
        )

    # Tree traversal is complete. Exact-source attention needs neither
    # dq/log_total nor frontier buffers; release them before its matmuls.
    del fq, fn, fc, dq, log_total

    # Exact sources (conditioning sink plus incomplete final leaf) are
    # excluded from the HTRC tree and merged here in bounded chunks.
    exact_tokens = 0
    if exact_k is not None or exact_v is not None:
        if exact_k is None or exact_v is None:
            raise ValueError("HTRC exact K/V must be supplied together")
        if exact_k.ndim != 2 or exact_v.ndim != 2:
            raise ValueError("HTRC exact K/V must be [Texact,D]")
        if exact_k.shape != exact_v.shape:
            raise ValueError("HTRC exact K/V shapes must match")
        if int(exact_k.shape[1]) != d:
            raise ValueError("HTRC exact K/V head dimension mismatch")
        exact_tokens = int(exact_k.shape[0])
        chunk_rows = max(1, int(exact_chunk_rows))
        if exact_tokens > 0:
            # Query rows are already packed by centroid in qbatch. Process one
            # centroid slab at a time so exact-source logits/value numerators
            # never materialize for all qpad rows simultaneously. Each query
            # still sees the identical exact K/V chunks and log-sum-exp merge;
            # only independent query rows are scheduled in smaller batches.
            sorted_start = 0
            for cid_i, count in enumerate(counts):
                qa = cid_i * max_rows
                qb = qa + int(count)
                qslice = qsorted[
                    sorted_start:sorted_start + int(count)
                ]
                sorted_start += int(count)
                if int(count) <= 0:
                    continue
                for es in range(0, exact_tokens, chunk_rows):
                    ee = min(exact_tokens, es + chunk_rows)
                    kt = exact_k[es:ee].float()
                    vt = exact_v[es:ee].float()
                    logits = (qslice @ kt.T) / math.sqrt(float(d))
                    em = logits.amax(dim=-1)
                    logits.sub_(em[:, None]).exp_()
                    ez = logits.sum(dim=-1)
                    en = logits @ vt

                    old_m = acc_m[qa:qb]
                    old_z = acc_z[qa:qb]
                    old_n = acc_n[qa:qb]
                    has_old = old_z > 0
                    has_new = ez > 0
                    both = has_old & has_new
                    nm = torch.where(
                        both,
                        torch.maximum(old_m, em),
                        torch.where(has_new, em, old_m),
                    )
                    old_scale = torch.where(
                        has_old,
                        torch.exp(old_m - nm),
                        torch.zeros_like(old_m),
                    )
                    new_scale = torch.where(
                        has_new,
                        torch.exp(em - nm),
                        torch.zeros_like(em),
                    )
                    old_n.mul_(old_scale[:, None])
                    en.mul_(new_scale[:, None])
                    old_n.add_(en)
                    old_z.mul_(old_scale)
                    ez.mul_(new_scale)
                    old_z.add_(ez)
                    old_m.copy_(nm)


    del acc_m, qsorted

    # Validate and normalize only real rows one centroid slab at a time.
    # Avoid materializing valid_num [Q,D], out_pad [Qpad,D], and sorted_out
    # [Q,D] simultaneously. Write normalized rows directly to final order.
    out = torch.empty((qn, d), device=q.device, dtype=torch.float32)
    bad_rows_total = 0
    bad_samples = []
    sorted_start = 0
    for cid_i, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        qa = cid_i * max_rows
        qb = qa + count
        zslice = acc_z[qa:qb]
        nslice = acc_n[qa:qb]
        bad_z_nonfinite = ~torch.isfinite(zslice)
        bad_z_empty = zslice <= 0
        bad_n_nonfinite = ~torch.isfinite(nslice).all(dim=-1)
        bad_mask = bad_z_nonfinite | bad_z_empty | bad_n_nonfinite
        if bool(bad_mask.any().item()):
            local_bad = torch.nonzero(
                bad_mask, as_tuple=False
            ).flatten()
            bad_rows_total += int(local_bad.numel())
            if len(bad_samples) < 16:
                take = local_bad[:16 - len(bad_samples)]
                for local_i in take.detach().cpu().tolist():
                    bad_samples.append(
                        (
                            int(sorted_start + local_i),
                            float(zslice[local_i].detach().cpu().item()),
                            bool(bad_z_nonfinite[local_i].item()),
                            bool(bad_z_empty[local_i].item()),
                            bool(bad_n_nonfinite[local_i].item()),
                        )
                    )
        if bad_rows_total == 0:
            normalized = nslice / zslice.clamp_min(1.0e-20)[:, None]
            out.index_copy_(
                0,
                order[sorted_start:sorted_start + count],
                normalized,
            )
        sorted_start += count

    if bad_rows_total:
        raise RuntimeError(
            "HTRC bucketed traversal produced an empty/non-finite accumulator "
            f"for {int(bad_rows_total)} real queries; "
            f"samples={bad_samples} "
            f"required={tuple(int(x) for x in req)} "
            f"caps={tuple(int(x) for x in caps_used)}"
        )
    # Correctness/status fields are authoritative and must remain
    # fail-closed above. Optional diagnostics are best-effort only: a broken
    # metric must never discard an otherwise valid attention result.
    info = {
        "caps": tuple(int(x) for x in caps_used),
        "required": tuple(int(x) for x in req),
        "overflow": bool(overflow),
        "residual_frontier": bool(residual),
    }
    try:
        info.update(
            {
                "max_rows": int(max_rows),
                "qpad": int(qpad),
                "tau": float(tau),
                "root_open_fraction": float(root_open_fraction),
                "exact_tokens": int(exact_tokens),
            }
        )
    except Exception as exc:
        _LOG.warning(
            "HTRC optional traversal telemetry failed harmlessly: %r",
            exc,
        )
        info["telemetry_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
    return out, info


__all__ = [
    "prepare_leaf_geometry",
    "numeric_telemetry_snapshot",
    "numeric_telemetry_delta",
    "select_head_geometry",
    "build_head_state",
    "build_root_kmeans_structural_forest",
    "assign_queries",
    "calibrate_tau",
    "compact_state_for_eval",
    "bucketed_attention",
]
