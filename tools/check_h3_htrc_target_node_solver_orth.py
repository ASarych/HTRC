from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    return float(
        torch.linalg.vector_norm(af - bf)
        / torch.linalg.vector_norm(bf).clamp_min(1e-20)
    )


def orth_error(Q: torch.Tensor) -> float:
    k = int(Q.shape[-1])
    eye = torch.eye(k, device=Q.device, dtype=torch.float32)[None]
    gram = torch.matmul(Q.float().transpose(1, 2), Q.float())
    return rel_l2(gram, eye.expand_as(gram))


@torch.inference_mode()
def selected_basis_stats(U: torch.Tensor):
    r = int(U.shape[-1])
    eye = torch.eye(r, device=U.device, dtype=torch.float32)[None]
    gram = torch.matmul(U.float().transpose(1, 2), U.float())
    diag = torch.diagonal(gram, dim1=-2, dim2=-1)
    return {
        "selected_orth_rel_l2": rel_l2(gram, eye.expand_as(gram)),
        "selected_norm_min": float(torch.sqrt(diag.clamp_min(0)).min()),
        "selected_norm_max": float(torch.sqrt(diag.clamp_min(0)).max()),
    }


@torch.inference_mode()
def effective_rank_stats(Q: torch.Tensor, threshold: float = 1e-6):
    gram = torch.matmul(Q.float().transpose(1, 2), Q.float())
    diag = torch.diagonal(gram, dim1=-2, dim2=-1)
    active = diag > float(threshold)
    rank = active.sum(dim=1)
    return {
        "effective_rank_min": int(rank.min()),
        "effective_rank_mean": float(rank.float().mean()),
        "effective_rank_max": int(rank.max()),
        "targets_rank_lt8": int((rank < 8).sum()),
        "targets_rank_lt16": int((rank < int(Q.shape[-1])).sum()),
    }


@torch.inference_mode()
def gram_stats(Q: torch.Tensor):
    q = Q.float()
    gram = torch.matmul(q.transpose(1, 2), q)
    eig = torch.linalg.eigvalsh(gram)
    diag = torch.diagonal(gram, dim1=-2, dim2=-1)
    eye = torch.eye(
        int(Q.shape[-1]), device=Q.device, dtype=torch.bool
    )[None]
    off = gram.masked_fill(eye, 0.0).abs()
    return {
        "gram_eig_min": float(eig.min()),
        "gram_eig_max": float(eig.max()),
        "gram_diag_min": float(diag.min()),
        "gram_diag_max": float(diag.max()),
        "gram_max_abs_offdiag": float(off.max()),
    }


@torch.inference_mode()
def prepare_pairs(leaf, node_leaf_idx, node_leaf_mask, cids, nids):
    ids = node_leaf_idx.index_select(0, nids)
    mask = node_leaf_mask.index_select(0, nids)
    lz = leaf["logz"][cids[:, None], ids]
    g = leaf["grad"][cids[:, None], ids].float()
    lz = lz.masked_fill(~mask, -torch.inf)
    w = torch.softmax(lz, dim=1)
    pg = torch.einsum("ml,mld->md", w, g)
    return w.contiguous(), pg.contiguous()


@torch.inference_mode()
def cholqr(X: torch.Tensor, passes: int = 1, eps: float = 1e-6):
    Q = X.float()
    k = int(Q.shape[-1])
    eye = torch.eye(k, device=Q.device, dtype=torch.float32)[None]
    for _ in range(int(passes)):
        gram = torch.matmul(Q.transpose(1, 2), Q)
        scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(-1)
        jitter = (scale * float(eps)).clamp_min(1e-12)
        L = torch.linalg.cholesky(gram + jitter[:, None, None] * eye)
        qt = torch.linalg.solve_triangular(
            L, Q.transpose(1, 2), upper=False
        )
        Q = qt.transpose(1, 2).contiguous()
    return Q


@torch.inference_mode()
def solve(
    kernels,
    *,
    w,
    pg,
    cids,
    nids,
    node_leaf_idx,
    node_leaf_mask,
    leaf,
    block_dim,
    iterations,
    seed,
    mode,
    k_tile,
):
    m = int(cids.numel())
    d = int(pg.shape[-1])
    k = min(int(block_dim), d)
    gen = torch.Generator(device=pg.device)
    gen.manual_seed(int(seed))
    Q = torch.randn(
        (m, d, k), generator=gen, device=pg.device, dtype=torch.float32
    )

    if mode == "qr":
        Q, _ = torch.linalg.qr(Q, mode="reduced")
    elif mode == "cholqr":
        Q = cholqr(Q, passes=1)
    elif mode == "cholqr2":
        Q = cholqr(Q, passes=2)
    elif mode == "noinit_qr":
        pass
    elif mode == "noinit_cholqr":
        pass
    elif mode == "noinit_cholqr2":
        pass
    elif mode == "mgs":
        Q = kernels.htrc_mgs(Q, passes=1)
    elif mode == "mgs2":
        Q = kernels.htrc_mgs(Q, passes=2)
    elif mode in (
        "noinit_mgs",
        "noinit_mgs2",
        "noinit_mgs2_polar1",
        "noinit_mgs2_polar2",
    ):
        pass
    else:
        raise ValueError(f"unknown mode: {mode}")

    def apply(X):
        return kernels.htrc_target_h_apply_tiled(
            X,
            w,
            pg,
            cids,
            nids,
            node_leaf_idx,
            node_leaf_mask,
            leaf["grad"],
            leaf["evals"],
            leaf["modes"],
            k_tile=int(k_tile),
        )

    for _ in range(int(iterations)):
        Y = apply(Q)
        Y = apply(Y)
        if mode in ("qr", "noinit_qr"):
            Q, _ = torch.linalg.qr(Y, mode="reduced")
        elif mode in ("cholqr", "noinit_cholqr"):
            Q = cholqr(Y, passes=1)
        elif mode in ("cholqr2", "noinit_cholqr2"):
            Q = cholqr(Y, passes=2)
        elif mode in ("mgs", "noinit_mgs"):
            Q = kernels.htrc_mgs(Y, passes=1)
        elif mode in (
            "mgs2",
            "noinit_mgs2",
            "noinit_mgs2_polar1",
            "noinit_mgs2_polar2",
        ):
            Q = kernels.htrc_mgs(Y, passes=2)
            if mode == "noinit_mgs2_polar1":
                Q = kernels.htrc_polar_retract(Q, steps=1)
            elif mode == "noinit_mgs2_polar2":
                Q = kernels.htrc_polar_retract(Q, steps=2)
        else:
            raise ValueError(f"unknown mode: {mode}")

    HQ = apply(Q)
    T = torch.matmul(Q.transpose(1, 2), HQ)
    T = 0.5 * (T + T.transpose(1, 2))
    ev, V = torch.linalg.eigh(T)
    order = torch.argsort(ev.abs(), dim=1, descending=True)[:, :8]
    lam = torch.gather(ev, 1, order)
    idx = order[:, None, :].expand(-1, V.shape[1], -1)
    U = torch.matmul(Q, torch.gather(V, 2, idx))
    return lam.contiguous(), U.contiguous(), Q


def bench_cuda(fn, repeats: int):
    out = fn()
    del out
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(int(repeats)):
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        vals.append(float(start.elapsed_time(end)))
        del out
    return {
        "median_ms": statistics.median(vals),
        "mean_ms": statistics.mean(vals),
        "min_ms": min(vals),
        "max_ms": max(vals),
    }


def peak_delta_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated())
    del out
    torch.cuda.synchronize()
    return max(0, peak - base)


@torch.inference_mode()
def action_probe(lam, U, probe):
    p = torch.einsum("mdr,mdk->mrk", U.float(), probe.float())
    return torch.einsum("mdr,mr,mrk->mdk", U.float(), lam.float(), p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--physical-head", type=int, default=27)
    ap.add_argument("--batch-sizes", default="512,1024,2560")
    ap.add_argument(
        "--modes",
        default="qr,noinit_qr,noinit_mgs2,noinit_mgs2_polar1,noinit_mgs2_polar2",
    )
    ap.add_argument("--block-dim", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--k-tile", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    dev = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]
    htrc = load_module(
        "apl_h3_solver_htrc", root / "h3" / "htrc_live.py"
    )
    kernels = load_module(
        "apl_h3_solver_kernels", root / "h3" / "rrc3_kernels.py"
    )
    helpers = load_module(
        "apl_h3_solver_helpers",
        root / "tools" / "check_h3_htrc_direct_lazy_builder.py",
    )

    try:
        obj = torch.load(a.raw, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(a.raw, map_location="cpu", mmap=True)

    raw_heads = obj.get("raw_head_index")
    mapping = (
        [int(x) for x in raw_heads.tolist()]
        if isinstance(raw_heads, torch.Tensor)
        else list(range(int(obj["q_qhd"].shape[1])))
    )
    ph = int(a.physical_head)
    if ph not in mapping:
        raise RuntimeError(f"raw capture lacks head {ph}; mapping={mapping}")
    si = mapping.index(ph)

    q = obj["q_qhd"][:, si].float().to(dev).contiguous()
    k = obj["k_khd"][:, si].to(dev).contiguous()[:, None, :]
    v = obj["v_khd"][:, si].to(dev).contiguous()[:, None, :]
    geom = htrc.prepare_leaf_geometry(k, v, leaf_work_chunk=32)
    g = htrc.select_head_geometry(geom, 0)
    centers = htrc.sampled_query_centers(q, count=128, clusters=32)
    forest = htrc.build_root_kmeans_implicit_forest(
        g["km"].detach().cpu(),
        roots=80,
        fanout=4,
        proj_dim=24,
        iters=8,
        seed=20260925,
        head_index=ph,
    )
    leaf = helpers.build_leaf_authority(
        htrc, centers, g, 4096, centroid_chunk=4
    )
    node_leaf_idx, node_leaf_mask = helpers.prepare_node_leaf_table(
        forest, dev
    )

    C = int(centers.shape[0])
    roots = forest["root_ids"].to(dev, dtype=torch.long)
    all_cids = torch.arange(C, device=dev, dtype=torch.long)[:, None].expand(
        C, int(roots.numel())
    ).reshape(-1)
    all_nids = roots[None, :].expand(C, -1).reshape(-1)
    total = int(all_cids.numel())

    batches = [int(x) for x in a.batch_sizes.split(",") if x.strip()]
    modes = [x.strip() for x in a.modes.split(",") if x.strip()]
    allowed = {
        "qr", "cholqr", "cholqr2", "mgs", "mgs2",
        "noinit_qr", "noinit_cholqr", "noinit_cholqr2",
        "noinit_mgs", "noinit_mgs2",
        "noinit_mgs2_polar1", "noinit_mgs2_polar2",
    }
    bad = [x for x in modes if x not in allowed]
    if bad:
        raise ValueError(f"unsupported modes: {bad}")

    report = {
        "format": "apl-h3-htrc-target-node-solver-orth-v1",
        "raw": str(a.raw),
        "physical_head": ph,
        "device": torch.cuda.get_device_name(dev),
        "block_dim": int(a.block_dim),
        "iterations": int(a.iterations),
        "k_tile": int(a.k_tile),
        "modes": modes,
        "cases": [],
    }

    for requested in batches:
        m = min(int(requested), total)
        print(f"stage=batch start m={m}", flush=True)
        cids = all_cids[:m].contiguous()
        nids = all_nids[:m].contiguous()
        w, pg = prepare_pairs(
            leaf, node_leaf_idx, node_leaf_mask, cids, nids
        )
        probe = torch.randn(
            (m, int(q.shape[-1]), 4), device=dev, dtype=torch.float32
        )

        baseline = solve(
            kernels,
            w=w,
            pg=pg,
            cids=cids,
            nids=nids,
            node_leaf_idx=node_leaf_idx,
            node_leaf_mask=node_leaf_mask,
            leaf=leaf,
            block_dim=int(a.block_dim),
            iterations=int(a.iterations),
            seed=20260927 + ph + m,
            mode="qr",
            k_tile=int(a.k_tile),
        )
        base_lam, base_u, base_q = baseline
        base_action = action_probe(base_lam, base_u, probe)

        row = {"requested_batch": int(requested), "batch": m, "modes": {}}
        for mode in modes:
            print(f"stage=mode m={m} mode={mode}", flush=True)
            try:
                lam, U, Q = solve(
                    kernels,
                    w=w,
                    pg=pg,
                    cids=cids,
                    nids=nids,
                    node_leaf_idx=node_leaf_idx,
                    node_leaf_mask=node_leaf_mask,
                    leaf=leaf,
                    block_dim=int(a.block_dim),
                    iterations=int(a.iterations),
                    seed=20260927 + ph + m,
                    mode=mode,
                    k_tile=int(a.k_tile),
                )
                action = action_probe(lam, U, probe)
                quality = {
                    "action_rel_l2_vs_qr": rel_l2(action, base_action),
                    "orth_rel_l2": orth_error(Q),
                    "lam_rel_l2_vs_qr": rel_l2(lam, base_lam),
                    **gram_stats(Q),
                    **effective_rank_stats(Q),
                    **selected_basis_stats(U),
                }
                del lam, U, Q, action

                timing = bench_cuda(
                    lambda mode=mode: solve(
                        kernels,
                        w=w,
                        pg=pg,
                        cids=cids,
                        nids=nids,
                        node_leaf_idx=node_leaf_idx,
                        node_leaf_mask=node_leaf_mask,
                        leaf=leaf,
                        block_dim=int(a.block_dim),
                        iterations=int(a.iterations),
                        seed=20260927 + ph + m,
                        mode=mode,
                        k_tile=int(a.k_tile),
                    ),
                    int(a.repeats),
                )
                peak = peak_delta_bytes(
                    lambda mode=mode: solve(
                        kernels,
                        w=w,
                        pg=pg,
                        cids=cids,
                        nids=nids,
                        node_leaf_idx=node_leaf_idx,
                        node_leaf_mask=node_leaf_mask,
                        leaf=leaf,
                        block_dim=int(a.block_dim),
                        iterations=int(a.iterations),
                        seed=20260927 + ph + m,
                        mode=mode,
                        k_tile=int(a.k_tile),
                    )
                )
                row["modes"][mode] = {
                    **quality,
                    "timing": timing,
                    "peak_delta_bytes": peak,
                    "oom": False,
                }
            except (torch.cuda.OutOfMemoryError, torch.linalg.LinAlgError) as exc:
                row["modes"][mode] = {
                    "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
                    "error": str(exc),
                }
                torch.cuda.empty_cache()

        report["cases"].append(row)
        del base_lam, base_u, base_q, base_action, baseline
        del cids, nids, w, pg, probe
        torch.cuda.empty_cache()

    valid = []
    for case in report["cases"]:
        for mode, mr in case["modes"].items():
            if "timing" in mr:
                valid.append((case["batch"], mode, mr))
    if valid:
        best = min(valid, key=lambda x: x[2]["timing"]["median_ms"] / x[0])
        report["summary"] = {
            "best_batch": int(best[0]),
            "best_mode": best[1],
            "best_ms_per_target": (
                best[2]["timing"]["median_ms"] / float(best[0])
            ),
            "best_action_rel_l2_vs_qr": best[2]["action_rel_l2_vs_qr"],
            "best_orth_rel_l2": best[2]["orth_rel_l2"],
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"JSON: {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
