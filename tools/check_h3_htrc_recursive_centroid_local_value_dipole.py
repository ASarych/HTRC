from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    return float(
        torch.linalg.vector_norm(af - bf)
        / torch.linalg.vector_norm(bf).clamp_min(1e-20)
    )


def leaf_value_jacobian(center: torch.Tensor, g, summary):
    """Exact antipodal leaf conditional-value Jacobian at one centroid.

    Returns [B,Dv,Dq].  This is the historical first-order value-dipole
    authority written directly from the pair rank-1 derivative identity.
    """
    q = center.float()
    a = torch.einsum("d,bpd->bp", q, g["c"])
    h = torch.einsum("d,bpd->bp", q, g["r"])
    t = torch.einsum("d,bd->b", q, g["kappa"])

    ch = torch.cosh(h)
    sh = torch.sinh(h)
    e = (1.0 + 0.5 * a).square() + 1.0e-8
    de = 1.0 + 0.5 * a
    et = 2.0 * torch.exp(t[:, None])

    base = g["vm"][:, None, :] + g["M"]
    F = (
        e[..., None]
        * (ch[..., None] * base + sh[..., None] * g["Dv"])
        + de[..., None] * ch[..., None] * g["M"]
    )
    Cvec = (
        de[..., None]
        * (ch[..., None] * base + sh[..., None] * g["Dv"])
        + 0.5 * ch[..., None] * g["M"]
    )
    Rvec = (
        e[..., None]
        * (sh[..., None] * base + ch[..., None] * g["Dv"])
        + de[..., None] * sh[..., None] * g["M"]
    )

    dN = (
        torch.einsum("bp,bpe,bd->bed", et, F, g["kappa"])
        + torch.einsum("bp,bpe,bpd->bed", et, Cvec, g["c"])
        + torch.einsum("bp,bpe,bpd->bed", et, Rvec, g["r"])
    )
    Z = summary["Z"][0].float().clamp_min(1.0e-20)
    V = summary["V0"][0].float()
    grad = summary["grad"][0].float()
    A = dN / Z[:, None, None]
    J = A - V[:, :, None] * grad[:, None, :]
    return J.contiguous()


def recursive_merge(summary, leaf_j, forest, plan, device):
    """Exact recursive merge of (logZ,g,V,J) from leaves to all tree nodes."""
    leaf_logz = summary["logZ"][0].float()
    leaf_g = summary["grad"][0].float()
    leaf_v = summary["V0"][0].float()
    B, D = leaf_g.shape
    N = len(forest["nodes"])
    lids = forest["leaf_node_for_block"].to(device).long()

    logz = torch.empty((N,), device=device, dtype=torch.float32)
    grad = torch.empty((N, D), device=device, dtype=torch.float32)
    value = torch.empty((N, D), device=device, dtype=torch.float32)
    jac = torch.empty((N, D, D), device=device, dtype=torch.float32)

    logz[lids] = leaf_logz
    grad[lids] = leaf_g
    value[lids] = leaf_v
    jac[lids] = leaf_j

    for depth in range(1, int(forest["max_depth"]) + 1):
        if depth not in plan:
            continue
        tids, cidx, mask = plan[depth]
        clz = logz[cidx]
        clz = torch.where(mask, clz, torch.full_like(clz, -torch.inf))
        plz = torch.logsumexp(clz, dim=1)
        w = torch.softmax(clz, dim=1) * mask

        child_g = grad[cidx]
        child_v = value[cidx]
        child_j = jac[cidx]

        pg = (w[..., None] * child_g).sum(dim=1)
        pv = (w[..., None] * child_v).sum(dim=1)
        cov = torch.einsum(
            "mf,mfe,mfd->med",
            w,
            child_v - pv[:, None, :],
            child_g,
        )
        pj = (w[..., None, None] * child_j).sum(dim=1) + cov

        logz[tids] = plz
        grad[tids] = pg
        value[tids] = pv
        jac[tids] = pj

    return logz, grad, value, jac


def direct_from_leaves(nid: int, summary, leaf_j, forest, device):
    node = forest["nodes"][int(nid)]
    bids = node.bids.to(device=device, dtype=torch.long)
    lz = summary["logZ"][0].float().index_select(0, bids)
    g = summary["grad"][0].float().index_select(0, bids)
    v = summary["V0"][0].float().index_select(0, bids)
    j = leaf_j.index_select(0, bids)

    w = torch.softmax(lz, dim=0)
    pg = (w[:, None] * g).sum(dim=0)
    pv = (w[:, None] * v).sum(dim=0)
    pj = (
        (w[:, None, None] * j).sum(dim=0)
        + torch.einsum("b,be,bd->ed", w, v - pv[None, :], g)
    )
    plz = torch.logsumexp(lz, dim=0)
    return plz, pg, pv, pj


def sample_ids(ids, count: int):
    ids = [int(x) for x in ids]
    if len(ids) <= count:
        return ids
    pos = torch.linspace(0, len(ids) - 1, count).round().long().tolist()
    return [ids[i] for i in pos]


def rank_metrics(batch: torch.Tensor, ranks):
    if int(batch.shape[0]) == 0:
        return None
    s = torch.linalg.svdvals(batch.float())
    e = s.square()
    total = e.sum(dim=-1).clamp_min(1.0e-30)
    cum = torch.cumsum(e, dim=-1) / total[:, None]

    def need(frac: float):
        return (cum < frac).sum(dim=-1) + 1

    out = {
        "matrices": int(batch.shape[0]),
        "rank99_mean": float(need(0.99).float().mean().item()),
        "rank99_max": int(need(0.99).max().item()),
        "rank999_mean": float(need(0.999).float().mean().item()),
        "rank999_max": int(need(0.999).max().item()),
        "approx_rel_frob": {},
    }
    for r in ranks:
        rr = min(int(r), int(s.shape[-1]))
        tail = e[:, rr:].sum(dim=-1)
        err = torch.sqrt(tail / total)
        out["approx_rel_frob"][str(r)] = {
            "mean": float(err.mean().item()),
            "max": float(err.max().item()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Verify exact recursive aggregation of the accepted centroid-local "
            "HTRC value dipole and measure internal-node Jacobian rank."
        )
    )
    ap.add_argument("raw", type=Path)
    ap.add_argument("--physical-heads", default="1,16,27,55")
    ap.add_argument("--centroids", type=int, default=4)
    ap.add_argument("--nodes-per-depth", type=int, default=8)
    ap.add_argument("--ranks", default="4,8,16,24,32")
    ap.add_argument("--leaf-work-chunk", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    dev = torch.device("cuda")
    root = Path(__file__).resolve().parents[1]
    htrc = load_module("apl_h3_recursive_value_dipole", root / "h3" / "htrc_live.py")

    try:
        obj = torch.load(
            a.raw.expanduser().resolve(),
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        obj = torch.load(a.raw.expanduser().resolve(), map_location="cpu", mmap=True)

    for key in ("q_qhd", "k_khd", "v_khd"):
        if not isinstance(obj.get(key), torch.Tensor):
            raise RuntimeError(f"raw capture missing {key}")

    raw_heads = obj.get("raw_head_index")
    mapping = (
        [int(x) for x in raw_heads.tolist()]
        if isinstance(raw_heads, torch.Tensor)
        else list(range(int(obj["q_qhd"].shape[1])))
    )
    heads = [int(x) for x in a.physical_heads.split(",") if x.strip()]
    ranks = [int(x) for x in a.ranks.split(",") if x.strip()]
    missing = [h for h in heads if h not in mapping]
    if missing:
        raise RuntimeError(f"raw capture lacks heads {missing}; mapping={mapping}")

    report = {
        "format": "apl-h3-htrc-recursive-centroid-local-value-dipole-v1",
        "raw": str(a.raw),
        "heads": heads,
        "centroids_per_head": int(a.centroids),
        "nodes_per_depth": int(a.nodes_per_depth),
        "ranks": ranks,
        "scope": (
            "exact algebra gate for recursive centroid-local value-dipole merge; "
            "plus internal-node Jacobian numerical-rank diagnostics"
        ),
        "cases": [],
    }

    all_flat_err = []
    by_depth_spectra = {}

    for ph in heads:
        si = mapping.index(ph)
        q = obj["q_qhd"][:, si].float().to(dev).contiguous()
        k = obj["k_khd"][:, si].to(dev).contiguous()[:, None, :]
        v = obj["v_khd"][:, si].to(dev).contiguous()[:, None, :]

        geom = htrc.prepare_leaf_geometry(
            k, v, leaf_work_chunk=int(a.leaf_work_chunk)
        )
        g = htrc.select_head_geometry(geom, 0)
        centers = htrc.sampled_query_centers(q, count=128, clusters=32)
        forest = htrc.build_root_kmeans_structural_forest(
            g["km"].detach().cpu(),
            roots=80,
            fanout=4,
            proj_dim=24,
            iters=8,
            seed=20260925,
            head_index=int(ph),
        )
        plan = htrc._level_plan(forest, dev)

        ccount = int(centers.shape[0])
        take_c = min(int(a.centroids), ccount)
        cids = torch.linspace(0, ccount - 1, take_c).round().long().tolist()

        for cid in cids:
            center = centers[int(cid)]
            summary = htrc.centroid_summary_batch(center[None], g)
            leaf_j = leaf_value_jacobian(center, g, summary)
            logz, grad, value, jac = recursive_merge(
                summary, leaf_j, forest, plan, dev
            )

            flat_checks = []
            depth_metrics = {}
            for depth in range(1, int(forest["max_depth"]) + 1):
                ids = forest["levels"].get(depth, [])
                picked = sample_ids(ids, int(a.nodes_per_depth))
                if not picked:
                    continue

                pj = jac.index_select(
                    0, torch.tensor(picked, device=dev, dtype=torch.long)
                )
                rm = rank_metrics(pj, ranks)
                depth_metrics[str(depth)] = rm
                by_depth_spectra.setdefault(str(depth), []).append(
                    {
                        "head": ph,
                        "centroid": int(cid),
                        "metrics": rm,
                    }
                )

                # Check a small, deterministic subset by flattening all leaves.
                for nid in picked[: min(4, len(picked))]:
                    dlz, dg, dv, dj = direct_from_leaves(
                        nid, summary, leaf_j, forest, dev
                    )
                    errs = {
                        "node": int(nid),
                        "depth": int(depth),
                        "logz_abs": float((logz[nid] - dlz).abs().item()),
                        "grad_rel_l2": rel_l2(grad[nid], dg),
                        "value_rel_l2": rel_l2(value[nid], dv),
                        "jac_rel_l2": rel_l2(jac[nid], dj),
                    }
                    maxerr = max(
                        errs["grad_rel_l2"],
                        errs["value_rel_l2"],
                        errs["jac_rel_l2"],
                    )
                    all_flat_err.append(maxerr)
                    flat_checks.append(errs)

            max_case_err = max(
                [
                    max(
                        x["grad_rel_l2"],
                        x["value_rel_l2"],
                        x["jac_rel_l2"],
                    )
                    for x in flat_checks
                ]
                or [0.0]
            )
            if max_case_err > 2.0e-5:
                raise RuntimeError(
                    f"recursive value-dipole identity failed head={ph} "
                    f"centroid={cid} max_rel={max_case_err:.3e}"
                )

            row = {
                "physical_head": ph,
                "centroid_index": int(cid),
                "centroid_norm": float(torch.linalg.vector_norm(center).item()),
                "max_recursive_vs_flat_rel_l2": max_case_err,
                "flat_checks": flat_checks,
                "rank_by_depth": depth_metrics,
            }
            report["cases"].append(row)
            print(
                f"head={ph} centroid={cid} "
                f"max_recursive_flat={max_case_err:.3e}",
                flush=True,
            )

            del leaf_j, logz, grad, value, jac, summary
            torch.cuda.empty_cache()

        del q, k, v, geom, g, centers, forest, plan
        torch.cuda.empty_cache()

    # Aggregate rank metrics from already-small JSON summaries.
    depth_summary = {}
    for depth, rows in by_depth_spectra.items():
        metrics = [x["metrics"] for x in rows if x["metrics"] is not None]
        if not metrics:
            continue
        d = {
            "cases": len(metrics),
            "rank99_mean": statistics.mean(m["rank99_mean"] for m in metrics),
            "rank99_max": max(m["rank99_max"] for m in metrics),
            "rank999_mean": statistics.mean(m["rank999_mean"] for m in metrics),
            "rank999_max": max(m["rank999_max"] for m in metrics),
            "approx_rel_frob": {},
        }
        for r in ranks:
            key = str(r)
            d["approx_rel_frob"][key] = {
                "mean_of_means": statistics.mean(
                    m["approx_rel_frob"][key]["mean"] for m in metrics
                ),
                "max": max(
                    m["approx_rel_frob"][key]["max"] for m in metrics
                ),
            }
        depth_summary[depth] = d

    report["summary"] = {
        "cases": len(report["cases"]),
        "max_recursive_vs_flat_rel_l2": max(all_flat_err or [0.0]),
        "rank_by_depth": depth_summary,
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nsummary:", json.dumps(report["summary"], indent=2), flush=True)
    print(f"JSON: {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
