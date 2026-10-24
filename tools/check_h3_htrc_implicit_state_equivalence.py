from __future__ import annotations

import argparse
import importlib.util
import json
import math
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


def clone_geometry(g):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in g.items()}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    return float(torch.linalg.vector_norm(af - bf) / torch.linalg.vector_norm(bf).clamp_min(1e-20))


def forest_depth_vector(forest):
    if "node_depth" in forest:
        return forest["node_depth"].to(torch.int16)
    n = int(forest.get("node_count", int(forest["child_count"].numel())))
    out = torch.empty(n, dtype=torch.int16)
    for depth, ids in forest["levels"].items():
        if ids:
            out[torch.tensor(ids, dtype=torch.long)] = int(depth)
    return out


def forest_equal(a, b):
    fields = [
        "root_ids",
        "leaf_node_for_block",
        "child_idx",
        "child_count",
        "subtree_end",
        "leaf_block",
    ]
    out = {name: bool(torch.equal(a[name], b[name])) for name in fields}
    out["node_depth"] = bool(torch.equal(
        forest_depth_vector(a), forest_depth_vector(b)
    ))
    out["max_depth"] = int(a["max_depth"]) == int(b["max_depth"])
    ac = int(a.get("node_count", int(a["child_count"].numel())))
    bc = int(b.get("node_count", int(b["child_count"].numel())))
    out["node_count"] = ac == bc
    return out


def evaluator_equivalence(htrc, kernels, q, s_exp, s_imp, fractions):
    # Evenly spread a bounded set of real queries across the current call.
    take = min(512, int(q.shape[0]))
    qidx = torch.linspace(
        0, int(q.shape[0]) - 1, take, device=q.device
    ).round().long()
    qq = q.index_select(0, qidx).contiguous()

    estate_exp = htrc.compact_state_for_eval(
        s_exp, vector_dtype=torch.bfloat16
    )
    estate_imp = htrc.compact_state_for_eval(
        s_imp, vector_dtype=torch.bfloat16
    )

    # State-build-only topology must not leak into the online evaluator ABI.
    forbidden = ("subtree_end", "leaf_block")
    leaked_exp = [x for x in forbidden if x in estate_exp]
    leaked_imp = [x for x in forbidden if x in estate_imp]
    if leaked_exp or leaked_imp:
        raise RuntimeError(
            f"legacy state-build topology leaked into estate: "
            f"explicit={leaked_exp} implicit={leaked_imp}"
        )

    assign_exp = htrc.assign_queries(qq, s_exp["centers"])
    assign_imp = htrc.assign_queries(qq, s_imp["centers"])
    if not torch.equal(assign_exp, assign_imp):
        raise RuntimeError("query assignment differs explicit vs implicit")

    # bucketed_attention's default root capacity is the frozen 2%-open
    # production policy (6.4*Q).  This equivalence gate deliberately sweeps
    # through 5% and 10%, so size one shared capacity for the largest tested
    # fraction instead of turning a diagnostic sweep into a false overflow.
    roots = int(estate_exp["root_ids"].numel())
    fanout = int(estate_exp["child_idx"].shape[1])
    max_frac = max(float(x) for x in fractions)
    needed = int(math.ceil(int(qq.shape[0]) * roots * max_frac * fanout * 1.25))
    root_capacity = 1 << max(0, needed - 1).bit_length()

    rows = []
    for frac in fractions:
        out_exp, info_exp = htrc.bucketed_attention(
            qq,
            assign_exp,
            estate_exp,
            kernels,
            open_fraction=float(frac),
            root_capacity_override=root_capacity,
        )
        out_imp, info_imp = htrc.bucketed_attention(
            qq,
            assign_imp,
            estate_imp,
            kernels,
            open_fraction=float(frac),
            root_capacity_override=root_capacity,
        )
        err = rel_l2(out_imp, out_exp)
        if err > 2e-6:
            raise RuntimeError(
                f"evaluator mismatch open={frac}: rel_l2={err:.3e}"
            )
        # Correctness/status fields must agree exactly. Timing/diagnostic
        # extras, if any, are intentionally ignored.
        for key in ("required", "overflow", "residual_frontier", "exact_tokens"):
            if info_exp.get(key) != info_imp.get(key):
                raise RuntimeError(
                    f"evaluator info mismatch key={key}: "
                    f"{info_exp.get(key)} != {info_imp.get(key)}"
                )
        rows.append({
            "open_fraction": float(frac),
            "output_rel_l2": err,
            "required": info_imp.get("required"),
            "overflow": info_imp.get("overflow"),
            "residual_frontier": info_imp.get("residual_frontier"),
        })
    return {
        "queries": int(qq.shape[0]),
        "root_capacity_override": int(root_capacity),
        "forbidden_online_fields_absent": True,
        "fractions": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--physical-heads", default="1,16,27,55")
    ap.add_argument("--centroid-chunk", type=int, default=4)
    ap.add_argument("--leaf-eigh-chunk", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    root = Path(__file__).resolve().parents[1]
    htrc = load_module("apl_h3_implicit_state_gate", root / "h3" / "htrc_live.py")
    kernels = load_module("apl_h3_implicit_state_kernels", root / "h3" / "rrc3_kernels.py")

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
    heads = [int(x) for x in a.physical_heads.split(",") if x.strip()]

    report = {
        "format": "apl-h3-htrc-implicit-state-equivalence-v1",
        "raw": str(a.raw),
        "heads": heads,
        "cases": [],
    }

    for ph in heads:
        if ph not in mapping:
            raise RuntimeError(f"raw capture lacks head {ph}; mapping={mapping}")
        si = mapping.index(ph)
        q = obj["q_qhd"][:, si].float().cuda().contiguous()
        k = obj["k_khd"][:, si].cuda().contiguous()[:, None, :]
        v = obj["v_khd"][:, si].cuda().contiguous()[:, None, :]

        geom_all = htrc.prepare_leaf_geometry(k, v, leaf_work_chunk=32)
        gh = htrc.select_head_geometry(geom_all, 0)

        common = dict(
            roots=80,
            fanout=4,
            query_clusters=32,
            query_sample=128,
            proj_dim=24,
            kmeans_iters=8,
            seed=20260925,
            head_index=int(ph),
            centroid_chunk=int(a.centroid_chunk),
            factor_mode="gram_cholesky",
            leaf_eigh_chunk_matrices=int(a.leaf_eigh_chunk),
            vector_dtype=torch.bfloat16,
        )

        p0 = {}
        t0 = time.perf_counter()
        s_exp = htrc.build_head_state(
            q, clone_geometry(gh),
            forest_mode="root_kmeans_structural_explicit",
            profile=p0,
            **common,
        )
        torch.cuda.synchronize()
        exp_wall = time.perf_counter() - t0

        p1 = {}
        t1 = time.perf_counter()
        s_imp = htrc.build_head_state(
            q, clone_geometry(gh),
            forest_mode="root_kmeans_structural",
            profile=p1,
            **common,
        )
        torch.cuda.synchronize()
        imp_wall = time.perf_counter() - t1

        feq = forest_equal(s_exp["forest"], s_imp["forest"])
        if not all(feq.values()):
            raise RuntimeError(f"forest mismatch head={ph}: {feq}")

        tensors = ["centers", "logz", "grad", "value", "lam", "U", "trace", "captured_trace"]
        errs = {name: rel_l2(s_imp[name], s_exp[name]) for name in tensors}
        max_err = max(errs.values())
        if max_err > 2e-5:
            raise RuntimeError(
                f"implicit state mismatch head={ph}: max_rel={max_err:.3e} errs={errs}"
            )

        eval_eq = evaluator_equivalence(
            htrc, kernels, q, s_exp, s_imp, (0.02, 0.05, 0.10)
        )

        row = {
            "physical_head": ph,
            "forest_equivalence": feq,
            "state_rel_l2": errs,
            "max_state_rel_l2": max_err,
            "explicit_profile": p0,
            "implicit_profile": p1,
            "explicit_wall_s": exp_wall,
            "implicit_wall_s": imp_wall,
            "evaluator_equivalence": eval_eq,
        }
        report["cases"].append(row)
        print(
            f"head={ph} max_state_rel={max_err:.3e} "
            f"forest_exp={p0.get('forest', float('nan')):.4f}s "
            f"forest_imp={p1.get('forest', float('nan')):.4f}s "
            f"wall_exp={exp_wall:.4f}s wall_imp={imp_wall:.4f}s",
            flush=True,
        )

        del s_exp, s_imp, gh, geom_all, q, k, v
        torch.cuda.empty_cache()

    report["summary"] = {
        "equivalent_all": all(
            all(c["forest_equivalence"].values()) and c["max_state_rel_l2"] <= 2e-5
            for c in report["cases"]
        ),
        "max_state_rel_l2": max(c["max_state_rel_l2"] for c in report["cases"]),
        "max_evaluator_rel_l2": max(
            x["output_rel_l2"]
            for c in report["cases"]
            for x in c["evaluator_equivalence"]["fractions"]
        ),
        "mean_explicit_forest_s": sum(c["explicit_profile"].get("forest", 0.0) for c in report["cases"]) / len(report["cases"]),
        "mean_implicit_forest_s": sum(c["implicit_profile"].get("forest", 0.0) for c in report["cases"]) / len(report["cases"]),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\\nsummary:", json.dumps(report["summary"], indent=2), flush=True)
    print(f"JSON: {a.out}", flush=True)


if __name__ == "__main__":
    main()
