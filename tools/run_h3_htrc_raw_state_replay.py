from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules while the class
    # decorator executes, so register the synthetic module before exec_module.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def load_raw(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu", mmap=True)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected dict")
    for key in ("q_qhd", "k_khd", "v_khd"):
        if not isinstance(obj.get(key), torch.Tensor):
            raise RuntimeError(f"{path}: missing {key}")
    return obj


def parse_heads(raw: str, count: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(count))
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("--heads selected no heads")
    for h in out:
        if h < 0 or h >= count:
            raise ValueError(f"saved head {h} outside [0,{count})")
    return out


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Replay current HTRC state-build directly from the historical raw "
            "Q/K/V math snapshot. No ComfyUI/model startup and no traversal."
        )
    )
    p.add_argument("raw", type=Path)
    p.add_argument("--heads", default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--centroid-chunk", type=int, default=4)
    p.add_argument("--roots", type=int, default=80)
    p.add_argument("--fanout", type=int, default=4)
    p.add_argument("--query-clusters", type=int, default=32)
    p.add_argument("--query-sample", type=int, default=128)
    p.add_argument("--proj-dim", type=int, default=24)
    p.add_argument("--kmeans-iters", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument(
        "--forest-mode",
        choices=("kmeans", "root_kmeans_structural"),
        default="root_kmeans_structural",
    )
    p.add_argument(
        "--factor-mode",
        choices=("qr", "gram_cholesky"),
        default="gram_cholesky",
    )
    p.add_argument("--leaf-eigh-chunk", type=int, default=4096)
    p.add_argument("--leaf-work-chunk", type=int, default=32)
    p.add_argument(
        "--validate-depth1-basis",
        action="store_true",
        help=(
            "After timed state-build, compare the K-structural depth-1 basis "
            "against the historical centroid-0-derived basis."
        ),
    )
    p.add_argument(
        "--htrc-blob",
        default="",
        help=(
            "Optional git blob/commit:path source for h3/htrc_live.py. "
            "A plain blob SHA is sufficient for the recorded baseline."
        ),
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    module_path = root / "h3" / "htrc_live.py"
    temp_path = None
    if args.htrc_blob:
        payload = subprocess.check_output(
            ["git", "show", str(args.htrc_blob)],
            cwd=root,
        )
        tmp = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix="_htrc_live.py",
            delete=False,
        )
        try:
            tmp.write(payload)
            tmp.flush()
            temp_path = Path(tmp.name)
        finally:
            tmp.close()
        module_path = temp_path

    try:
        htrc = load_module(
            "apl_h3_htrc_research_replay",
            module_path,
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass

    obj = load_raw(args.raw)
    q_all = obj["q_qhd"]
    k_all = obj["k_khd"]
    v_all = obj["v_khd"]
    if q_all.ndim != 3 or k_all.ndim != 3 or v_all.ndim != 3:
        raise ValueError("raw q_qhd/k_khd/v_khd must be rank-3 [T,H,D]")
    if k_all.shape != v_all.shape:
        raise ValueError("raw K/V shapes differ")
    if q_all.shape[1] != k_all.shape[1]:
        raise ValueError("raw Q/K saved-head counts differ")
    if q_all.shape[2] != k_all.shape[2]:
        raise ValueError("raw Q/K head dims differ")

    saved_heads = int(q_all.shape[1])
    head_ids = parse_heads(args.heads, saved_heads)
    raw_head_index = obj.get("raw_head_index")
    device = torch.device(args.device)

    result: dict[str, Any] = {
        "format": "apl-h3-htrc-raw-state-replay-v1",
        "source": str(args.raw),
        "q_rows": int(q_all.shape[0]),
        "kv_rows": int(k_all.shape[0]),
        "saved_heads": saved_heads,
        "head_dim": int(q_all.shape[2]),
        "centroid_chunk": int(args.centroid_chunk),
        "forest_mode": args.forest_mode,
        "factor_mode": args.factor_mode,
        "leaf_eigh_chunk": int(args.leaf_eigh_chunk),
        "htrc_source": (
            str(args.htrc_blob) if args.htrc_blob else "working-tree"
        ),
        "heads": [],
    }

    for saved_head in head_ids:
        physical = (
            int(raw_head_index[saved_head])
            if isinstance(raw_head_index, torch.Tensor)
            else int(saved_head)
        )
        print(
            f"head saved={saved_head} physical={physical}: geometry",
            flush=True,
        )
        qh = q_all[:, saved_head].float().to(device).contiguous()
        kh = k_all[:, saved_head].to(device).contiguous()[:, None, :]
        vh = v_all[:, saved_head].to(device).contiguous()[:, None, :]

        sync(device)
        t0 = time.perf_counter()
        geom = htrc.prepare_leaf_geometry(
            kh,
            vh,
            leaf_work_chunk=int(args.leaf_work_chunk),
        )
        gh = htrc.select_head_geometry(geom, 0)
        sync(device)
        geometry_s = time.perf_counter() - t0

        before = htrc.numeric_telemetry_snapshot()
        stage_profile = {}
        sync(device)
        t1 = time.perf_counter()
        state = htrc.build_head_state(
            qh,
            gh,
            roots=int(args.roots),
            fanout=int(args.fanout),
            query_clusters=int(args.query_clusters),
            query_sample=int(args.query_sample),
            proj_dim=int(args.proj_dim),
            kmeans_iters=int(args.kmeans_iters),
            seed=int(args.seed),
            head_index=int(physical),
            centroid_chunk=int(args.centroid_chunk),
            forest_mode=args.forest_mode,
            factor_mode=args.factor_mode,
            leaf_eigh_chunk_matrices=int(args.leaf_eigh_chunk),
            profile=stage_profile,
        )
        sync(device)
        state_s = time.perf_counter() - t1
        numeric = htrc.numeric_telemetry_delta(before)

        depth1_basis_validation = None
        if args.validate_depth1_basis:
            # Validate the grouped QR-free structural factors against the
            # historical physical depth-1 Hessian.  This runs after timing.
            plan_v = htrc._level_plan(state["forest"], device)
            cc0 = state["centers"][:1]
            summary0 = htrc.centroid_summary_batch(cc0, gh)
            logz0, grad0, _value0, _plan0, weights0 = (
                htrc.node_base_and_value(
                    summary0,
                    state["forest"],
                    device,
                    plan=plan_v,
                    return_level_weights=True,
                )
            )
            leaf_qr0 = htrc.prepare_leaf_curvature_basis(gh)
            le0, lm0, _lt0, _lc0 = htrc.leaf_curvature(
                cc0,
                gh,
                summary0,
                leaf_qr=leaf_qr0,
                eigh_chunk_matrices=int(args.leaf_eigh_chunk),
            )
            factors = htrc.prepare_depth1_structural_basis(
                gh, state["forest"], plan_v
            )
            tids1, cidx1, _mask1 = plan_v[1]
            w1 = weights0[1]
            dev1 = grad0[:, cidx1] - grad0[:, tids1, None, :]

            rel_values = []
            abs_values = []
            ortho_values = []
            dims = []
            child_counts = []
            for grp in factors:
                rows = grp["rows"]
                Fg = int(grp["child_count"])
                Mg = int(rows.numel())
                Kg = int(grp["Q"].shape[-1])
                wg = w1.index_select(1, rows)[:, :, :Fg]
                devg = dev1.index_select(1, rows)[:, :, :Fg, :]
                child_u = lm0[:, grp["child_blocks"]]
                child_l = le0[:, grp["child_blocks"]]
                D0 = int(grad0.shape[-1])
                R0 = int(child_l.shape[-1])

                ym = child_u.permute(0, 1, 3, 2, 4).reshape(
                    1, Mg, D0, Fg * R0
                )
                ya = (wg[..., None] * child_l).reshape(
                    1, Mg, Fg * R0
                )
                yd = devg.permute(0, 1, 3, 2)
                y = torch.cat([ym, yd], dim=-1)
                coeff = torch.cat([ya, wg], dim=-1)
                h_phys = (
                    y * coeff.unsqueeze(-2)
                ) @ y.transpose(-1, -2)

                qg = grp["Q"]
                eye = torch.eye(
                    Kg, device=qg.device, dtype=qg.dtype
                )[None]
                ortho_err = (
                    qg.transpose(-1, -2) @ qg - eye
                )
                ortho_max = float(ortho_err.abs().amax().item())

                qn = qg[None]
                h_small = torch.matmul(
                    qn.transpose(-1, -2),
                    torch.matmul(h_phys, qn),
                )
                h_recon = torch.matmul(
                    qn, torch.matmul(h_small, qn.transpose(-1, -2))
                )
                h_res = h_phys - h_recon
                h_fro = torch.linalg.matrix_norm(
                    h_phys, ord="fro", dim=(-2, -1)
                ).clamp_min(1.0e-30)
                res_fro = torch.linalg.matrix_norm(
                    h_res, ord="fro", dim=(-2, -1)
                )
                rel_values.append(float((res_fro / h_fro).amax().item()))
                abs_values.append(float(h_res.abs().amax().item()))
                ortho_values.append(ortho_max)
                dims.append(Kg)
                child_counts.append(Fg)

            depth1_basis_validation = {
                "structural_dims": dims,
                "child_counts": child_counts,
                "hessian_residual_max_abs": max(abs_values),
                "hessian_residual_rel_fro_max": max(rel_values),
                "basis_orthogonality_max_abs": max(ortho_values),
            }

        row = {
            "saved_head": int(saved_head),
            "physical_head": int(physical),
            "leaves": int(geom["km"].shape[0]),
            "nodes": int(state["logz"].shape[1]),
            "centroids": int(state["centers"].shape[0]),
            "geometry_seconds": float(geometry_s),
            "state_seconds": float(state_s),
            "numeric": numeric,
            "stage_profile": stage_profile,
            "depth1_basis_validation": depth1_basis_validation,
        }
        result["heads"].append(row)
        print(
            f"head {physical}: leaves={row['leaves']} nodes={row['nodes']} "
            f"geometry={geometry_s:.3f}s state={state_s:.3f}s "
            f"leaf_eigh={numeric.get('leaf_eigh_matrices', 0)} "
            f"chol={numeric.get('chol_matrices', 0)} "
            f"leaf_stage={stage_profile.get('leaf_curvature', 0.0):.3f}s "
            f"hier={stage_profile.get('hierarchy', 0.0):.3f}s",
            flush=True,
        )
        if depth1_basis_validation is not None:
            print(
                "  depth1-basis: "
                f"children={depth1_basis_validation['child_counts']} "
                f"dims={depth1_basis_validation['structural_dims']} "
                f"H_rel_fro_max="
                f"{depth1_basis_validation['hessian_residual_rel_fro_max']:.3e} "
                f"H_abs_max="
                f"{depth1_basis_validation['hessian_residual_max_abs']:.3e} "
                f"Q_orth_max="
                f"{depth1_basis_validation['basis_orthogonality_max_abs']:.3e}",
                flush=True,
            )

        del state, gh, geom, qh, kh, vh
        if device.type == "cuda":
            torch.cuda.empty_cache()

    state_times = [x["state_seconds"] for x in result["heads"]]
    result["aggregate"] = {
        "heads": len(result["heads"]),
        "state_seconds_sum": float(sum(state_times)),
        "state_seconds_mean": (
            float(sum(state_times) / len(state_times)) if state_times else 0.0
        ),
        "state_seconds_max": max(state_times) if state_times else 0.0,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
