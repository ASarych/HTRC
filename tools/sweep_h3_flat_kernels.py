"""Per-stage kernel config sweep of the flat HTRC operator at the live H3 shape.

Run on the GPU machine (needs the ComfyUI venv python: torch + triton):
  python tools\\sweep_h3_flat_kernels.py [--mode balanced] [--heads 3]

Shape = live runtime: 40759 queries, 37232 main keys + 3520 exact sink rows (220 forced leaves) +
7 tail keys, head_dim 128, tile 64, one whole-sequence chunk, bf16 leaf state, eager, workspace
reuse.  Stages (stats / select / tile) are tuned independently from the stage-synchronised profile
(coordinate descent, each knob swept with the others fixed at the current best).  One report file
(JSON) goes to D:\\ComfyOutput\\H3\\math_probe; the best knobs are printed as the line to paste into
$env:APL_H3_FLAT_KERN.  Drift = relative L2 difference of the tuned output against the default
config on the same head (numerics-only knobs should give ~1e-3 or less).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_inputs(n, m, sink_rows, tail_rows, d, dev):
    g = torch.Generator(device=dev).manual_seed(1234)
    basis = torch.randn(24, d, device=dev, generator=g)
    def mix(rows, noise):
        c = torch.randn(rows, 24, device=dev, generator=g) * 0.45
        return (c @ basis + noise * torch.randn(rows, d, device=dev, generator=g)).to(torch.bfloat16)
    q = mix(n, 0.6)
    k = mix(m + tail_rows, 0.6)
    v = torch.randn(m + tail_rows, d, device=dev, generator=g).to(torch.bfloat16)
    return q, k[:m], v[:m], (k[m:].float(), v[m:].float()) if tail_rows else None


def find_captures(pattern):
    import glob
    out = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        try:
            obj = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        except Exception:
            continue
        if isinstance(obj, dict) and all(k in obj for k in ("q_qhd", "k_khd", "v_khd")) \
                and int(obj["q_qhd"].shape[0]) >= 4096:
            out.append((f, obj))
    return out


def capture_head(obj, head, dev):
    mp = obj.get("raw_head_index")
    mp = [int(x) for x in mp.tolist()] if isinstance(mp, torch.Tensor) else list(range(obj["q_qhd"].shape[1]))
    if head not in mp:
        return None
    i = mp.index(head)
    q = obj["q_qhd"][:, i].to(dev).contiguous()
    k = obj["k_khd"][:, i].to(dev).contiguous()
    v = obj["v_khd"][:, i].to(dev).contiguous()
    keep = (int(k.shape[0]) // 16) * 16              # keys past the last full leaf are the tail
    tail = (k[keep:].float(), v[keep:].float()) if keep < int(k.shape[0]) else None
    return q, k[:keep], v[:keep], tail


def dense_ref(q, k, v, tail):
    kk = torch.cat([k, tail[0].to(k.dtype)], 0) if tail is not None else k
    vv = torch.cat([v, tail[1].to(v.dtype)], 0) if tail is not None else v
    return torch.nn.functional.scaled_dot_product_attention(
        q[None, None].float(), kk[None, None].float(), vv[None, None].float())[0, 0]


def run_head(flat, runner_cfg, q, k, v, tail, ws, prof=None):
    cfg = copy.deepcopy(runner_cfg)
    if prof is not None:
        cfg["prof"] = prof
    return flat.flat_attention(q, k, v, tail=tail, ws=ws, **cfg)


def stage_ms(flat, cfg, data, ws, iters):
    q, k, v, tail = data
    acc = {"stats": [], "select": [], "tile": []}
    for i in range(iters + 2):
        prof = {}
        torch.cuda.synchronize()
        run_head(flat, cfg, q, k, v, tail, ws, prof)
        if i >= 2:
            for s in acc:
                acc[s].append(prof[s])
    return {s: statistics.median(v) for s, v in acc.items()}


def head_ms(flat, cfg, data, ws, iters):
    q, k, v, tail = data
    for _ in range(3):
        run_head(flat, cfg, q, k, v, tail, ws)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        run_head(flat, cfg, q, k, v, tail, ws)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def structural_variants(flat, cfg, caps, a, dev):
    """Cost and REAL error (vs dense SDPA) of structural choices on saved full-Q captures
    (bear prompt, several blocks/calls x heads). cfg is the live config; sink forced leaves are
    off here (captures hold no conditioning-sink layout), as in the earlier 72-map scans."""
    base_exact = cfg["exact_leaves"]
    plan = [
        ("leaf16 max-select (live)", 16, {}),
        ("leaf16 exact /2", 16, {"exact_leaves": base_exact // 2}),
        ("leaf16 exact x2", 16, {"exact_leaves": base_exact * 2}),
        ("leaf32 same exact tokens", 32, {"exact_leaves": base_exact // 2}),
        ("leaf32 same exact leaves", 32, {}),
        ("leaf16 tile32", 16, {"tile": 32}),
        ("leaf16 tile128", 16, {"tile": 128}),
        ("leaf16 kmeans order", 16, {"order": "kmeans"}),
        ("leaf16 bound-select", 16, {"select": "bound"}),
        ("quant q int8 (per row)", 16, {"quant": {"q": "int8"}}),
        ("quant K/V int8 (per row)", 16, {"quant": {"kv": "int8"}}),
        ("quant state int8 (per row)", 16, {"quant": {"state": "int8"}}),
        ("quant q+K/V+state int8", 16, {"quant": {"q": "int8", "kv": "int8", "state": "int8"}}),
        ("quant q+K/V int8 group32", 16, {"quant": {"q": "int8", "kv": "int8", "group": 32}}),
        ("quant q+K/V fp8 (compare)", 16, {"quant": {"q": "fp8", "kv": "fp8"}}),
        ("REAL int8 QK (g,K,q int8; PV bf16)", 16, {"store": "int8"}),
        ("REAL int8 QK bn64", 16, {"store": "int8", "kern_extra": {"tile_bn": 64}}),
        ("bf16 baseline (old PROD: bn32)", 16, {"store": "bf16", "kern_extra": {"tile_bn": 32}}),
        ("REAL int8 QK bn64 stages2", 16, {"store": "int8", "kern_extra": {"tile_bn": 64, "tile_stages": 2}}),
        ("REAL int8 QK bn64 stages3", 16, {"store": "int8", "kern_extra": {"tile_bn": 64, "tile_stages": 3}}),
        ("REAL int8 QK bn128", 16, {"store": "int8", "kern_extra": {"tile_bn": 128}}),
        ("REAL int8 QK bn64 rb2", 16, {"store": "int8", "kern_extra": {"tile_bn": 64, "tile_rb": 2}}),
        ("REAL int8 QK stages2", 16, {"store": "int8", "kern_extra": {"tile_stages": 2}}),
        ("REAL int8 QK warps8", 16, {"store": "int8", "kern_extra": {"tile_warps": 8}}),
        ("state fp8 (live cfg)", 16, {"store": "fp8"}),
        ("state fp8 group2", 16, {"store": "fp8", "kern_group": 2}),
        ("state fp8 E96 leaf32 group2", 32, {"store": "fp8", "kern_group": 2, "exact_leaves": 96}),
        ("state bf16 E96 leaf32 group2", 32, {"kern_group": 2, "exact_leaves": 96}),
        ("group2 (centre per 128 rows)", 16, {"kern_group": 2}),
        ("group4 (centre per 256 rows)", 16, {"kern_group": 4}),
        ("group2 exact x1.5", 16, {"kern_group": 2, "exact_leaves": int(base_exact * 1.5)}),
        ("group4 exact x2", 16, {"kern_group": 4, "exact_leaves": base_exact * 2}),
    ]
    if a.curve:          # error/time budget curve: exact leaves x (plain, group2, leaf32)
        plan = []
        for ex in (16, 32, 64, 128, 192, 256, 384, 512, 768):
            plan.append((f"E{ex} leaf16", 16, {"exact_leaves": ex}))
            plan.append((f"E{ex} leaf16 group2", 16, {"exact_leaves": ex, "kern_group": 2}))
            plan.append((f"E{ex // 2} leaf32 group2", 32, {"exact_leaves": max(8, ex // 2), "kern_group": 2}))
    maps = []
    for f, obj in caps:
        for h in (1, 16, 27, 55):
            maps.append((f, obj, h))
    maps = maps[: a.max_maps]
    rows = []
    for name, leaf, over in plan:
        c = copy.deepcopy(cfg)
        over = dict(over)
        gsz = over.pop("kern_group", None)
        kex = over.pop("kern_extra", None)
        c.update(over)
        if kex:
            c["kern"] = {**c.get("kern", {}), **kex}
        if gsz:
            c["kern"] = {**c.get("kern", {}), "group": gsz}
        c["leaf"] = leaf
        c["sink_leaves"] = 0
        errs, mss = [], []
        stg = None
        failed = None
        for f, obj, h in maps:
            d = capture_head(obj, h, dev)
            if d is None:
                continue
            q, k, v, tail = d
            main = (int(k.shape[0]) // leaf) * leaf
            t = tail
            if main < int(k.shape[0]):
                extra = (k[main:].float(), v[main:].float())
                t = (torch.cat([extra[0], tail[0]], 0), torch.cat([extra[1], tail[1]], 0)) if tail is not None else extra
            ref = dense_ref(q, k, v, tail)
            ws = {}
            try:
                out = flat.flat_attention(q, k[:main], v[:main], tail=t, ws=ws, **c)
                errs.append(float((out - ref).norm() / ref.norm()) * 100)
                if len(mss) < 2:
                    mss.append(head_ms(flat, c, (q, k[:main], v[:main], t), ws, 5))
                    if stg is None:
                        stg = {s_: round(x, 2) for s_, x in stage_ms(flat, c, (q, k[:main], v[:main], t), ws, 3).items()}
            except Exception as exc:
                failed = str(exc)[:120]
                break
            del q, k, v, ref
        if failed or not errs:
            rows.append({"variant": name, "error": failed or "no maps"})
            print(f"  variant {name}: failed ({failed})")
            continue
        errs.sort()
        row = {"variant": name, "maps": len(errs), "head_ms": round(statistics.median(mss), 2),
               "err_median_pct": round(errs[len(errs) // 2], 2),
               "err_p90_pct": round(errs[int(len(errs) * 0.9)], 2), "err_max_pct": round(errs[-1], 2),
               "stage_ms": stg}
        rows.append(row)
        print(f"  {name:<28} {row['head_ms']:6.2f} ms  err median/p90/max "
              f"{row['err_median_pct']:.1f}/{row['err_p90_pct']:.1f}/{row['err_max_pct']:.1f} %  ({row['maps']} maps)  stages {stg}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--queries", type=int, default=40759)
    ap.add_argument("--main-keys", type=int, default=40752 - 3520 - 0)
    ap.add_argument("--sink-rows", type=int, default=3520)
    ap.add_argument("--tail", type=int, default=7)
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    ap.add_argument("--no-variants", action="store_true", help="skip the structural variant table")
    ap.add_argument("--captures", default=r"D:\ComfyOutput\H3\math_probe\**\*.pt",
                    help="glob of saved full-Q captures (q_qhd/k_khd/v_khd); real data for timing and error")
    ap.add_argument("--max-maps", type=int, default=24)
    ap.add_argument("--curve", action="store_true", help="error/time budget curve over exact leaves instead of the variant list")
    a = ap.parse_args()
    dev = torch.device("cuda")
    flat = load("sweep_htrc_flat", "h3/htrc_flat.py")
    runner = load("sweep_htrc_flat_runner", "h3/htrc_flat_runner.py")
    cfg = dict(flat.PROD)
    cfg.update(runner.MODES[a.mode])
    cfg["kern"] = dict(cfg.get("kern", {}))
    cfg["tile_chunk"] = 1024
    cfg["sink_leaves"] = a.sink_rows // 16
    m = a.main_keys + a.sink_rows
    m = (m // 16) * 16
    caps = find_captures(a.captures)
    print(f"full-Q captures found: {len(caps)} ({a.captures})")
    data = None
    if caps:
        d = capture_head(caps[0][1], 1, dev)
        if d is not None:
            data = d
            m = int(d[1].shape[0])
            a.queries = int(d[0].shape[0])
            a.tail = 0 if d[3] is None else int(d[3][0].shape[0])
            cfg["sink_leaves"] = 0
            print(f"using real capture {Path(caps[0][0]).name}: q={a.queries} k={m} tail={a.tail}")
    if data is None:
        print("WARNING: no captures, falling back to synthetic data (timing only, errors meaningless)")
        data = make_inputs(a.queries, m, a.sink_rows, a.tail, 128, dev)
    ws = {}
    base_cfg = copy.deepcopy(cfg)
    ref = run_head(flat, base_cfg, *data[:3], data[3], ws).clone()
    base_stage = stage_ms(flat, base_cfg, data, ws, a.iters)
    base_head = head_ms(flat, base_cfg, data, ws, 10)
    print("default  stages", {k: round(v, 2) for k, v in base_stage.items()}, "head", round(base_head, 2))

    grid = {
        "stats": [("stats_bt", [8, 16, 32]), ("stats_bl", [2, 4, 8]), ("stats_warps", [2, 4, 8])],
        "select": [("sel_bn", [16, 32, 64]), ("sel_warps", [2, 4, 8]), ("splits", [4, 8, 16, 32])],
        "tile": [("tile_rb", [1, 2, 4]), ("tile_bn", [16, 32, 64]), ("tile_warps", [2, 4, 8]), ("tile_stages", [1, 2, 3])],
    }
    log = []
    best_cfg = copy.deepcopy(cfg)
    best_stage = dict(base_stage)
    for stage, knobs in grid.items():
        for name, values in knobs:
            for val in values:
                trial = copy.deepcopy(best_cfg)
                if name == "splits":
                    trial["splits"] = val
                else:
                    trial["kern"][name] = val
                try:
                    ms = stage_ms(flat, trial, data, ws, a.iters)
                except Exception as exc:                       # invalid tile shape / out of resources
                    log.append({"stage": stage, "knob": name, "value": val, "error": str(exc)[:120]})
                    continue
                log.append({"stage": stage, "knob": name, "value": val, "ms": round(ms[stage], 3),
                            "all": {k: round(v, 2) for k, v in ms.items()}})
                # splits affects select and tile: judge by select+tile
                key = (lambda r: r["select"] + r["tile"]) if name == "splits" else (lambda r: r[stage])
                if key(ms) < key(best_stage) * 0.97:           # >3% win only (noise guard)
                    best_cfg, best_stage = trial, ms
                    print(f"  {stage}: {name}={val} -> {ms[stage]:.2f} ms")
    variants = [] if a.no_variants else structural_variants(flat, best_cfg, caps, a, dev) if caps else []
    tuned = run_head(flat, best_cfg, *data[:3], data[3], ws).clone()
    drift = float((tuned - ref).norm() / ref.norm().clamp_min(1e-9))
    tuned_head = head_ms(flat, best_cfg, data, ws, 10)
    print("tuned    stages", {k: round(v, 2) for k, v in best_stage.items()}, "head", round(tuned_head, 2),
          "drift", f"{drift:.2e}")
    env = json.dumps({"splits": best_cfg["splits"], **best_cfg["kern"]}, separators=(",", ":"))
    print("\n$env:APL_H3_FLAT_KERN='" + env + "'")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = out / f"flat_kernel_sweep_{time.strftime('%Y%m%d-%H%M%S')}.json"
    rep.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0), "mode": a.mode, "shape": {"q": a.queries, "k": m, "tail": a.tail},
        "default": {"stage_ms": base_stage, "head_ms": base_head, "kern": base_cfg["kern"], "splits": base_cfg["splits"]},
        "tuned": {"stage_ms": best_stage, "head_ms": tuned_head, "kern": best_cfg["kern"], "splits": best_cfg["splits"]},
        "captures": len(caps), "drift_rel_l2": drift, "env": env, "trials": log, "variants": variants}, indent=1), encoding="utf-8")
    print("report:", rep)


if __name__ == "__main__":
    main()
