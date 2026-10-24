"""Error STRUCTURE of Sol (tau sweep) vs flat HTRC (exact-leaf / group sweep) on saved full-Q captures,
from ~5% up to ~50% relL2.  Same maps as the other sweeps; keys cut to a multiple of 32, no sink.

Per config (median over maps unless noted):
  ms       time per head (median of 3 maps, single head, bf16)
  err      relL2 vs dense SDPA, %  (median / p90 over maps)
  rows>30  % of query rows whose own relative error exceeds 30 %  (heavy tail = visible local damage)
  rows>50  same for 50 %
  row p99  99th percentile of per-row relative error, %
  bias     share of error energy that is the token-mean (systematic drift) vs per-token noise, %
  corr1    correlation of the error vectors of neighbouring tokens (smooth -> blur, ~0 -> grain)
  gain     |out| / |ref| (<1: attention mass/energy lost, >1: amplified)
  cos      cosine(out, ref) over all rows
  worst h  head index with the largest median error
Run (ComfyUI venv python, GPU):  python tools\\compare_h3_error_structure.py
One JSON report goes to D:\\ComfyOutput\\H3\\math_probe.
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


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw = _load("sweep_mod", "tools/sweep_h3_flat_kernels.py")


def metrics(out, ref):
    out, ref = out.float(), ref.float()
    e = out - ref
    rn = ref.norm(dim=1).clamp_min(1e-9)
    row = e.norm(dim=1) / rn
    tot = float(e.norm() / ref.norm()) * 100
    mean_e = e.mean(0, keepdim=True)
    bias = float(mean_e.pow(2).sum() * e.shape[0] / e.pow(2).sum().clamp_min(1e-12)) * 100
    en = e / e.norm(dim=1, keepdim=True).clamp_min(1e-9)
    corr1 = float((en[1:] * en[:-1]).sum(1).mean())
    return {
        "err": tot,
        "rows30": float((row > 0.3).float().mean()) * 100,
        "rows50": float((row > 0.5).float().mean()) * 100,
        "rowp99": float(torch.quantile(row[:: max(1, row.numel() // 200000)], 0.99)) * 100,
        "bias": bias,
        "corr1": corr1,
        "gain": float(out.norm() / ref.norm()),
        "cos": float((out * ref).sum() / (out.norm() * ref.norm())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--captures", default=r"D:\ComfyOutput\H3\math_probe\**\*.pt")
    ap.add_argument("--max-maps", type=int, default=24)
    ap.add_argument("--taus", default="0.7,1.0,1.3,2.0,3.0,4.0,6.0,10.0")
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    a = ap.parse_args()
    import comfy_kitchen as ck
    dev = torch.device("cuda")
    flat = _load("cmp_htrc_flat", "h3/htrc_flat.py")
    runner = _load("cmp_htrc_flat_runner", "h3/htrc_flat_runner.py")
    base = dict(flat.PROD)
    base.update(runner.MODES[a.mode])
    base["kern"] = dict(base.get("kern", {}))
    base["tile_chunk"] = 1024
    base["sink_leaves"] = 0
    caps = sw.find_captures(a.captures)
    maps = [(f, o, h) for f, o in caps for h in (1, 16, 27, 55)][: a.max_maps]

    configs = []                                           # (name, kind, params)
    for tau in [float(x) for x in a.taus.split(",")]:
        configs.append((f"sol tau {tau:g}", "sol", {"tau": tau}))
    # flat: (exact leaves, leaf, group)
    for ex, leaf, gr in ((384, 16, 1), (192, 16, 1), (96, 16, 1), (32, 16, 1), (16, 16, 1),
                         (48, 32, 2), (16, 32, 2), (8, 32, 2), (8, 32, 4), (4, 32, 4), (2, 32, 4)):
        configs.append((f"flat E{ex} leaf{leaf} g{gr}", "flat", {"exact_leaves": ex, "leaf": leaf, "group": gr}))

    rows = []
    for name, kind, p in configs:
        per, per_head, mss = [], {}, []
        failed = None
        for f, obj, h in maps:
            d = sw.capture_head(obj, h, dev)
            if d is None:
                continue
            q, k, v, _t = d
            n = (min(q.shape[0], k.shape[0]) // 32) * 32
            q, k, v = q[:n].contiguous(), k[:n].contiguous(), v[:n].contiguous()
            ref = torch.nn.functional.scaled_dot_product_attention(
                q[None, None].float(), k[None, None].float(), v[None, None].float())[0, 0]
            try:
                if kind == "sol":
                    out = ck.sol_attn(q[None, :, None], k[None, :, None], v[None, :, None], tau=p["tau"])[0, :, 0]
                else:
                    c = copy.deepcopy(base)
                    c["exact_leaves"] = p["exact_leaves"]
                    c["leaf"] = p["leaf"]
                    if p["group"] > 1:
                        c["kern"]["group"] = p["group"]
                    out = flat.flat_attention(q, k, v, tail=None, ws={}, **c)
            except Exception as exc:
                failed = str(exc)[:120]
                break
            if len(mss) < 3:                                   # ms per head, first 3 maps, 3 timed runs
                if kind == "sol":
                    fn = lambda: ck.sol_attn(q[None, :, None], k[None, :, None], v[None, :, None], tau=p["tau"])
                else:
                    ws_t = {}
                    fn = lambda: flat.flat_attention(q, k, v, tail=None, ws=ws_t, **c)
                fn()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(3):
                    fn()
                e1.record()
                torch.cuda.synchronize()
                mss.append(e0.elapsed_time(e1) / 3)
            m = metrics(out, ref)
            per.append(m)
            per_head.setdefault(h, []).append(m["err"])
            del q, k, v, ref, out
        if failed or not per:
            rows.append({"config": name, "error": failed or "no maps"})
            print(f"{name}: failed ({failed})")
            continue
        med = lambda key: statistics.median(x[key] for x in per)
        errs = sorted(x["err"] for x in per)
        worst = max(per_head, key=lambda h: statistics.median(per_head[h]))
        row = {"config": name, "maps": len(per), "ms": round(statistics.median(mss), 2),
               "err_median": round(med("err"), 1), "err_p90": round(errs[int(len(errs) * 0.9)], 1),
               "rows30": round(med("rows30"), 1), "rows50": round(med("rows50"), 1),
               "rowp99": round(med("rowp99"), 0), "bias": round(med("bias"), 1),
               "corr1": round(med("corr1"), 2), "gain": round(med("gain"), 3), "cos": round(med("cos"), 3),
               "worst_head": worst, "worst_head_err": round(statistics.median(per_head[worst]), 1)}
        rows.append(row)
        print(f"{name:<22} {row['ms']:5.2f} ms  err {row['err_median']:5.1f}/{row['err_p90']:5.1f}  rows>30 {row['rows30']:5.1f}%  "
              f">50 {row['rows50']:5.1f}%  p99 {row['rowp99']:4.0f}%  bias {row['bias']:4.1f}%  "
              f"corr1 {row['corr1']:5.2f}  gain {row['gain']:.3f}  cos {row['cos']:.3f}  worst h{worst} {row['worst_head_err']}%")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = out / f"error_structure_{time.strftime('%Y%m%d-%H%M%S')}.json"
    rep.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "mode": a.mode, "rows": rows}, indent=1), encoding="utf-8")
    print("report:", rep)


if __name__ == "__main__":
    main()
