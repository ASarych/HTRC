"""Kernel-level profile of the flat HTRC operator on real captures: measured roofline of THIS card,
achieved TFLOP/s and GB/s per stage, Triton compile info (registers, spills, shared memory), and the
exact Nsight Compute command for the tile kernel.  One JSON report in D:\\ComfyOutput\\H3\\math_probe.

  python tools\\profile_flat_kernel.py [--mode balanced]
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import time
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import importlib.util

spec = importlib.util.spec_from_file_location("sweep_mod", ROOT / "tools" / "sweep_h3_flat_kernels.py")
sw = importlib.util.module_from_spec(spec)
sys.modules["sweep_mod"] = sw
spec.loader.exec_module(sw)


def timeit(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def roofline(dev):
    out = {}
    for dt, name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32_tf32")):
        if name == "fp32_tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
        a = torch.randn(8192, 8192, device=dev, dtype=dt)
        b = torch.randn(8192, 8192, device=dev, dtype=dt)
        ms = timeit(lambda: a @ b, 8)
        out[f"matmul_{name}_TFLOPs"] = round(2 * 8192 ** 3 / ms / 1e9, 1)
        del a, b
    # skinny shapes similar to the tile kernel (64 x 128 @ 128 x N), batched
    q = torch.randn(640, 64, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(640, 128, 2560, device=dev, dtype=torch.bfloat16)
    ms = timeit(lambda: torch.bmm(q, k), 10)
    out["bmm_64x128x2560_TFLOPs"] = round(2 * 640 * 64 * 128 * 2560 / ms / 1e9, 1)
    x = torch.empty(1 << 28, device=dev, dtype=torch.uint8)
    y = torch.empty_like(x)
    ms = timeit(lambda: y.copy_(x), 10)
    out["copy_GBps"] = round(2 * x.numel() / ms / 1e6, 0)
    props = torch.cuda.get_device_properties(dev)
    out["sm_count"] = props.multi_processor_count
    out["l2_MiB"] = round(getattr(props, "L2_cache_size", 0) / (1 << 20), 1)
    return out


def kernel_info(mod):
    info = {}
    for name in ("_stats_kernel", "_rowmax_kernel", "_tile_kernel", "_merge_kernel"):
        k = getattr(mod, name, None)
        rows = []
        try:
            caches = getattr(k, "device_caches", None) or {}
            for dev_cache in caches.values():
                kc = dev_cache[0]
                for ck in kc.values():
                    meta = getattr(ck, "metadata", None)
                    rows.append({"n_regs": getattr(ck, "n_regs", None), "n_spills": getattr(ck, "n_spills", None),
                                 "shared_B": getattr(meta, "shared", None) if meta is not None else None,
                                 "num_warps": getattr(meta, "num_warps", None) if meta is not None else None})
        except Exception as exc:
            rows.append({"error": str(exc)[:100]})
        info[name] = rows[:6]
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--captures", default=r"D:\ComfyOutput\H3\math_probe\**\*.pt")
    ap.add_argument("--out", default=r"D:\ComfyOutput\H3\math_probe")
    a = ap.parse_args()
    dev = torch.device("cuda")
    flat = sw.load("prof_htrc_flat", "h3/htrc_flat.py")
    runner = sw.load("prof_htrc_flat_runner", "h3/htrc_flat_runner.py")
    cfg = dict(flat.PROD)
    cfg.update(runner.MODES[a.mode])
    cfg["kern"] = dict(cfg.get("kern", {}))
    cfg["tile_chunk"] = 1024
    cfg["sink_leaves"] = 0
    caps = sw.find_captures(a.captures)
    q, k, v, tail = sw.capture_head(caps[0][1], 1, dev)
    n = (int(k.shape[0]) // 16) * 16
    k, v = k[:n], v[:n]
    ws = {}
    res = {"gpu": torch.cuda.get_device_name(0), "roofline": roofline(dev)}
    print("roofline", res["roofline"])
    stages = sw.stage_ms(flat, cfg, (q, k, v, None), ws, 5)
    Q, D, leaf = int(q.shape[0]), int(q.shape[1]), cfg.get("leaf", 16)
    nb, E = n // leaf, cfg["exact_leaves"]
    far_flop = 4.0 * Q * nb * D
    near_flop = 4.0 * Q * E * leaf * D
    stats_flop = 4.0 * (Q // cfg["tile"]) * n * D          # per tile: first-order stats over all keys
    tile_ms = stages["tile"]
    res["stages_ms"] = {s: round(v_, 2) for s, v_ in stages.items()}
    res["flops_G"] = {"far": round(far_flop / 1e9, 1), "near": round(near_flop / 1e9, 1), "stats": round(stats_flop / 1e9, 1)}
    res["tile_TFLOPs"] = round((far_flop + near_flop) / tile_ms / 1e9, 1)
    res["stats_TFLOPs"] = round(stats_flop / stages["stats"] / 1e9, 1)
    peak = res["roofline"]["matmul_bf16_TFLOPs"]
    res["tile_pct_of_measured_bf16_matmul"] = round(100 * res["tile_TFLOPs"] / peak, 1)
    res["tile_pct_of_skinny_bmm"] = round(100 * res["tile_TFLOPs"] / res["roofline"]["bmm_64x128x2560_TFLOPs"], 1)
    # state traffic: g, vbar bf16 per (tile, leaf) written by stats, read by rowmax (g) and tile (g, vbar)
    ntile = Q // cfg["tile"]
    state_bytes = ntile * nb * D * 2 * 2
    res["state_MiB"] = round(state_bytes / (1 << 20), 1)
    res["state_traffic_ms_at_copy_bw"] = round(3 * state_bytes / (res["roofline"]["copy_GBps"] * 1e6), 2)
    mod = flat._load_triton() if hasattr(flat, "_load_triton") else None
    if mod is not None:
        res["kernel_info"] = kernel_info(mod)
    print(json.dumps({k_: v_ for k_, v_ in res.items() if k_ != "kernel_info"}, indent=1))
    print("kernel_info", json.dumps(res.get("kernel_info"), indent=1))
    ncu = shutil.which("ncu")
    res["ncu"] = ncu
    res["ncu_command"] = ('ncu --kernel-name regex:_tile_kernel --launch-skip 3 --launch-count 1 '
                          '--set full -o D:\\ComfyOutput\\H3\\math_probe\\flat_tile_ncu '
                          f'"{sys.executable}" tools\\profile_flat_kernel.py --mode {a.mode}')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = out / f"flat_kernel_profile_{time.strftime('%Y%m%d-%H%M%S')}.json"
    rep.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("report:", rep)
    print("Nsight Compute:", ncu or "ncu not on PATH (install Nsight Compute / use the CUDA toolkit)")


if __name__ == "__main__":
    main()
