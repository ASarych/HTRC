"""Clean-baseline INT8 query-cache A/B for the RTX 5070 Ti H3 Flat Tile.

The original Python-range tail-in-merge R32 is pinned at e47cc7a.
The current experimental copy is compiled with cache none / dq / both.
Do not compare new alternatives solely with each other; pinned
old-range and 571a7a exact original are separate controls.

The rowmax sp=0 program stores the quantized query only once per tile:
  dq:   dq8 int8 + sdq float32    (~2.06 MiB per 256-tile chunk)
  both: above plus q8 + sq      (~4.13 MiB per 256-tile chunk).
Tile reads those instead of re-quantizing in every split CTA.

Uses saved native full-Q H3 head inputs; no video denoising.
PowerShell:
  & $py -B tools\benchmark_h3_flat_query_cache.py --heads 1,16 --modes fast,balanced
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
PINNED = "e47cc7a310936547ab6c7d97bd07526eb6a135fe"
SPEC = importlib.util.spec_from_file_location(
    "_query_cache_test_helpers", ROOT / "tools" / "benchmark_h3_flat_tail_merge.py"
)
helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helper
SPEC.loader.exec_module(helper)

exact = helper.exact
sw = exact.helpers.sw


def pinned_modules(path):
    names = ("htrc_flat.py", "htrc_flat_triton.py", "htrc_flat_runner.py")
    for filename in names:
        src = exact.read_git_blob(PINNED, f"h3/{filename}")
        if filename == "htrc_flat.py":
            marker = 'name = "_apl_htrc_flat_triton"'
            if src.count(marker) != 1:
                raise RuntimeError("Pinned source loader structure changed")
            src = src.replace(marker, 'name = "_apl_htrc_flat_triton_qcache_pin"')
        (path / filename).write_text(src, encoding="utf-8")
    return (
        exact.load_module("_qcache_pinned_flat", path / "htrc_flat.py"),
        exact.load_module("_qcache_pinned_runner", path / "htrc_flat_runner.py"),
    )


def batched_event_ms(fn, reps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / reps


def compare(data, head, mode, original, original_runner,
            pinned, pinned_runner, new, new_runner,
            chunk, warmups, rounds, batch, tolerance):
    q, k, v, tail = data
    if tail is None or not 0 < int(tail[0].shape[0]) <= 16:
        raise ValueError("Saved H3 native capture must contain 1..16 tail keys")
    original_cfg = exact.configuration(original, original_runner, mode, chunk)
    pinned_cfg = exact.configuration(pinned, pinned_runner, mode, chunk)
    new_cfg = exact.configuration(new, new_runner, mode, chunk)
    if original_cfg != pinned_cfg or pinned_cfg != new_cfg:
        raise RuntimeError("Original/pinned/new PROD and mode configs differ")

    cfg, modules, spaces = {}, {}, {}
    for name in ("pinned_R32", "new_none", "cache_dq", "cache_both"):
        raw = pinned_cfg if name == "pinned_R32" else new_cfg
        c = copy.deepcopy(raw)
        c["kern"].update(
            tile_tail_merge=True,
            tile_tail_merge_rows=32,
            tile_passes="fused",
            tile_far_prefetch_lz=False,
            tile_pv_dot_acc="none",
            tile_late_near_q=False,
        )
        if name != "pinned_R32":
            c["kern"]["tile_select_query_cache"] = {
                "new_none": "none",
                "cache_dq": "dq",
                "cache_both": "both",
            }[name]
        cfg[name] = c
        modules[name] = pinned if name == "pinned_R32" else new
        spaces[name] = {}
    orig_ws = {}

    def run(name):
        return modules[name].flat_attention(
            q, k, v, tail=tail, ws=spaces[name], **cfg[name]
        )

    def reference():
        return original.flat_attention(
            q, k, v, tail=tail, ws=orig_ws, **original_cfg
        )

    compiled, failures = {}, {}
    for name in cfg:
        try:
            with helper.record_kernel(modules[name], "_tile_kernel") as tile_meta:
                with helper.record_kernel(modules[name], "_rowmax_kernel") as sel_meta:
                    run(name)
                    torch.cuda.synchronize()
            compiled[name] = {"tile":list(tile_meta), "select":list(sel_meta)}
        except Exception as exc:
            if name in ("pinned_R32","new_none"):
                raise RuntimeError(f"Control {name} compilation failed") from exc
            failures[name] = {
                "type":type(exc).__name__,
                "error":str(exc)[:2000],
                "traceback_tail":traceback.format_exc()[-2500:],
            }

    baseline = reference().clone()
    pinned_out = run("pinned_R32").clone()
    denom_orig = baseline.float().norm().clamp_min(1.0e-20)
    denom_pin = pinned_out.float().norm().clamp_min(1.0e-20)
    parity = {}
    for name in cfg:
        if name in failures:
            continue
        try:
            val = run(name).clone()
            d_orig = val.float() - baseline.float()
            d_pin = val.float() - pinned_out.float()
            rel_orig = float(d_orig.norm() / denom_orig)
            rel_pin = float(d_pin.norm() / denom_pin)
            finite = bool(torch.isfinite(val).all())
            parity[name] = {
                "rel_l2_vs_exact_original":rel_orig,
                "rel_l2_vs_pinned":rel_pin,
                "max_abs_vs_pinned":float(d_pin.abs().max()),
                "finite":finite,
                "pass":finite and max(rel_orig,rel_pin) <= tolerance,
            }
            del val,d_orig,d_pin
        except Exception as exc:
            failures[name] = {
                "type":type(exc).__name__,
                "error":str(exc)[:2000],
                "traceback_tail":traceback.format_exc()[-2500:],
            }
    del baseline,pinned_out
    if not parity.get("pinned_R32",{}).get("pass",False):
        raise RuntimeError("Pinned accelerated baseline did not match exact original")
    if not parity.get("new_none",{}).get("pass",False):
        raise RuntimeError("New disabled mode failed original/pinned parity")

    active = [name for name in cfg if name not in failures and parity[name]["pass"]]
    for _ in range(warmups):
        for name in active:
            run(name)
    torch.cuda.synchronize()
    times={}
    for candidate in ("new_none","cache_dq","cache_both"):
        if candidate not in active:
            continue
        samples={"pinned_R32":[],candidate:[]}
        for _ in range(rounds):
            for name in ("pinned_R32",candidate,candidate,"pinned_R32"):
                samples[name].append(
                    batched_event_ms(lambda name=name: run(name),batch)
                )
        median={name:statistics.median(v) for name,v in samples.items()}
        times[candidate]={
            "median_full_head_ms":median,
            "speedup_pinned_over_candidate":median["pinned_R32"]/median[candidate],
            "samples_ms":samples,
        }

    tile = int(pinned_cfg["tile"])
    n_tiles=min(chunk,(len(q)+tile-1)//tile)
    per_tile_rows=tile
    per_row_bytes=int(q.shape[1])+4
    cache_mib={
        "dq":n_tiles*per_tile_rows*per_row_bytes/(1024**2),
        "both":2*n_tiles*per_tile_rows*per_row_bytes/(1024**2),
    }
    return {
        "head":head,"mode":mode,
        "q":int(q.shape[0]),"k":int(k.shape[0]),
        "tail":int(tail[0].shape[0]),
        "tile_chunk":chunk,
        "cache_extra_mib_per_chunk":cache_mib,
        "compiled_kernel_resources":compiled,
        "parity":parity,"failures":failures,
        "paired_head_timings":times,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--capture",
        default=r"D:\ComfyOutput\H3\math_probe\20260926-201504\attention_block_00_call_01.pt",
    )
    ap.add_argument("--heads",default="1,16")
    ap.add_argument("--modes",default="fast,balanced")
    ap.add_argument("--chunk",type=int,default=256)
    ap.add_argument("--warmups",type=int,default=4)
    ap.add_argument("--rounds",type=int,default=6)
    ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--tolerance",type=float,default=1e-4)
    ap.add_argument("--out",default=r"D:\ComfyOutput\H3\math_probe")
    args=ap.parse_args()
    heads=list(dict.fromkeys(int(x) for x in args.heads.split(",") if x.strip()))
    modes=list(dict.fromkeys(x.strip() for x in args.modes.split(",") if x.strip()))
    if not heads or not modes or any(m not in ("fast","balanced","accurate","max") for m in modes):
        ap.error("Provide valid heads and modes")
    if min(args.chunk,args.warmups,args.rounds,args.batch) < 1:
        ap.error("chunk, warmups, rounds, batch must be >=1")
    if not Path(args.capture).is_file():
        ap.error("Trusted --capture must exist")
    data=torch.load(args.capture,map_location="cpu",mmap=True,weights_only=False)
    if not isinstance(data,dict) or any(key not in data for key in ("q_qhd","k_khd","v_khd")):
        ap.error("Capture does not hold expected Q/K/V")
    rows,skipped=[],[]
    with tempfile.TemporaryDirectory(prefix="qcache_pinned_") as d1:
        with tempfile.TemporaryDirectory(prefix="qcache_original_") as d2:
            pinned,pinned_runner=pinned_modules(Path(d1))
            original,original_runner=exact.original_modules(exact.ORIGINAL,Path(d2))
            new=sw.load("_qcache_current_flat","h3/htrc_flat.py")
            runner=sw.load("_qcache_current_runner","h3/htrc_flat_runner.py")
            for head in heads:
                sample=sw.capture_head(data,head,torch.device("cuda"))
                if sample is None:
                    skipped.append({"head":head,"reason":"absent raw_head_index"})
                    continue
                for mode in modes:
                    try:
                        row=compare(sample,head,mode,original,original_runner,
                                    pinned,pinned_runner,new,runner,args.chunk,
                                    args.warmups,args.rounds,args.batch,args.tolerance)
                        rows.append(row)
                        print(json.dumps({
                            "head":head,"mode":mode,
                            "memory":row["cache_extra_mib_per_chunk"],
                            "compiled":row["compiled_kernel_resources"],
                            "parity":row["parity"],
                            "failures":row["failures"],
                            "speedups":{
                                key:timing["speedup_pinned_over_candidate"]
                                for key,timing in row["paired_head_timings"].items()
                            },
                        },indent=2))
                    except Exception as exc:
                        skipped.append({
                            "head":head,"mode":mode,
                            "type":type(exc).__name__,
                            "reason":"benchmark_exception",
                            "error":str(exc)[:2000],
                            "traceback_tail":traceback.format_exc()[-2600:],
                        })
                del sample

    report={
        "experiment":"h3_flat_compact_select_query_quantization_cache_v1",
        "gpu":torch.cuda.get_device_name(),
        "capture":args.capture,
        "source_commits":{
            "pinned_tail_merge":PINNED,
            "exact_original":exact.ORIGINAL,
            "experiment":subprocess.check_output(
                ["git","rev-parse","HEAD"],cwd=ROOT,text=True
            ).strip(),
        },
        "heads":heads,"modes":modes,"rounds":args.rounds,
        "batch":args.batch,
        "notes":[
            "PARENT is pinned old Python-range accelerated e47cc7a; no failed tl.range specialization.",
            "Control old source reloaded under separate private Python module name.",
            "rowmax split0 stores exact INT8 q/dq quantization and FP32 row scale.",
            "No new kernel launches; cache uses ~2.06/4.13 MiB for 256 tiles.",
            "Full native saved heads; no cropped proxy, no sink layout, no CUDA Graph, no denoise.",
            "Compare disabled new_none to pinned first; reject default-path compiler regression.",
            "Compiled spills and rel-L2 <=1e-4 are separate acceptance gates.",
        ],
        "rows":rows,"skipped_or_failed":skipped,
    }
    dest=Path(args.out)
    dest.mkdir(parents=True,exist_ok=True)
    filename=dest/f"flat_query_cache_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    filename.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",filename)
    if not rows or any(item.get("reason")=="benchmark_exception" for item in skipped) or any(
        entry["failures"] or any(not x["pass"] for x in entry["parity"].values())
        for entry in rows
    ):
        raise SystemExit("FAIL: query-cache compile or parity; JSON saved")


if __name__=="__main__":
    main()
