"""Shape-matched HTRC query-preparation microprofile; no model/video load."""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", r"D:\ComfyCache\triton")
sys.path.insert(0, r"D:\ComfyNodes-flat-split-20261111")

import torch
from h3 import htrc_flat_head_batch as batch
from h3 import htrc_flat_live_backend as live
from h3 import htrc_flat_model_shadow as shared


def stats(values):
    return {
        "total_ms": round(sum(values), 3),
        "mean_ms": round(statistics.mean(values), 3),
        "p50_ms": round(statistics.median(values), 3),
        "p95_ms": round(sorted(values)[max(0, math.ceil(.95 * len(values)) - 1)], 3),
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device=torch.device("cuda")
    free_before,total=torch.cuda.mem_get_info(device)
    if free_before < 8 * 1024**3:
        raise SystemExit(f"Refusing profile with only {free_before/1024**3:.2f} GiB free VRAM")

    seq,heads,dim,tiles,group_cap=18723,56,128,16,11
    tile_rows=64
    cfg=shared.flat_config("fast", tiles)
    tri=live._triton()
    tri._dll()

    # Zero-filled tensors keep the microprofile deterministic while exercising
    # the same tensor sizes, casts, gathers, Triton grids and top-k selection.
    q=torch.zeros((seq,heads,dim),device=device,dtype=torch.bfloat16)
    k=torch.zeros_like(q)
    v=torch.zeros_like(q)
    capture={"q_qhd":q,"k_khd":k,"v_khd":v,"raw_head_index":list(range(heads))}

    cache_times=[]
    workspace_times=[]
    prep_wall=[]
    stage_values={}
    call_count=0
    t0=time.perf_counter()
    for begin in range(0,heads,group_cap):
        ids=list(range(begin,min(heads,begin+group_cap)))
        t=time.perf_counter()
        kv=batch.prepare_gpu_head_kv_cache(capture,ids,tri)
        cache_times.append((time.perf_counter()-t)*1000)
        t=time.perf_counter()
        ws=shared.alloc_workspace(capture,len(ids),tiles,cfg)
        workspace_times.append((time.perf_counter()-t)*1000)
        try:
            for qi in range(0,seq,tiles*tile_rows):
                qe=min(seq,qi+tiles*tile_rows)
                valid=qe-qi
                src=q[qi:qe]
                if valid < tiles*tile_rows:
                    src=torch.cat((src,src[-1:].expand(tiles*tile_rows-valid,-1,-1)),dim=0)
                view=dict(capture)
                view["q_qhd"]=src
                stages={}
                t=time.perf_counter()
                states,*_=batch.prepare_batched_head_states(
                    view,ids,cfg,tri,tile_count=tiles,quant_backend="triton",
                    input_cast="gpu",kv_cache=kv,query_workspace=ws,
                    valid_query_rows=valid,stage_timings_ms=stages,
                )
                prep_wall.append((time.perf_counter()-t)*1000)
                for name,value in stages.items():
                    stage_values.setdefault(name,[]).append(float(value))
                del states,view,src
                call_count+=1
        finally:
            del kv,ws
    torch.cuda.synchronize(device)
    total_ms=(time.perf_counter()-t0)*1000
    free_after,_=torch.cuda.mem_get_info(device)
    out={
        "kind":"shape_matched_synthetic_qkv_microprofile",
        "shape":{"tokens":seq,"heads":heads,"dim":dim,"dtype":"bf16","tiles":tiles,"tile_rows":tile_rows,"group_cap":group_cap},
        "config":{"mode":"fast","exact_leaves":int(cfg["exact_leaves"]),"center_group":int(cfg.get("kern",{}).get("group",1)),"triton_cache":os.environ["TRITON_CACHE_DIR"]},
        "attention_prep_calls":call_count,
        "groups":math.ceil(heads/group_cap),
        "windows":math.ceil(seq/(tiles*tile_rows)),
        "total_profile_wall_ms":round(total_ms,3),
        "kv_cache_build_per_group":stats(cache_times),
        "workspace_alloc_per_group":stats(workspace_times),
        "prepare_total_per_window_group":stats(prep_wall),
        "prepare_stages_per_window_group":{name:stats(values) for name,values in stage_values.items()},
        "stage_sum_ms_instrumented":round(sum(sum(v) for v in stage_values.values()),3),
        "gpu_free_before_gib":round(free_before/1024**3,3),
        "gpu_free_after_gib":round(free_after/1024**3,3),
        "gpu_total_gib":round(total/1024**3,3),
        "caveat":"Synthetic zero values; shape/path are representative, but cache state, allocator pressure, real data and whole-model overlap differ. Per-stage timers synchronize CUDA and add diagnostic overhead.",
    }
    path=Path(r"D:\ComfyOutput\H3\math_probe\priestess_flat_gpu\qprep_shape_profile.json")
    path.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
