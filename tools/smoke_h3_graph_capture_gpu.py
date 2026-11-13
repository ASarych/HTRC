#!/usr/bin/env python3
"""Tiny real-CUDA preflight of HTRC Graph capture WITHOUT loading MiniMax H3.

Purpose: catch unsupported captured CUDA pointers and allocator/Graph
cleanup BEFORE a multi-GiB, six-step HyperFlow run. It is a safety
gate, not a performance/quality measurement. Runs in a separate Python
process; any failed capture must not poison the Comfy backend process.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3 import htrc_flat_directflow_live as flow


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("HTRC Graph capture smoke requires CUDA")
    if os.environ.get("APL_HTRC_GRAPH_CAPTURE_SAFE_MU") != "1":
        raise RuntimeError("Graph smoke refuses legacy capture-allocated MU")
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    # Balanced mode requires 512 complete 16-token leaves for its
    # production geometry. 8192 tokens is still tiny versus the 40,759
    # token HyperFlow workload, with no weights or text/VAEs loaded.
    heads, tokens, dim = 8, 8192, 128
    torch.manual_seed(424242)
    try:
        with torch.inference_mode():
            real_q = torch.randn(
                (1,heads,tokens,dim),device=dev,dtype=torch.bfloat16)
            real_k = torch.randn_like(real_q)
            real_v = torch.randn_like(real_q)
            marker = torch.zeros(
                (1,1,1,1),device=dev,dtype=torch.bfloat16)
            fake = marker.expand(1,heads,tokens,dim)

            def fill_kv(kstage,vstage):
                if kstage.device != dev or vstage.device != dev:
                    raise RuntimeError("Graph-owned K/V unexpectedly off CUDA")
                kstage.copy_(real_k)
                vstage.copy_(real_v)

            def fill_q(start,stop,stage):
                if stage.device != dev:
                    raise RuntimeError("Graph-owned Q unexpectedly off CUDA")
                src = real_q[0,:,start:stop,:]
                actual = stop-start
                stage[:,:actual,:].copy_(src)
                if actual < stage.shape[1]:
                    stage[:,actual:,:].copy_(
                        src[:,-1:,:].expand(-1,stage.shape[1]-actual,-1))

            started = time.perf_counter()
            outs = []
            # The SECOND call exercises reuse of the captured session,
            # updating the same Graph-owned K/V/Q buffers.
            for _ in range(2):
                out = flow.graph_attention_bthd(
                    fake,fake,fake,
                    heads=heads,mode="balanced",tiles=16,
                    budget_mib=1024,reserve_mib=256,
                    workspace=True,compare_only=True,
                    query_provider=fill_q,kv_provider=fill_kv,
                    research_source_kind="native_h3",
                    research_group_size=heads)
                if (out.device != dev
                    or tuple(out.shape)!=(1,tokens,heads,dim)
                    or not torch.isfinite(out).all().item()):
                    raise RuntimeError("Live HTRC Graph smoke output invalid")
                outs.append(out)
                real_q.mul_(0.9)
            torch.cuda.synchronize(dev)
            ms = (time.perf_counter()-started)*1000
            print(
                f"HTRC Graph CUDA capture smoke PASS: 2 real "
                f"native-provider Graph replays, shape={(1,tokens,heads,dim)}, "
                f"wall_ms={ms:.2f}, "
                f"peak_allocated_mib={torch.cuda.max_memory_allocated(dev)/(1<<20):.1f}",
                flush=True,
            )
            del outs
    finally:
        # Destroy Graph-pool objects only AFTER stream completion.
        if torch.cuda.is_available():
            torch.cuda.synchronize(dev)
        flow.clear_session_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize(dev)


if __name__ == "__main__":
    main()
