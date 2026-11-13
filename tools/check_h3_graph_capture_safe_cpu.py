#!/usr/bin/env python3
"""Import-free CPU source gate for Windows CUDA-Graph pointer-lifetime repair.

This is NOT a GPU parity benchmark. Actual capture/replay is separately
gated by smoke_h3_graph_capture_gpu.py in a short-lived CUDA process.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
HEAD_BATCH=ROOT/"h3"/"htrc_flat_head_batch.py"
GRAPH=ROOT/"h3"/"htrc_flat_directflow_live.py"
SMOKE=ROOT/"tools"/"smoke_h3_graph_capture_gpu.py"

def check():
    hb=HEAD_BATCH.read_text(encoding="utf-8")
    graph=GRAPH.read_text(encoding="utf-8")
    smoke=SMOKE.read_text(encoding="utf-8")
    for file,source in ((HEAD_BATCH,hb),(GRAPH,graph),(SMOKE,smoke)):
        ast.parse(source,filename=str(file))
    checks={
        "only opt-in captured pointer changes": (
            'os.environ.get("APL_HTRC_GRAPH_CAPTURE_SAFE_MU")' in hb
            and 'os.environ.get("APL_HTRC_GRAPH_CAPTURE_SAFE_MU","0")=="1"' in graph
        ),
        "native direct MU from preallocated CUDA buffer": (
            'MU=(query_workspace.get("MU") if query_workspace is not None' in hb
            and 'if MU is None:' in hb
            and 'elif (not isinstance(MU,torch.Tensor)' in hb
            and '"native_direct"' in graph
            and '"native_torch_mu"' in graph
            and "triton_module._direct_q_mu_kernel" in hb
        ),
        "stable captured topk values and indices": (
            '"_TOPK_VALUES" in query_workspace' in hb
            and '"_TOPK_IDS" in query_workspace' in hb
            and 'torch.topk(sel,kx,dim=-1,sorted=False,out=(topv,ids))' in hb
            and 'idx.copy_(ids.view(h,centers,kx))' in hb
        ),
        "balanced config without explicit group keeps centre group 1": (
            'centers = geom.tile_count // int(cfg.get("kern", {}).get("group", 1))'
                in graph
            and 'cfg["kern"]["group"]' not in graph
        ),
        "MU allocated outside capture": (
            'ws["MU"] = torch.empty(' in graph
            and graph.index('ws["MU"] = torch.empty(')
              < graph.index("with torch.cuda.graph(graph):")
        ),
        "topk allocated outside capture": (
            'ws["_TOPK_VALUES"] = torch.empty(' in graph
            and 'ws["_TOPK_IDS"] = torch.empty(' in graph
            and graph.index('ws["_TOPK_IDS"] = torch.empty(')
              < graph.index("with torch.cuda.graph(graph):")
        ),
        "smoke GPU before model": (
            "def fill_q(start,stop,stage):" in smoke
            and "def fill_kv(kstage,vstage):" in smoke
            and "for _ in range(2):" in smoke
            and "flow.graph_attention_bthd(" in smoke
            and 'research_source_kind="native_h3"' in smoke
            and "flow.clear_session_cache()" in smoke
        ),
    }
    for label,ok in checks.items():
        print(("PASS" if ok else "FAIL")+": "+label)
    return 0 if all(checks.values()) else 1

if __name__=="__main__":
    sys.exit(check())
