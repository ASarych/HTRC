#!/usr/bin/env python3
"""CPU/source admission for real Tile32 leaf-stats BT16/BT32 CUDA Graph A/B.

No torch import, CUDA allocation, H3 model load or GPU benchmark here.
Checks exact capture-cache separation, opt-in kernel dispatch, and
guarded one-JSON full-output parity experiment, NOT kernel performance.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/"h3"/"htrc_flat_head_batch.py"
FLOW=ROOT/"h3"/"htrc_flat_directflow_live.py"
GPU=ROOT/"tools"/"benchmark_h3_tile32_stats_bt_ab_gpu.py"


def check():
    text={}
    for p in (BATCH,FLOW,GPU):
        text[p.name]=p.read_text(encoding="utf-8")
        ast.parse(text[p.name],filename=str(p))
    batch=text[BATCH.name]
    flow=text[FLOW.name]
    gpu=text[GPU.name]
    for marker in (
        'os.environ.get("APL_HTRC_TILE32_STATS_BT", "16")',
        'stats_bt_raw not in ("16", "32")',
        'BT,BL=int(stats_bt_raw),4',
        'blocks=math.ceil(centers/BT)',
        'BATCH_TILE_BLOCKS_PER_HEAD=blocks',
        'query_staging != "native_direct"',
        'kv_cache is None or query_workspace is None',
        'vb_mode != "materialized"',
        'h != 8',
        'tile_count <= 32',
    ):
        assert marker in batch,"Broken kernel opt-in: "+marker
    for marker in (
        '_env_int("APL_HTRC_TILE32_STATS_BT", 16, 16, 32)',
        'stats_bt not in (16, 32)',
        'stats_bt == 32 and (',
        'tiles != 32 or not compare_only',
        'research_group_size != 8',
        'research_source_kind != "native_h3"',
        'research_source_kind, stats_bt)',
        '"stats_kernel_bt":stats_bt',
    ):
        assert marker in flow,"Broken Tile32 session safety: "+marker
    # The 2026-11-14 first GPU run failed at BT16_capture before
    # launching a kernel: DirectFlow requires inference-only tensors.
    # Enforce the actual AST decorator on the benchmark's GPU entry point,
    # not a loose string match or a global grad-mode state mutation.
    gpu_tree=ast.parse(gpu,filename=str(GPU))
    emit_defs=[
        node for node in gpu_tree.body
        if isinstance(node,ast.FunctionDef) and node.name=="emit"
    ]
    assert len(emit_defs)==1, "Expected exactly one GPU benchmark entry"
    decorators=emit_defs[0].decorator_list
    assert len(decorators)==1, "GPU entry requires one scoped decorator"
    d=decorators[0]
    assert (
        isinstance(d,ast.Call)
        and isinstance(d.func,ast.Attribute)
        and isinstance(d.func.value,ast.Name)
        and d.func.value.id=="torch"
        and d.func.attr=="inference_mode"
        and not d.args and not d.keywords
    ), "GPU benchmark must enter torch.inference_mode() before DirectFlow"
    for marker in (
        'for bt in (16, 32):',
        'for bank_index in (0, 1):',
        'for _ in range(rounds):',
        'research_group_size=8',
        'research_source_kind="native_h3"',
        'torch.equal(a,b)',
        'stage_gpu_events=event_intervals',
        '"leaf_stats"',
        '"tile_kernel"',
        '"merge_kernel"',
        'flow.clear_session_cache()',
        '"promotion_to_real_h3":"UNRUN_NOT_AUTHORIZED"',
        '"status":"FAIL_CLOSED"',
        'temporary.replace(args.output)',
        'default=40759',
    ):
        assert marker in gpu,"Broken A/B test safety: "+marker
    tokens=40759
    leaves=tokens//16
    groups=7
    windows=math.ceil(tokens/(32*64))
    assert windows==20
    assert leaves==2547
    assert math.ceil(leaves/4)==637
    stats_grid={
        bt: {
            "centres_per_cta":bt,
            "cta_per_full_window_per_8_head_group":
                8*math.ceil(32/bt)*math.ceil(leaves/4),
            "cta_per_full_attention":
                groups*windows*8*math.ceil(32/bt)*math.ceil(leaves/4),
        }
        for bt in (16,32)
    }
    assert stats_grid[16]["cta_per_full_attention"]==2*(
        stats_grid[32]["cta_per_full_attention"]
    )
    assert stats_grid[16]["cta_per_full_attention"]==1426880
    print("PASS: Tile32 BT16/BT32 guarded 40,759-token kernel A/B; "
          "2x fewer leaf-stats CTA program instances, not 2x less math")
    print("  full-attention stats CTA:", stats_grid)
    print("  CUDA kernel execution / numerical parity / speed: UNRUN")
    return 0


if __name__=="__main__":
    raise SystemExit(check())
