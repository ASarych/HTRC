#!/usr/bin/env python3
"""No-CUDA safety gate for the 32-tile H3 Graph research A/B.

Compute the actual 40,759-token seven-group window plan and the known
persistent staging byte estimate, then structurally require fail-closed
model-resident admission and a separate 16-vs-32 CUDA output guard.
Private CUDA Graph pools and peak memory are NOT analytically proven.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3 import htrc_flat_windows as windows
from h3 import htrc_flat_graph_group_plan as plan

FLOW = ROOT / "h3" / "htrc_flat_directflow_live.py"
GPU = ROOT / "tools" / "smoke_h3_graph_tiles_ab_gpu.py"
TOKENS = 40759
HEADS = 56
GROUP = 8
MIB = 1 << 20


def check():
    assert HEADS % GROUP == 0
    assert list(windows.iter_query_windows(16, 16))
    cases = {}
    for tiles, count in ((16, 40), (32, 20)):
        geometry = list(windows.iter_query_windows(
            TOKENS, tiles, tile_rows=64, center_group=1
        ))
        assert len(geometry) == count
        assert geometry[0].tile_count == tiles
        assert sum(g.valid_rows for g in geometry) == TOKENS
        assert geometry[-1].valid_rows != geometry[-1].padded_rows
        assert sum(g.padded_rows for g in geometry) >= TOKENS
        model = plan.graph_stage_head_bytes(
            tokens=TOKENS, tiles=tiles, mode="balanced",
            vb_mode="materialized"
        )
        assert model["private_cuda_graph_pool_unmeasured"] is True
        assert model["unique_graph_geometries"] == 2
        assert len(model["windows"]) == 2
        cases[tiles] = {
            "windows_per_group": count,
            "replays_per_attention": count * (HEADS // GROUP),
            "persistent_stage_mib_per_group": (
                model["estimated_persisted_bytes_per_head"] * GROUP / MIB
            ),
        }
    assert cases[16]["replays_per_attention"] == 280
    assert cases[32]["replays_per_attention"] == 140
    delta = (
        cases[32]["persistent_stage_mib_per_group"] -
        cases[16]["persistent_stage_mib_per_group"]
    )
    # The new candidate trades ~292 MiB of estimated persistent stage
    # for HALF as many Graph launches. Actual CUDA graph private pools
    # remain unmeasured, so this only validates the admission estimate.
    assert 200 < delta < 400, delta
    assert cases[32]["persistent_stage_mib_per_group"] < 1152

    flow = FLOW.read_text(encoding="utf-8")
    smoke = GPU.read_text(encoding="utf-8")
    for file, contents in ((FLOW,flow),(GPU,smoke)):
        ast.parse(contents, filename=str(file))
    required_flow = (
        'os.environ.get("APL_HTRC_DIRECTFLOW_TILES32_ADMISSION") == "1"',
        'if tiles != 32:',
        "group_plan.graph_stage_head_bytes(",
        'required = 8 * int(estimate["estimated_persisted_bytes_per_head"])',
        "torch.cuda.mem_get_info(bank[0].device)",
        "free < required + 1024 * _MIB",
        "raise MemoryError(",
        "extra_reserved > max_extra*_MIB or free < min_free*_MIB",
        "graph.replay()",
    )
    for marker in required_flow:
        assert marker in flow, "Missing 32-tile safety: " + marker
    required_smoke = (
        "for tiles in (16, 32):",
        "heads, tokens, dim = 8, 8319, 128",
        "for replay in range(2):",
        "torch.equal(a, b)",
        "flow.clear_session_cache()",
        'research_source_kind="native_h3"',
        "weight_checkpoint_loaded",
        "FAIL_NONIDENTICAL_OUTPUT",
    )
    for marker in required_smoke:
        assert marker in smoke, "Missing 32-tile CUDA A/B: " + marker
    print("HTRC Tiles32 CPU safety PASS:", cases,
          "estimated_extra_persistent_mib=", round(delta, 3),
          "graph_private_pool_unmeasured=True")
    return 0


if __name__ == "__main__":
    sys.exit(check())
