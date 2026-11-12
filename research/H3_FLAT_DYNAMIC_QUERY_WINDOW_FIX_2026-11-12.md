# H3 Flat variable query-window fix — dynamic, not fixed five tiles (2026-11-12)

Branch: experiment/flat-direct-qkv-kernels-20261112. Opt-in Direct QKV experiments only.

## Uploaded first GPU A/B (failure record)

The first local RTX 5070 Ti A/B was **FAIL** on the final of 19 windows at 18,723 tokens, 56 source heads, 16 tiles/window. It completed first/middle representative output checks on 4 heads but failed for the final window at head range 0..4. Its 291 remaining Q rows were padded to all 1,024 rows (=16 tiles). Direct Q produced MU=0 for completely inactive padded centers, while the staged control computed their means from repeated-last-row padding. Downstream G/VB/LZ/IDX/EXCL therefore differed **in unneeded padded centers**. Direct K/V remained equal to the staged baseline in the observed state checks. This is a diagnosis from the failure location/state pattern, not a fresh corrected GPU PASS.

Evidence: [flat_direct_qkv_ab_initial_failure_20261112.json](evidence/flat_direct_qkv_ab_initial_failure_20261112.json).

## Fix: universal window planner

For each query window with real remaining rows R, configured capacity W (tiles) and center grouping G:

\`\`\`python
needed_tiles = ceil(R / tile_rows)
active_tiles = ceil(needed_tiles / G) * G
assert 0 < active_tiles <= W
padded_rows = active_tiles * tile_rows
\`\`\`

W and G are parameters; 5 is **not** an algorithm constant. For W=16, tile_rows=64, R=291, G=1, active_tiles=5. A different window size or group produces a correspondingly different count. Grouped center math is retained; no reduction of exact leaves, K/V rows, top-k count or near-field selection.

- \`h3/htrc_flat_windows.py\`: pure planner and \`compact_workspace_view\`.
- \`h3/htrc_flat_live_backend.py\`: compute per-window active tile count; pad only to the current last tile or last group; dispatch exact same stats/rowmax/topk/Tile/Merge over the active grid.
- \`compact_workspace_view\` retains the **single** preallocated workspace. It exposes contiguous shape-correct prefix aliases of G/VB/LZ/SG/SEL/IDX/EXCL/partials/out, not \`[:, :short]\` gaps across heads. Memory storage pointer unchanged. Sequential dispatch on the same CUDA stream preserves write ordering.
- \`tools/benchmark_h3_flat_direct_qkv_ab.py\`: A/B uses per-window active geometry, reports it explicitly, checks compact baseline output against the old full-padded baseline on real rows, then compares all four arms including the last window.
- \`tools/test_h3_flat_tail_geometry.py\`: CPU test many query lengths, 4/8/16/32 window capacities, G=1/2/4, several head counts; check no extra tensor backing storage.

## Validation status

Isolated local CPU unit tests of the pure window planner and workspace alias (torch CPU) **PASS**: five methods. Exact GitHub module import/unittest, Triton compile and end-to-end RTX A/B on the new HEAD remain **UNRUN** until executed in the user's CUDA environment. Do not promote.

Single expected report: \`D:\ComfyOutput\H3\math_probe\flat_direct_qkv_ab.json\`. Must show all 4-head and 56-head cases \`completed=true\`, 0 errors, \`legacy_tail_output_checks>0\`, and final-window state/output parity before speed is considered.

No new quality coefficient and no production/default change.
