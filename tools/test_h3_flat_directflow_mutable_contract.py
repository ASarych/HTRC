"""CPU-only source contracts for mutable-QKV CUDA Graph experiment.

These tests do not assert GPU numeric parity, capture success or runtime speed.
"""
from __future__ import annotations
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/"tools"/"benchmark_h3_flat_directflow_mutable_qkv_ab.py"
LIVE=ROOT/"h3"/"htrc_flat_live_backend.py"
PREP=ROOT/"h3"/"htrc_flat_head_batch.py"


class MutableGraphContract(unittest.TestCase):
    def test_python_ast(self):
        for path in (BENCH,LIVE,PREP):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"),filename=str(path))

    def test_qkv_refresh_and_quant_same_stream(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('stage["k"].copy_(bank[1])',src)
        self.assertIn('stage["v"].copy_(bank[2])',src)
        self.assertIn('tr._quant_key_int8_native[',src)
        self.assertIn('tail_k.copy_(stage["k"][0,:,-ntail:,:])',src)
        self.assertIn('tail_v.copy_(stage["v"][0,:,-ntail:,:])',src)
        self.assertIn('target[:,:rows,:].copy_(src)',src)
        self.assertIn('reg["graph"].replay()',src)
        self.assertLess(src.index('update_kv_stage(stage,bank,tr)'),
                        src.index('reg["graph"].replay()'))

    def test_reuse_geometry_not_fixed_last_five(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('def geometry_key(window):',src)
        self.assertIn('return (window.tile_count,window.valid_rows)',src)
        self.assertIn('windows.iter_query_windows(',src)
        self.assertIn('keys[geometry_key(geometry)]=geometry',src)
        self.assertIn('stage["registry"][geometry_key(geom)]',src)
        self.assertNotIn('tile_count=5',src)

    def test_true_input_change_and_exact_full_output(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('sources=create_input_banks(args)',src)
        self.assertIn('torch.equal(eager_out,graph_out)',src)
        self.assertIn('if torch.equal(saved_output,graph_out):',src)
        self.assertIn('tuple(torch.randn(shape',src)
        self.assertIn('checks["OUTPUT_FULL"]',src)
        self.assertIn('checks[name]=bool(torch.equal(eager[name],graph[name]))',src)
        self.assertIn('snapshot_state(st)',src)
        self.assertIn('K8/KS refreshed once per attention call',src)

    def test_eager_workspace_needs_complete_qkv_capture(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('eager_source_capture=shared.capture_from_bhsd(',src)
        self.assertIn('*sources[0],args.group_heads',src)
        self.assertIn('eager_ws=shared.alloc_workspace(',src)
        self.assertIn('eager_source_capture,args.group_heads,args.tiles,cfg',src)
        self.assertNotIn('stage["capture"],args.group_heads,args.tiles,cfg',src)

    def test_memory_and_production_boundaries(self):
        src=BENCH.read_text(encoding="utf-8")
        live=LIVE.read_text(encoding="utf-8")
        self.assertIn('max_memory_reserved()',src)
        self.assertIn('signed_reserved_delta_mib',src)
        self.assertIn('peak_above_precapture_reserved_mib',src)
        self.assertIn('flat_directflow_mutable_qkv_ab.json',src)
        self.assertIn('capture_break_even_full_attention_calls_by_sync_wall',src)
        self.assertIn('"quality_coefficient_used":False',src)
        self.assertNotIn('torch.cuda.graph(',live)


if __name__=="__main__":
    unittest.main()
