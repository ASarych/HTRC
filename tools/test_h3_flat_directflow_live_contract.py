"""CPU-only contracts for opt-in real-model Direct-Flow Graph integration.

No CUDA inference, no GPU timings, and no claim about actual H3 parity.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT/"h3"/"htrc_flat_live_backend.py"
GRAPH=ROOT/"h3"/"htrc_flat_directflow_live.py"

def _extract(name):
    tree=ast.parse(GRAPH.read_text(encoding="utf-8"),filename=str(GRAPH))
    funcs=[node for node in tree.body
           if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(funcs)!=1:
        raise AssertionError("Missing helper: "+name)
    module=ast.Module(body=funcs,type_ignores=[])
    ast.fix_missing_locations(module)
    scope={"os":os}
    exec(compile(module,str(GRAPH),"exec"),scope)
    return scope[name]


class LiveDirectFlowContract(unittest.TestCase):
    def test_full_python_source_syntax_and_optin(self):
        live=LIVE.read_text(encoding="utf-8")
        graph=GRAPH.read_text(encoding="utf-8")
        for path,source in ((LIVE,live),(GRAPH,graph)):
            with self.subTest(path=path.name):
                ast.parse(source,filename=str(path))
        self.assertIn('APL_HTRC_FLAT_DISPATCH","eager"',live)
        self.assertIn('if dispatch is None else str(dispatch)',live)
        self.assertIn('if choice == "graph":',live)
        self.assertIn('dispatch="eager"',graph)
        self.assertIn('return out.unsqueeze(0)',live)
        self.assertIn('return output.unsqueeze(0)',graph)

    def test_q_window_true_rows_and_padding_cpu(self):
        import torch
        copy=_extract("_copy_q")
        q=torch.arange(1*4*70*128,dtype=torch.float32).reshape(1,4,70,128).bfloat16()
        win=SimpleNamespace(start=64,end=70,valid_rows=6,padded_rows=64,tile_count=1)
        stage=torch.empty((4,64,128),dtype=torch.bfloat16)
        copy(q,win,stage)
        self.assertTrue(torch.equal(stage[:,:6],q[0,:,64:70]))
        self.assertTrue(torch.equal(
            stage[:,6:],q[0,:,-1:,:].expand(-1,58,-1)
        ))
        with self.assertRaises(ValueError):
            copy(q,win,torch.empty((4,32,128),dtype=torch.bfloat16))

    def test_window_geometry_key_independent_of_position(self):
        key=_extract("_geometry_key")
        a=SimpleNamespace(tile_count=16,valid_rows=1024,start=0)
        b=SimpleNamespace(tile_count=16,valid_rows=1024,start=1024)
        c=SimpleNamespace(tile_count=5,valid_rows=291,start=1024)
        self.assertEqual(key(a),key(b))
        self.assertNotEqual(key(a),key(c))

    def test_memory_and_group_bounds(self):
        env=_extract("_env_int")
        with patch.dict(os.environ,{"TEST_HTRC_LIMIT":"11"}):
            self.assertEqual(env("TEST_HTRC_LIMIT",4,1,11),11)
            with self.assertRaises(ValueError):
                env("TEST_HTRC_LIMIT",4,1,10)
        with patch.dict(os.environ,{"TEST_HTRC_LIMIT":"0"}):
            with self.assertRaises(ValueError):
                env("TEST_HTRC_LIMIT",4,1,11)
        graph=GRAPH.read_text(encoding="utf-8")
        for needle in ("_head_plan(","max_extra*_MIB",
                       "free < min_free*_MIB", "clear_session_cache()"):
            self.assertIn(needle,graph)

    def test_qkv_refresh_before_replay_and_exact_parity(self):
        graph=GRAPH.read_text(encoding="utf-8")
        refresh=graph[graph.index("def _refresh_kv("):graph.index("def _replay(")]
        self.assertIn('stage["k"].copy_(bank[1])',refresh)
        self.assertIn('stage["v"].copy_(bank[2])',refresh)
        self.assertIn('tr._quant_key_int8_native[',refresh)
        self.assertIn('tk.copy_(stage["k"][0, :, -count:, :])',refresh)
        replay=graph[graph.index("def _replay("):graph.index("def _measurement(")]
        self.assertLess(replay.index("_refresh_kv(stage, bank, tr)"),
                        replay.index('item["graph"].replay()'))
        self.assertLess(replay.index('_copy_q(bank[0], geom, item["q"])'),
                        replay.index('item["graph"].replay()'))
        self.assertIn('torch.equal(group_out, eager_out[0])',graph)
        self.assertIn('if not exact:',graph)
        self.assertIn('if verify:',graph)
        self.assertIn('quality_coefficient_used":False',graph)
        self.assertIn('fixed_exact_leaves":256',graph)
        self.assertIn('status":"FAIL_CLOSED"',graph)

    def test_cross_stream_staging_dependency(self):
        graph=GRAPH.read_text(encoding="utf-8")
        self.assertIn('torch.cuda.current_stream(dev).wait_event(previous)',graph)
        self.assertIn('completed.record(torch.cuda.current_stream(dev))',graph)
        self.assertIn('pending.synchronize()',graph)
        self.assertIn('torch.cuda.synchronize(dev)',graph)

    def test_single_report_no_tensor_dumps_or_workflow_mutation(self):
        graph=GRAPH.read_text(encoding="utf-8")
        self.assertIn("flat_directflow_real_h3_ab.json",graph)
        self.assertIn('tmp.replace(path)',graph)
        self.assertNotIn('torch.save(',graph)
        self.assertNotIn('torch.load(',graph)
        self.assertNotIn('os.system(',graph)


if __name__=="__main__":
    unittest.main()
