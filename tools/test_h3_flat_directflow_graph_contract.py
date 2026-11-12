"""CPU static contract: Direct-Flow static-input graph experiment is opt-in."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
BENCH=ROOT/"tools"/"benchmark_h3_flat_directflow_graph_ab.py"
PREP=ROOT/"h3"/"htrc_flat_head_batch.py"
LIVE=ROOT/"h3"/"htrc_flat_live_backend.py"


class DirectFlowGraphStaticTest(unittest.TestCase):
    def test_ast(self):
        for path in (BENCH,PREP,LIVE):
            with self.subTest(name=str(path)):
                ast.parse(path.read_text(encoding="utf-8"),filename=str(path))

    def test_graph_only_in_separate_benchmark(self):
        bench=BENCH.read_text(encoding="utf-8")
        live=LIVE.read_text(encoding="utf-8")
        self.assertIn("with torch.cuda.graph(graph):",bench)
        self.assertIn("graph.replay()",bench)
        self.assertIn("bit_exact_eager_vs_replay",bench)
        self.assertIn("graph_extra_reserved_mib",bench)
        self.assertIn("graph_capture_wall_ms",bench)
        self.assertIn("capture_break_even_replays_by_sync_wall",bench)
        self.assertIn("No production or live-backend dispatch integration",bench)
        self.assertNotIn("torch.cuda.graph(",live)

    def test_gather_positions_are_lazy(self):
        prep=PREP.read_text(encoding="utf-8")
        self.assertIn("pos=None",prep)
        self.assertIn("nonlocal pos",prep)
        self.assertIn("if pos is None:",prep)
        self.assertIn("raw.index_select(1,pos)",prep)

    def test_full_and_short_geometry_never_duplicate_same_window(self):
        s=BENCH.read_text(encoding="utf-8")
        self.assertIn("full=next((g for g in geometries",s)
        self.assertIn("short=next((g for g in reversed(geometries)",s)
        self.assertIn('if not requested:',s)
        self.assertNotIn('requested=[geometries[0]]',s)

    def test_fast_preset_implicit_group_and_validation(self):
        # Execute the real pure configuration helper without importing
        # torch/Triton; catching absent `group` before GPU is allocated.
        tree=ast.parse(BENCH.read_text(encoding="utf-8"))
        helper=next(n for n in tree.body
                    if isinstance(n,ast.FunctionDef)
                    and n.name=="center_group_from_config")
        subset=ast.Module(body=[helper],type_ignores=[])
        ast.fix_missing_locations(subset)
        env={}
        exec(compile(subset,str(BENCH),"exec"),env)
        resolve=env["center_group_from_config"]
        self.assertEqual(resolve({"kern":{}}),1)
        self.assertEqual(resolve({}),1)
        for v in (1,2,4):
            self.assertEqual(resolve({"kern":{"group":v}}),v)
        for v in (None,False,0,-1,3,8,1.0,"1"):
            with self.subTest(value=v),self.assertRaises(ValueError):
                resolve({"kern":{"group":v}})
        with self.assertRaises(ValueError):
            resolve({"kern":None})

    def test_fixed_math_and_one_report(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('"exact_leaves"])!=256',src)
        self.assertIn('center_group=center_group_from_config(cfg)',src)
        self.assertIn('center_group=center_group',src)
        self.assertNotIn('cfg["kern"]["group"]',src)
        self.assertIn("flat_directflow_graph_ab.json",src)
        self.assertIn('default="both"',src)
        self.assertIn('window_capture(cap,geom)',src)
        self.assertIn("torch.equal(reference[key],actual)",src)
        self.assertIn('"quality_coefficient_used":False',src)


if __name__=="__main__":
    unittest.main()
