"""CPU static contract: no Q-quality coefficient; exact Q layout optimization only."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/"h3"/"htrc_flat_head_batch.py"
LIVE=ROOT/"h3"/"htrc_flat_live_backend.py"
BENCH=ROOT/"tools"/"benchmark_h3_flat_q_input_pack_ab.py"


class FixedMathQPrepTest(unittest.TestCase):
    def test_syntax(self):
        for path in (PREP,LIVE,BENCH):
            with self.subTest(file=str(path)):
                ast.parse(path.read_text(encoding="utf-8"),filename=str(path))

    def test_no_live_speed_quality_coefficient(self):
        s=LIVE.read_text(encoding="utf-8")
        tree=ast.parse(s)
        candidates=[
            node for node in ast.walk(tree)
            if isinstance(node,ast.FunctionDef)
            and node.name=="flat_attention_bthd"
        ]
        self.assertEqual(len(candidates),1)
        args=[a.arg for a in candidates[0].args.kwonlyargs]
        self.assertNotIn("quality",args)
        self.assertNotIn("qstep",s)
        self.assertNotIn('cfg["quality"]',s)
        self.assertNotIn('cfg["exact_leaves"]=max(1,round(',s)
        self.assertIn('"quality_coefficient_used":False',s)
        self.assertIn("shared.flat_config(mode,tiles)",s)

    def test_optin_q_pack_keeps_identical_math(self):
        s=PREP.read_text(encoding="utf-8")
        self.assertIn('query_staging="gather"',s)
        self.assertIn('query_staging == "view_copy"',s)
        self.assertIn("positions != list(range(positions[0],positions[0]+h))",s)
        self.assertIn("Q.view(h,n_query,dim).copy_(native_hsd)",s)
        self.assertIn('Q=gather("q_qhd",n_query)',s)
        self.assertIn("MU=Q.mean(dim=2)",s)
        self.assertIn("ids=torch.topk(sel,kx",s)
        self.assertIn("excl.scatter_(1,ids,1)",s)

    def test_one_json_and_full_parity_gate(self):
        s=BENCH.read_text(encoding="utf-8")
        self.assertIn('flat_qprep_fixed_math_ab.json',s)
        self.assertIn('STATES=("Q","MU","G","VB","LZ","SG","IDX","EXCL")',s)
        self.assertIn("torch.equal(a[state],b[state])",s)
        self.assertIn("torch.equal(oa,ob)",s)
        self.assertIn('case["status"]="FAIL_MATH_MISMATCH"',s)
        self.assertIn('case["status"]="FAIL_OUTPUT_MISMATCH"',s)
        self.assertIn('"quality_coefficient_used":False',s)
        self.assertIn('PACKS=("gather","view_copy")',s)
        self.assertNotIn('repeat_available_heads',s)

    def test_cli_only_declared_args(self):
        tree=ast.parse(BENCH.read_text(encoding="utf-8"))
        names=set()
        fields=set()
        for node in ast.walk(tree):
            if (isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute)
                and node.func.attr=="add_argument"
                and isinstance(node.func.value,ast.Name)
                and node.func.value.id=="p"
                and node.args and isinstance(node.args[0],ast.Constant)):
                names.add(node.args[0].value[2:].replace("-","_"))
            if (isinstance(node,ast.Attribute)
                and isinstance(node.value,ast.Name)
                and node.value.id=="args"):
                fields.add(node.attr)
        self.assertEqual(set(),fields-names)


if __name__=="__main__":
    unittest.main()
