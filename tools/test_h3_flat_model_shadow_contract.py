"""CPU-only regression for safe native H3 model-resident Flat shadow wiring."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    "shadow":ROOT/"h3"/"htrc_flat_model_shadow.py",
    "runner":ROOT/"tools"/"run_h3_flat_model_shadow.py",
    "stage":ROOT/"h3"/"htrc_flat_head_batch.py",
}


class ModelShadowGateContract(unittest.TestCase):
    def test_sources_syntax_and_expected_entrypoints(self):
        for path in SOURCES.values():
            ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
        shadow=SOURCES["shadow"].read_text(encoding="utf-8")
        runner=SOURCES["runner"].read_text(encoding="utf-8")
        self.assertIn("def capture_from_bhsd(",shadow)
        self.assertIn("def run_model_shadow(",shadow)
        self.assertIn("def main():",runner)

    def test_only_true_native_heads_and_no_full_qkv_copy(self):
        shadow=SOURCES["shadow"].read_text(encoding="utf-8")
        self.assertIn('"raw_head_index":list(range(h))',shadow)
        self.assertIn('ids=list(range(pos,end))',shadow)
        self.assertIn('q.detach()[0].permute(1,0,2)',shadow)
        self.assertNotIn("repeat_available_heads",shadow)
        self.assertIn("source_k",SOURCES["stage"].read_text(encoding="utf-8"))

    def test_last_partial_query_tile_uses_valid_rows_for_mu(self):
        stage=SOURCES["stage"].read_text(encoding="utf-8")
        self.assertIn("valid_query_rows=None",stage)
        self.assertIn("if valid_query_rows is not None and valid_query_rows % tile:",stage)
        self.assertIn("MU[:,final_tile,:]",stage)
        shadow=SOURCES["shadow"].read_text(encoding="utf-8")
        self.assertIn("valid_query_rows=valid",shadow)

    def test_original_attention_result_always_returns(self):
        runner=SOURCES["runner"].read_text(encoding="utf-8")
        self.assertIn("original=original_attention(qc,kc,vc,heads_arg,*pos,**kw)",runner)
        self.assertIn("return original",runner)
        self.assertIn("minimax.optimized_attention=original_attention",runner)
        self.assertIn("attention_module.optimized_attention=original_attention",runner)
        self.assertIn('"attention_output_replaced":False',runner)
        self.assertIn("flat_model_resident_shadow.json",runner)

    def test_cli_attributes_are_declared(self):
        tree=ast.parse(SOURCES["runner"].read_text(encoding="utf-8"))
        flags=set()
        used=set()
        for node in ast.walk(tree):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and node.func.attr=="add_argument"
                and isinstance(node.func.value,ast.Name)
                and node.func.value.id=="p"
                and node.args and isinstance(node.args[0],ast.Constant)):
                value=node.args[0].value
                if isinstance(value,str) and value.startswith("--"):
                    flags.add(value[2:].replace("-","_"))
            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=="args":
                used.add(node.attr)
        self.assertEqual(set(),used-flags)


if __name__=="__main__":
    unittest.main()
