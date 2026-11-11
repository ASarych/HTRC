"""CPU-only regression for the GPU K/V reuse benchmark's argparse child CLI.

The single-JSON parent delegates each head count to a fresh subprocess.
This test catches misnamed argparse attributes before the CUDA benchmark.
"""
import ast
from pathlib import Path
import unittest

SCRIPT = Path(__file__).with_name("benchmark_h3_flat_gpu_kv_reuse.py")


class KvReuseCliContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))

    def test_every_args_attribute_is_defined_by_argparse(self):
        declared = set()
        used = set()
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "ap"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                flag = node.args[0].value
                self.assertTrue(flag.startswith("--"), flag)
                name = flag[2:].replace("-", "_")
                for keyword in node.keywords:
                    if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                        name = keyword.value.value
                declared.add(name)
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "args"):
                used.add(node.attr)
        self.assertTrue(declared)
        self.assertEqual(set(), used - declared,
                         "Benchmark refers to argparse fields never declared")

    def test_child_flag_and_single_case_lookup_match(self):
        code = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--_case-heads"', code)
        self.assertIn('"--_case-heads", str(count)', code)
        self.assertIn("count = args._case_heads", code)
        self.assertNotIn("args.case_heads", code)


if __name__ == "__main__":
    unittest.main()
