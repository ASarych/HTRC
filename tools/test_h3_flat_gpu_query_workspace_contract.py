"""CPU-only static gate for exact query-workspace A/B wiring (no CUDA)."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/"h3"/"htrc_flat_head_batch.py"
EVAL=ROOT/"tools"/"benchmark_h3_flat_full_pipeline_batch.py"
BENCH=ROOT/"tools"/"benchmark_h3_flat_gpu_query_workspace.py"

class QueryWorkspaceContractTest(unittest.TestCase):
    def test_all_sources_parse_and_have_required_functions(self):
        for path in (STAGE,EVAL,BENCH):
            tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
            self.assertTrue(tree.body)
        src=STAGE.read_text(encoding="utf-8")
        self.assertIn("def gpu_query_workspace_bytes(",src)
        self.assertIn("query_workspace=None",src)

    def test_recompute_every_query_window_without_stale_exclusion(self):
        src=STAGE.read_text(encoding="utf-8")
        self.assertIn('G=state_buffer("G"',src)
        self.assertIn('VB=state_buffer("VB"',src)
        self.assertIn('LZ=state_buffer("LZ"',src)
        self.assertIn('SG=state_buffer("SG"',src)
        self.assertIn("ws=query_workspace,sg=sgf",src)
        self.assertIn('idx=state_buffer("IDX"',src)
        self.assertIn('excl=state_buffer("EXCL"',src)
        self.assertLess(src.index("excl.zero_()"),src.index("excl.scatter_(1,ids,1)"))
        self.assertIn("triton_module._stats_kernel[",src)
        self.assertIn("ids=torch.topk(",src)

    def test_both_paths_share_kv_cache_but_only_one_uses_workspace(self):
        src=BENCH.read_text(encoding="utf-8")
        self.assertIn('MODES = ("fresh", "workspace")',src)
        self.assertIn("kv_cache=cache,",src)
        self.assertIn('if which == "workspace" else None',src)
        self.assertIn('query_workspace=allocate_query_workspace',src)
        self.assertIn('"fresh_with_common_kv_build_ms"',src)
        self.assertIn('"workspace_with_common_kv_build_and_setup_ms"',src)
        self.assertIn('flat_gpu_query_workspace_ab_',src)
        self.assertIn('subprocess.Popen(',src)

    def test_argparse_names_match_accesses(self):
        tree=ast.parse(BENCH.read_text(encoding="utf-8"))
        declared=set()
        used=set()
        for node in ast.walk(tree):
            if (isinstance(node,ast.Call)
                and isinstance(node.func,ast.Attribute)
                and node.func.attr=="add_argument"
                and isinstance(node.func.value,ast.Name)
                and node.func.value.id=="ap"
                and node.args and isinstance(node.args[0],ast.Constant)):
                flag=node.args[0].value
                if isinstance(flag,str) and flag.startswith("--"):
                    declared.add(flag[2:].replace("-","_"))
            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=="args":
                used.add(node.attr)
        self.assertEqual(set(),used-declared)

if __name__=="__main__":
    unittest.main()
