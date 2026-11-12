"""CPU-only no-GPU checks for selected real-model Flat (not shadow) semantics."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
BACKEND=ROOT/"h3"/"htrc_flat_live_backend.py"
SHADOW=ROOT/"h3"/"htrc_flat_model_shadow.py"
PREP=ROOT/"h3"/"htrc_flat_head_batch.py"


class FlatLiveContract(unittest.TestCase):
    def test_syntax(self):
        for file in (BACKEND,SHADOW,PREP):
            ast.parse(file.read_text(encoding="utf-8"),filename=str(file))

    def test_real_attention_output_not_shadow(self):
        src=BACKEND.read_text(encoding="utf-8")
        self.assertIn("def flat_attention_bthd(",src)
        self.assertIn("out=torch.empty((s,count,128),device=dev,dtype=q.dtype)",src)
        self.assertIn("return out.unsqueeze(0)",src)
        self.assertNotIn("return original",src)
        self.assertIn("out[qi:qe,begin:end,:].copy_(",src)
        self.assertIn("if torch.is_grad_enabled():",src)

    def test_full_q_and_unique_heads_only(self):
        src=BACKEND.read_text(encoding="utf-8")
        self.assertIn("ids=list(range(begin,end))",src)
        self.assertIn("for window in windows.iter_query_windows(",src)
        self.assertIn("tile_count=active_tiles",src)
        self.assertIn("windows.compact_workspace_view(",src)
        self.assertIn("valid_query_rows=valid",src)
        self.assertIn("tails=shared._tail(cap,ids)",src)
        self.assertIn("kv_cache=cache,query_workspace=active_ws",src)
        self.assertNotIn("repeat_available_heads",src)

    def test_release_per_window_and_per_group(self):
        src=BACKEND.read_text(encoding="utf-8")
        self.assertIn("del selected,merged,parts,states,view,src,active_ws",src)
        self.assertIn("del cache,ws,tails",src)
        self.assertIn("_head_plan(",src)
        self.assertIn("budget_mib=int(budget_mib)",src)

    def test_preserve_source_triton_and_no_global_hook(self):
        src=BACKEND.read_text(encoding="utf-8")
        self.assertIn("_apl_htrc_external_live_flat_triton",src)
        self.assertIn('Path(__file__).with_name("htrc_flat_triton.py")',src)
        self.assertNotIn("optimized_attention =",src)
        self.assertNotIn("torch.cuda.synchronize(",src)

if __name__=="__main__":
    unittest.main()
