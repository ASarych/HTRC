"""CPU-only static contract for opt-in native Q/K/V paths (no CUDA required)."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
TR=ROOT/"h3"/"htrc_flat_triton.py"
PREP=ROOT/"h3"/"htrc_flat_head_batch.py"
LIVE=ROOT/"h3"/"htrc_flat_live_backend.py"
BENCH=ROOT/"tools"/"benchmark_h3_flat_direct_qkv_ab.py"


class DirectQKVContractTest(unittest.TestCase):
    def test_python_syntax(self):
        for path in (TR,PREP,LIVE,BENCH):
            with self.subTest(path=str(path)):
                ast.parse(path.read_text(encoding="utf-8"),filename=str(path))

    def test_direct_triton_kernels(self):
        tree=ast.parse(TR.read_text(encoding="utf-8"))
        functions={node.name:node for node in tree.body
                   if isinstance(node,ast.FunctionDef)}
        for name in ("_direct_q_mu_kernel","_quant_key_int8_native",
                     "_stats_kernel","_rowmax_kernel",
                     "_tile_kernel","_merge_tail_kernel"):
            self.assertIn(name,functions)
        self.assertIn("tl.div_rn",TR.read_text(encoding="utf-8"))
        self.assertIn("cvt.rni.s32.f32",TR.read_text(encoding="utf-8"))
        self.assertIn("if DIRECT_KV:",TR.read_text(encoding="utf-8"))
        self.assertIn("if V_DIRECT:",TR.read_text(encoding="utf-8"))

    def test_opt_in_and_no_math_switch(self):
        prep=PREP.read_text(encoding="utf-8")
        live=LIVE.read_text(encoding="utf-8")
        self.assertIn('query_staging="gather"',prep)
        self.assertIn('native_direct=False',prep)
        self.assertIn('kv_staging="gather"',live)
        self.assertIn('query_staging="gather"',live)
        self.assertIn('cfg=shared.flat_config(mode,tiles)',live)
        self.assertIn('"quality_coefficient_used":False',live)
        self.assertIn('Q=gather("q_qhd",n_query)',prep)
        self.assertIn('KBF',prep)
        self.assertIn('VBF',prep)
        self.assertIn('ids=torch.topk(sel,kx',prep)
        self.assertIn('excl.scatter_(1,ids,1)',prep)
        self.assertNotIn('quality_coefficient',ast.unparse(
            next(node for node in ast.parse(live).body
                 if isinstance(node,ast.FunctionDef)
                 and node.name=="flat_attention_bthd"
            ).args
        ))

    def test_optional_gpu_stage_diagnostics_do_not_change_default(self):
        prep=PREP.read_text(encoding="utf-8")
        bench=BENCH.read_text(encoding="utf-8")
        self.assertIn('stage_gpu_events=None',prep)
        self.assertIn('if stage_gpu_events is not None:',prep)
        self.assertIn('stage_gpu_events.append((name,stage_gpu_prev,stage_finish))',prep)
        self.assertIn('--diagnose-only',bench)
        self.assertIn('def diagnose_stages(args,report):',bench)
        self.assertIn('"numerical_parity":"NOT_REPEATED_IN_STAGE_DIAGNOSTIC"',bench)
        self.assertIn('GPU-stage-intervals=',bench)

    def test_benchmark_four_arms_and_one_json(self):
        s=BENCH.read_text(encoding="utf-8")
        for name in ("baseline","direct_q","direct_kv","direct_qkv"):
            self.assertIn('"'+name+'"',s)
        self.assertIn("torch.equal(packed[\"K\"],native[\"K\"])",s)
        self.assertIn("torch.equal(packed[\"KS\"],native[\"KS\"])",s)
        self.assertIn('"real_model":False',s)
        self.assertIn('flat_direct_qkv_ab.json',s)
        self.assertNotIn('repeat_available_heads',s)


if __name__=="__main__":
    unittest.main()
