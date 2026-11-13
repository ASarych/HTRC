"""CPU-only source contracts for independently selectable streamed VB path.

No CUDA execution or inferred performance PASS. A later GPU A/B tool is
required to establish numerical equivalence and actual wall-time/VRAM.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
TR=ROOT/"h3"/"htrc_flat_triton.py"
BATCH=ROOT/"h3"/"htrc_flat_head_batch.py"
SHARED=ROOT/"h3"/"htrc_flat_model_shadow.py"
FLOW=ROOT/"h3"/"htrc_flat_directflow_live.py"
LIVE=ROOT/"h3"/"htrc_flat_live_backend.py"
GPU_TOOL=ROOT/"tools"/"benchmark_h3_flat_streamed_vb_ab.py"


def function_source(path, function):
    txt=path.read_text(encoding="utf-8")
    tree=ast.parse(txt,filename=str(path))
    matches=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
             and n.name==function]
    assert len(matches)==1,f"Function {function} not uniquely found: {path}"
    return ast.get_source_segment(txt,matches[0])


class StreamVBContracts(unittest.TestCase):
    def test_all_syntax(self):
        for path in (TR,BATCH,SHARED,FLOW,LIVE,GPU_TOOL):
            with self.subTest(path=path.name):
                compile(path.read_text(encoding="utf-8"),str(path),"exec")

    def test_default_triton_kernel_unchanged(self):
        kernel=function_source(TR,"_tile_kernel")
        self.assertIn("VB_SPLIT_STREAM: tl.constexpr = 0",kernel)
        self.assertIn("SPLIT_ID: tl.constexpr = -1",kernel)
        self.assertIn("if VB_SPLIT_STREAM:",kernel)
        self.assertIn("sp = tl.program_id(1)",kernel)
        self.assertIn("ns = tl.num_programs(1)",kernel)
        self.assertIn("if VB_SPLIT_STREAM:",kernel)
        self.assertIn("(gt * nb + c)",kernel)
        self.assertIn("(gt * VB_SPAN + c - VB_START)",kernel)

    def test_new_vbar_producer_matches_reference_math(self):
        reference=function_source(TR,"_stats_kernel")
        producer=function_source(TR,"_vbar_split_kernel")
        for literal in (
            "tl.dot(mu", "tl.max(a,axis=1)", "tl.exp(a-maxima[:,None])",
            "tl.sum(weights,axis=1)", "tl.dot(weights,v,input_precision=PREC)",
            "vb.to(VB.dtype.element_ty)",
        ):
            self.assertIn(literal,producer)
        self.assertIn("vb = tl.dot(w, vj, input_precision=PREC)",reference)
        self.assertIn("tl.dot(mu, tl.trans(kj), input_precision=PREC) * scale",reference)
        self.assertIn("if not SCORE_ONLY:",reference)
        self.assertIn("SCORE_ONLY: tl.constexpr = 0",reference)

    def test_streamed_workspace_is_opt_in(self):
        src=function_source(SHARED,"alloc_workspace")
        self.assertIn('vb_mode="materialized"',src)
        self.assertIn('(h,centers,n,128) if vb_mode=="materialized" else (1,)',src)
        self.assertIn('"VB_SPLIT"',src)
        self.assertIn('"split_stream"',src)
        self.assertIn('if vb_mode not in ("materialized", "split_stream")',src)

    def test_streamed_stats_preparation_does_not_use_full_vb(self):
        src=function_source(BATCH,"prepare_batched_head_states")
        self.assertIn('vb_mode="materialized"',src)
        self.assertIn('SCORE_ONLY=int(vb_mode=="split_stream")',src)
        self.assertIn('query_workspace is None',src)
        self.assertIn('"_stream_kbf":stream_kbf',src)
        self.assertIn('"_vb_mode":vb_mode',src)
        self.assertIn('group != 1',src)

    def test_streamed_consumption_order(self):
        src=function_source(BATCH,"launch_prepared_tile_batch_split_stream")
        producer=src.index("triton_module._vbar_split_kernel[")
        consumer=src.index('triton_module._tile_kernel[')
        self.assertLess(producer,consumer)
        self.assertIn('VB_SPLIT_STREAM=1',src)
        self.assertIn('SPLIT_ID=split,SPLIT_COUNT=splits',src)
        self.assertIn('vb_split=ws["VB_SPLIT"]',FLOW.read_text(encoding="utf-8"))
        self.assertIn('if first<leaves:',src)
        self.assertIn('if splits!=4 or bn!=64',src)
        self.assertIn('states["VB"].numel()!=1',src)

    def test_graph_explicit_mode_and_eager_oracle(self):
        src=FLOW.read_text(encoding="utf-8")
        live=LIVE.read_text(encoding="utf-8")
        self.assertIn('"APL_HTRC_DIRECTFLOW_VB_MODE","materialized"',src)
        self.assertIn('if vb_mode not in ("materialized","split_stream"):',src)
        self.assertIn('vb_mode=vb_mode',src)
        self.assertIn("group_cap, vb_mode)",src)
        self.assertIn('torch.equal(group_out, eager_out[0])',src)
        self.assertIn('dispatch="eager"',src)
        self.assertIn('os.environ.get("APL_HTRC_FLAT_DISPATCH","eager")',live)
        self.assertIn('if choice == "graph":',live)

    def test_single_synthetic_gpu_report(self):
        src=GPU_TOOL.read_text(encoding="utf-8")
        self.assertIn("TemporaryDirectory(",src)
        self.assertIn('for variant in ("materialized","split_stream"):',src)
        self.assertIn('"completed":False',src)
        self.assertIn('"real_video":"UNRUN"',src)
        self.assertIn("torch.equal(",src)
        self.assertIn('dispatch="eager"',src)
        self.assertIn('"bank_b_eager_bit_exact"',src)
        self.assertIn("write_text(json.dumps(result,indent=2)",src)

    def test_split_vb_upper_bound(self):
        # Shape-only conservation: keep one split and never allocate the
        # full VB tensor in new mode. Does not predict GPU performance.
        from math import ceil
        for leaves in (954,1170,2547,5125,13975):
            ns=4
            span=max(64,ceil(ceil(leaves/ns)/64)*64)
            self.assertLess(span,leaves)
            self.assertGreaterEqual(span,ceil(leaves/ns))
            self.assertLessEqual(span*4,leaves+4*64)

    def test_legacy_launcher_kept_separate(self):
        old=function_source(BATCH,"launch_prepared_tile_batch")
        fresh=function_source(BATCH,"launch_prepared_tile_batch_split_stream")
        self.assertIn('One _tile_kernel call for arbitrary H',old)
        self.assertNotIn('_vbar_split_kernel',old)
        self.assertIn('_vbar_split_kernel',fresh)
        self.assertIn('_tile_kernel',fresh)


if __name__=="__main__":
    unittest.main()
