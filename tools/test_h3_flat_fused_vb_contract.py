"""No-GPU contracts for optional in-Tile Vbar fusion.

These tests parse code and check dispatch/memory gates. They DO NOT run
Triton, prove bitwise CUDA parity, or establish speedup.
"""
from __future__ import annotations
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
PATHS={
    "triton": ROOT/"h3"/"htrc_flat_triton.py",
    "batch": ROOT/"h3"/"htrc_flat_head_batch.py",
    "shared": ROOT/"h3"/"htrc_flat_model_shadow.py",
    "graph": ROOT/"h3"/"htrc_flat_directflow_live.py",
    "eager": ROOT/"h3"/"htrc_flat_live_backend.py",
    "benchmark": ROOT/"tools"/"benchmark_h3_flat_fused_vb_ab.py",
}

def read_fn(name,fn):
    src=PATHS[name].read_text(encoding="utf-8")
    tree=ast.parse(src)
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                 and node.name==fn]
    if len(definitions)!=1:
        raise AssertionError(f"Expected exactly one {fn} in {name}")
    return ast.get_source_segment(src,definitions[0])

class TestFusedVBContracts(unittest.TestCase):
    def test_syntax_all(self):
        for key,path in PATHS.items():
            with self.subTest(path=key):
                compile(path.read_text(encoding="utf-8"),str(path),"exec")

    def test_compiled_old_kernel_defaults_retained(self):
        src=read_fn("triton","_tile_kernel")
        self.assertIn("VB_SPLIT_STREAM: tl.constexpr = 0",src)
        self.assertIn("FUSED_VB: tl.constexpr = 0",src)
        self.assertIn("if FUSED_VB:",src)
        self.assertIn("elif VB_SPLIT_STREAM:",src)
        self.assertIn("VB + (gt * nb + c)",src)
        self.assertIn("sp = tl.program_id(1)",src)
        self.assertIn("KBF = None",src)

    def test_fused_helper_has_exact_reference_dot_geometry(self):
        src=read_fn("triton","_fused_vbar_one_leaf")
        old=read_fn("triton","_stats_kernel")
        self.assertIn("tl.broadcast_to(MU_VEC[None,:],(LEAF,D))",src)
        self.assertIn("tl.dot(mu_matrix,tl.trans(k),input_precision=PREC)*scale",src)
        self.assertIn("weights=tl.exp(scores-maximum)",src)
        self.assertIn("weights=weights/tl.sum(weights,axis=0)",src)
        self.assertIn("tl.dot(w_matrix,v,input_precision=PREC)",src)
        self.assertIn("return result.to(tl.bfloat16)",src)
        self.assertIn("vb = tl.dot(w, vj, input_precision=PREC)",old)

    def test_fused_helper_uses_native_kv_not_int8_approx(self):
        src=read_fn("triton","_fused_vbar_one_leaf")
        self.assertIn("KBF + head*KB_SH",src)
        self.assertIn("VBF + head*V_SH",src)
        self.assertNotIn("K8",src)
        self.assertNotIn("KS",src)
        self.assertNotIn("tl.store",src)

    def test_stats_still_runs_original_fixed_ranking(self):
        src=read_fn("batch","prepare_batched_head_states")
        self.assertIn('SCORE_ONLY=int(vb_mode in ("split_stream","fused_vb"))',src)
        self.assertIn("ids=torch.topk(sel,kx,dim=-1,sorted=False).indices",src)
        self.assertIn("excl.scatter_(1,ids,1)",src)
        self.assertIn('"_stream_kbf":stream_kbf',src)

    def test_no_vb_full_or_split_buffer_in_fused_mode(self):
        src=read_fn("shared","alloc_workspace")
        self.assertIn('vb_mode not in ("materialized", "split_stream", "fused_vb")',src)
        self.assertIn('(h,centers,n,128) if vb_mode=="materialized" else (1,)',src)
        self.assertIn('if vb_mode=="split_stream" else None',src)
        self.assertIn('"VB_SPLIT":vb_split',src)

    def test_tile_fused_requires_explicit_flag_and_native_kv(self):
        src=read_fn("batch","launch_prepared_tile_batch")
        self.assertIn("fused_vb=False",src)
        self.assertIn('states.get("_vb_mode")!="fused_vb"',src)
        self.assertIn("not direct or not direct_v",src)
        self.assertIn('"FUSED_VB":1,"KBF":kbf',src)
        self.assertIn("**vb_kernel_kwargs",src)
        self.assertIn("states[\"VB\"].numel()!=1",src)

    def test_old_split_stream_entrypoint_still_exists(self):
        src=read_fn("batch","launch_prepared_tile_batch_split_stream")
        self.assertIn('states.get("_vb_mode")!="split_stream"',src)
        self.assertIn("VB_SPLIT_STREAM=1",src)
        self.assertIn("triton_module._vbar_split_kernel[",src)

    def test_three_modes_explicit_no_fallback(self):
        src=read_fn("graph","graph_attention_bthd")
        self.assertIn('("materialized","split_stream","fused_vb")',src)
        self.assertIn('"APL_HTRC_DIRECTFLOW_VB_MODE","materialized"',src)
        self.assertIn("vb_mode",src)
        window=read_fn("graph","_one_window")
        self.assertIn('elif vb_mode == "fused_vb":',window)
        self.assertIn('if vb_mode == "split_stream":',window)
        self.assertIn("launch_prepared_tile_batch_fused_vb",window)
        self.assertIn("launch_prepared_tile_batch_split_stream",window)

    def test_real_graph_parity_still_checks_old_eager(self):
        src=read_fn("graph","graph_attention_bthd")
        self.assertIn("torch.equal(group_out, eager_out[0])",src)
        self.assertIn('dispatch="eager"',src)
        self.assertIn("extra_reserved > max_extra*_MIB",src)
        self.assertIn("free < min_free*_MIB",src)

    def test_eager_dispatch_default_untouched(self):
        src=PATHS["eager"].read_text(encoding="utf-8")
        self.assertIn('os.environ.get("APL_HTRC_FLAT_DISPATCH","eager")',src)

    def test_benchmark_has_one_report_two_fresh_banks(self):
        src=PATHS["benchmark"].read_text(encoding="utf-8")
        self.assertIn('for variant in ("materialized","fused_vb")',src)
        self.assertIn('bank_b_eager_bit_exact',src)
        self.assertIn('flat_fused_vb_ab.json',src)
        self.assertIn('write_text(json.dumps(result,indent=2)',src)
        self.assertIn('"real_video":"UNRUN"',src)
        self.assertIn('"completed":False',src)

    def test_no_hidden_q_coefficient(self):
        src=read_fn("graph","graph_attention_bthd")
        self.assertIn('expected_leaves = {"fast": 256, "balanced": 512}',src)
        self.assertIn('int(cfg.get("kern", {}).get("group", 1)) != 1',src)
        self.assertNotIn('quality_coefficient=0.3',src)

if __name__=="__main__":
    unittest.main()
