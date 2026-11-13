"""No-GPU functional gate for the actual one-pass H3 compact activation buffer."""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "h3" / "htrc_flat_h3_compact_once.py"


def _load():
    spec = importlib.util.spec_from_file_location("htrc_compact_cpu_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeH3:
    def __init__(self, heads=4):
        self.heads = heads
        self.head_dim = 128
        self.calls = 0

    def qkv_proj(self, x):
        self.calls += 1
        return torch.cat((x, x * 2, x * -0.7), dim=-1)


def _normalizer(model):
    def apply(fused, rope):
        assert rope is None
        return tuple(
            part.view(-1, model.heads, 128)
            for part in fused.split(model.heads * 128, dim=-1)
        )
    return apply


class CompactAllHeadsTests(unittest.TestCase):
    def test_one_pass_per_chunk_not_per_head_group(self):
        m = _load()
        model = FakeH3()
        torch.manual_seed(42)
        x = torch.randn((133, 512), dtype=torch.bfloat16)
        with torch.no_grad():
            stage = m.CompactAllHeadsQKV(
                model, x, None, chunk_rows=32,
                normalize_chunk=_normalizer(model))
            self.assertEqual(model.calls, math.ceil(133/32))
            self.assertEqual(stage.packed.shape, (3,4,133,128))
            self.assertEqual(stage.scales.shape, (3,4,133))
            self.assertEqual(stage.projection_chunks, 5)
            self.assertEqual(stage.total_projection_rows, 133)
            self.assertEqual(stage.persistent_bytes, 3*4*133*(128+4))
            self.assertEqual(stage.summary()["full_fused_qkv_passes_per_attention"], 1)
            for first, last in ((0,2),(2,4)):
                group=stage.group(first,last)
                k=torch.empty((1,2,133,128),dtype=torch.bfloat16)
                v=torch.empty_like(k)
                group.fill_kv(k,v)
                expect_k=(x*2).view(133,4,128)[:,first:last].transpose(0,1)
                expect_v=(x*-0.7).view(133,4,128)[:,first:last].transpose(0,1)
                self.assertLess((k[0]-expect_k).float().abs().mean().item(),0.02)
                self.assertLess((v[0]-expect_v).float().abs().mean().item(),0.02)
                q=torch.empty((2,23,128),dtype=torch.bfloat16)
                group.fill_q(119,133,q)
                expect_q=x[119:133].view(14,4,128)[:,first:last].transpose(0,1)
                self.assertLess((q[:,:14]-expect_q).float().abs().mean().item(),0.02)
                self.assertTrue(torch.equal(q[:,14:,:],
                    q[:,13:14,:].expand(-1,9,-1)))
            self.assertEqual(model.calls,5,"Graph providers must not reproject QKV")

    def test_new_forward_refreshes_but_single_stage_does_not_reproject(self):
        m=_load()
        model=FakeH3()
        x=torch.ones((35,512),dtype=torch.bfloat16)
        with torch.no_grad():
            first=m.CompactAllHeadsQKV(model,x,None,chunk_rows=16,
                                        normalize_chunk=_normalizer(model))
            prev=model.calls
            first.group(0,2).fill_kv(
                torch.empty((1,2,35,128),dtype=torch.bfloat16),
                torch.empty((1,2,35,128),dtype=torch.bfloat16))
            self.assertEqual(model.calls,prev)
            second=m.CompactAllHeadsQKV(model,x*2,None,chunk_rows=16,
                                         normalize_chunk=_normalizer(model))
            self.assertEqual(model.calls,prev+3)
            self.assertFalse(torch.equal(first.scales,second.scales))
            self.assertNotEqual(first.scales.data_ptr(),second.scales.data_ptr())

    def test_graph_wires_existing_single_stage_and_keeps_legacy_default(self):
        import ast
        native_file = ROOT / "h3" / "htrc_flat_h3_native_projection.py"
        source = native_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(native_file))
        graph = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "graph_attention_native_h3"
        )
        args = list(graph.args.kwonlyargs)
        defaults = graph.args.kw_defaults
        strategy_index = [arg.arg for arg in args].index("projection_strategy")
        self.assertEqual(
            ast.literal_eval(defaults[strategy_index]), "repeated_group")
        self.assertIn('build_compact_onepass_stage(attn, x, rope_freqs,', source)
        self.assertIn('"compact_int8_onepass"', source)
        self.assertIn('compact.group(begin, end) if compact is not None', source)
        self.assertIn('query_provider=producer.fill_q, kv_provider=producer.fill_kv', source)
        self.assertIn('"full_fused_qkv_repeated_per_head_group"', source)
        self.assertIn('1 if compact is not None else heads//group_heads', source)

    def test_fast_pack_opt_in_refuses_cpu_and_unknown_modes(self):
        m=_load()
        model=FakeH3()
        x=torch.ones((16,512),dtype=torch.bfloat16)
        with torch.no_grad(),patch.dict(
            os.environ,{"APL_HTRC_QKV_PACK_BACKEND":"triton"}):
            with self.assertRaisesRegex(RuntimeError,"live CUDA native in-place"):
                m.CompactAllHeadsQKV(
                    model,x,None,chunk_rows=8,
                    normalize_chunk=_normalizer(model))
        self.assertEqual(model.calls,0)
        with torch.no_grad(),patch.dict(
            os.environ,{"APL_HTRC_QKV_PACK_BACKEND":"unsupported"}):
            with self.assertRaisesRegex(ValueError,"Unknown compact"):
                m.CompactAllHeadsQKV(
                    model,x,None,chunk_rows=8,
                    normalize_chunk=_normalizer(model))
        self.assertEqual(model.calls,0)

    def test_triton_source_packs_all_qkv_without_fp32_activation_buffers(self):
        import ast
        fast_file=ROOT/"h3"/"htrc_flat_h3_pack_triton.py"
        fast=fast_file.read_text(encoding="utf-8")
        tree=ast.parse(fast,filename=str(fast_file))
        funcs={node.name:node for node in tree.body
               if isinstance(node,ast.FunctionDef)}
        self.assertIn("_pack_fused_qkv",funcs)
        self.assertIn("pack_normalized_fused",funcs)
        for token in (
            "tl.max(tl.abs(value), 2)",
            "libdevice.nearbyint(value / scale[:, :, None])",
            "tl.store(PACKED + output_offset, quant",
            "SCALES + scale_offset, scale",
            "triton.cdiv(rows, ROWS_PER_CTA)",
            "triton.cdiv(heads, HEADS_PER_CTA)",
        ):
            with self.subTest(token=token):
                self.assertIn(token,fast)
        self.assertNotIn("torch.empty(",fast)
        self.assertIn("from triton.language.extra import libdevice",fast)
        self.assertNotIn("tl.round(",fast)
        native=(ROOT/"h3"/"htrc_flat_h3_compact_once.py").read_text(encoding="utf-8")
        for token in (
            'APL_HTRC_QKV_PACK_BACKEND',
            'if self.pack_backend == "triton"',
            "from .htrc_flat_h3_pack_triton import pack_normalized_fused",
            "fused.data_ptr() + i * part_size",
            "pack_normalized_fused(",
            '"activation_packing_backend": self.pack_backend',
        ):
            with self.subTest(token=token):
                self.assertIn(token,native)
        self.assertIn('APL_HTRC_QKV_PACK_BACKEND", "torch"',native)
        self.assertIn("if pack_normalized_fused is None:",native)

    def test_admission_and_structure_fail_closed_before_projection(self):
        m=_load()
        model=FakeH3()
        x=torch.empty((4096,512),dtype=torch.bfloat16)
        with torch.no_grad(),patch.dict(os.environ,{"APL_HTRC_COMPACT_STAGE_CAP_MIB":"1"}):
            with self.assertRaisesRegex(RuntimeError,"stage requires"):
                m.CompactAllHeadsQKV(model,x,None,chunk_rows=4096,
                                      normalize_chunk=_normalizer(model))
        self.assertEqual(model.calls,0)
        with torch.no_grad():
            stage=m.CompactAllHeadsQKV(model,x[:16],None,chunk_rows=8,
                                       normalize_chunk=_normalizer(model))
            with self.assertRaises(ValueError):stage.group(0,5)
            with self.assertRaises(ValueError):stage.group(3,3)
            group=stage.group(0,2)
            with self.assertRaises(ValueError):
                group.fill_q(0,16,torch.empty((2,10,128),dtype=torch.bfloat16))
            with self.assertRaises(ValueError):
                group.fill_kv(torch.empty((1,1,16,128),dtype=torch.bfloat16),
                              torch.empty((1,2,16,128),dtype=torch.bfloat16))


if __name__ == "__main__":
    unittest.main()
