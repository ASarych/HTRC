"""CPU structural/native-H3 projection tests; no ComfyUI checkpoint needed.

These tests verify fused QKV slicing and RoPE position alignment using
a deterministic fake injected kernel. They do NOT validate Comfy's CUDA
rms_rope kernel nor packed NVFP4 weights; those require model GPU canary.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from h3 import htrc_flat_h3_native_projection as native

FLOW=ROOT/"h3"/"htrc_flat_directflow_live.py"
NATIVE=ROOT/"h3"/"htrc_flat_h3_native_projection.py"


class FakeNorm:
    def __init__(self, scale):
        self.weight=torch.ones(128, dtype=torch.bfloat16)*scale
        self.eps=1e-5

    def __call__(self, x):
        return x * self.weight


class FakeH3:
    def __init__(self, heads=4, width=19):
        self.heads=heads
        self.head_dim=128
        self.training=False
        self.q_norm=FakeNorm(0.75)
        self.k_norm=FakeNorm(1.25)
        self.weight=torch.randn(3*heads*128,width,dtype=torch.bfloat16)
        self.projector_calls=0

    def qkv_proj(self,x):
        self.projector_calls += 1
        # CPU fixture is deliberately rowwise so a batch-size-dependent
        # BLAS kernel cannot introduce spurious BF16 rounding differences.
        return torch.cat([
            torch.nn.functional.linear(row.unsqueeze(0),self.weight)
            for row in x
        ],dim=0)


def fake_rope_inplace(q,k,rope,qw,kw,*,epsilon,rot_dim):
    assert rot_dim == 8
    # Deterministic mock that encodes the absolute row position, so
    # improperly passing an unsliced RoPE table would fail these tests.
    position=rope[:, :, :, 0, 0, 0].unsqueeze(-1)
    q.mul_(qw.view(1,1,1,128)).add_(position)
    k.mul_(kw.view(1,1,1,128)).sub_(position)


class NativeQKVContractTests(unittest.TestCase):
    def test_sampled_head_chunks_match_whole_projection_and_absolute_rope(self):
        with torch.inference_mode():
            torch.manual_seed(12)
            model=FakeH3()
            x=torch.randn(21,19,dtype=torch.bfloat16)
            # Native table shape [1,S,1,rot/2,2,2].
            rope=torch.zeros((1,21,1,4,2,2),dtype=torch.bfloat16)
            rope[:, :, 0, 0, 0, 0]=torch.arange(21,dtype=torch.bfloat16)
            qw=model.q_norm.weight
            kw=model.k_norm.weight

            whole=model.qkv_proj(x)
            ref=native.normalized_native_group_chunk(
                model,whole,rope,begin=1,end=3,
                qw=qw,kw=kw,rms_rope_inplace=fake_rope_inplace)
            projected=[[],[],[]]
            for start in range(0,21,5):
                stop=min(start+5,21)
                fused=model.qkv_proj(x[start:stop])
                parts=native.normalized_native_group_chunk(
                    model,fused,rope[:, start:stop],begin=1,end=3,
                    qw=qw,kw=kw,rms_rope_inplace=fake_rope_inplace)
                for i,p in enumerate(parts):
                    projected[i].append(p.clone())
            for i in range(3):
                with self.subTest(qkv=i):
                    self.assertTrue(torch.equal(torch.cat(projected[i]),ref[i]))

    def test_without_rope_uses_native_norm_modules(self):
        with torch.inference_mode():
            model=FakeH3(heads=2)
            x=torch.randn(17,19,dtype=torch.bfloat16)
            fused=model.qkv_proj(x)
            q,k,v=native.normalized_native_group_chunk(
                model,fused,None,begin=0,end=2)
            q0,k0,v0=fused.split(2*128,dim=-1)
            self.assertTrue(torch.equal(q,model.q_norm(q0.view(17,2,128))))
            self.assertTrue(torch.equal(k,model.k_norm(k0.view(17,2,128))))
            self.assertTrue(torch.equal(v,v0.view(17,2,128)))

    def test_group_Q_is_fresh_per_projection_and_padding_correct(self):
        with torch.inference_mode():
            model=FakeH3(heads=4)
            x=torch.randn(19,19,dtype=torch.bfloat16)
            prod=native.NativeGroupProducer(model,x,None,0,2,chunk_rows=5)
            k=torch.empty((1,2,19,128),dtype=torch.bfloat16)
            v=torch.empty_like(k)
            with self.assertRaises(RuntimeError):
                prod.fill_q(0,4,torch.empty((2,4,128),dtype=torch.bfloat16))
            # CPU-test only: inject the validated pure native chunk primitive.
            def inject(attn,source,rope_freqs,*,begin,end,q_out,k_out,v_out,chunk_rows):
                for start in range(0,source.shape[0],chunk_rows):
                    stop=min(source.shape[0],start+chunk_rows)
                    fused=attn.qkv_proj(source[start:stop])
                    qpart,kpart,vpart=native.normalized_native_group_chunk(
                        attn,fused,None,begin=begin,end=end)
                    q_out[0,:,start:stop].copy_(qpart.permute(1,0,2))
                    k_out[0,:,start:stop].copy_(kpart.permute(1,0,2))
                    v_out[0,:,start:stop].copy_(vpart.permute(1,0,2))
                return {"projection_chunks":4,"projection_rows":19,
                        "head_range":[0,2]}
            with mock.patch.object(native,"project_group_native",side_effect=inject):
                prod.fill_kv(k,v)
                stage=torch.empty((2,8,128),dtype=torch.bfloat16)
                prod.fill_q(14,19,stage)
                self.assertTrue(torch.equal(
                    stage[:,:5],prod.q[0,:,14:19]))
                self.assertTrue(torch.equal(
                    stage[:,5:],stage[:,4:5].expand(-1,3,-1)))
                earlier=prod.q.clone()
                prod.x=x+0.5
                prod.fill_kv(k,v)
                self.assertFalse(torch.equal(earlier,prod.q))

    def test_original_comfy_projector_not_bypassed(self):
        source=NATIVE.read_text(encoding="utf8")
        flow=FLOW.read_text(encoding="utf8")
        ast.parse(source,filename=str(NATIVE))
        ast.parse(flow,filename=str(FLOW))
        self.assertIn("fused = attn.qkv_proj(x[start:stop])",source)
        self.assertIn("quant_ops.ck.rms_rope_split_half_",source)
        self.assertIn("mm.cast_to(attn.q_norm.weight, device=x.device)",source)
        self.assertIn("mm.cast_to(attn.k_norm.weight, device=x.device)",source)
        self.assertNotIn("attn.qkv_proj.weight",source)
        self.assertNotIn(".dequantize(",source)
        self.assertIn('research_source_kind="native_h3"',source)
        self.assertIn('if kv_provider is not None and group_cap != total_heads:',flow)
        self.assertIn('research_source_kind == "native_h3"',flow)
        self.assertIn("extra_reserved > max_extra*_MIB",flow)
        self.assertIn("if not exact and not compare_only:",flow)


if __name__=="__main__":
    unittest.main()
