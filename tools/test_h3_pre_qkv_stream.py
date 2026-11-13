"""CPU-only pre-QKV projection contract; no real H3 or CUDA proof."""
from __future__ import annotations

import ast
import contextlib
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3 import htrc_flat_pre_qkv_stream as pre

FLOW = ROOT / "h3" / "htrc_flat_directflow_live.py"
RUNNER = ROOT / "tools" / "benchmark_h3_pre_qkv_stream_ab.py"


class PreQKVStreamTests(unittest.TestCase):
    def test_head_shards_and_row_chunks_match_full_dense_projection(self):
        with torch.inference_mode():
            for n, heads, group_size, chunk in (
                (19, 3, 2, 5), (33, 4, 2, 8), (37, 5, 3, 16),
            ):
                torch.manual_seed(n)
                x = torch.randn((n, 23), dtype=torch.bfloat16)
                weight = torch.randn((3 * heads * 128, 23),
                                     dtype=torch.bfloat16)
                bias = torch.randn((3 * heads * 128,), dtype=torch.bfloat16)
                pre.validate_projection(x, weight, bias, heads)
                reference = F.linear(x, weight, bias).view(n, 3, heads, 128)
                for begin in range(0, heads, group_size):
                    end = min(heads, begin+group_size)
                    h = end-begin
                    wq, wkv, bq, bkv = pre.group_weights(
                        weight, bias, heads, begin, end)
                    k, v = pre.project_kv_chunks(x, wkv, bkv, h, chunk)
                    with self.subTest(n=n, begin=begin, tensor="K"):
                        self.assertTrue(torch.equal(
                            k[0].permute(1, 0, 2),
                            reference[:, 1, begin:end, :]))
                    with self.subTest(n=n, begin=begin, tensor="V"):
                        self.assertTrue(torch.equal(
                            v[0].permute(1, 0, 2),
                            reference[:, 2, begin:end, :]))
                    qfill = pre.query_window_provider(x, wq, bq, h)
                    for a, b in ((0, min(16, n)), (n-1, n)):
                        pad = b-a+3
                        stage = torch.empty((h, pad, 128), dtype=x.dtype)
                        qfill(a, b, stage)
                        got = stage[:, :b-a].permute(1, 0, 2)
                        expected = reference[a:b, 0, begin:end, :]
                        with self.subTest(n=n, begin=begin, window=(a,b)):
                            self.assertTrue(torch.equal(got, expected))
                            self.assertTrue(torch.equal(
                                stage[:, b-a:, :],
                                stage[:, b-a-1:b-a, :].expand(-1, 3, -1)))

    def test_reject_incompatible_projection_or_stage(self):
        with torch.inference_mode():
            x = torch.zeros((20, 16), dtype=torch.bfloat16)
            w = torch.zeros((3 * 4 * 128, 16), dtype=torch.bfloat16)
            pre.validate_projection(x, w, None, 4)
            with self.assertRaises(ValueError):
                pre.validate_projection(x, w, None, 5)
            with self.assertRaises(ValueError):
                pre.validate_projection(x, w.float(), None, 4)
            with self.assertRaises(ValueError):
                pre.group_weights(w, None, 4, 3, 5)
            _, wkv, _, bkv = pre.group_weights(w, None, 4, 0, 2)
            with self.assertRaises(ValueError):
                pre.project_kv_chunks(x, wkv, bkv, 2, 0)
            wq, _, bq, _ = pre.group_weights(w, None, 4, 0, 2)
            fill = pre.query_window_provider(x, wq, bq, 2)
            with self.assertRaises(ValueError):
                fill(0, 20, torch.empty((2, 8, 128), dtype=x.dtype))

    def test_graph_research_provider_not_exposed_to_production(self):
        flow = FLOW.read_text(encoding="utf-8")
        bench = RUNNER.read_text(encoding="utf-8")
        ast.parse(flow, filename=str(FLOW))
        ast.parse(bench, filename=str(RUNNER))
        self.assertIn("not compare_only or not callable(query_provider)", flow)
        self.assertIn("query_provider(geom.start, geom.end, item[\"q\"])",flow)
        self.assertIn("query_provider is not None",flow)
        self.assertIn("if query_provider is None:",flow)
        self.assertIn("extra_reserved > max_extra*_MIB",flow)
        self.assertIn("free < min_free*_MIB",flow)
        self.assertIn("if not exact and not compare_only:",flow)
        self.assertIn('"nvfp4": "UNRUN_NOT_IMPLEMENTED"',bench)
        self.assertIn('"h3_rmsnorm_rope": "UNRUN_NOT_IMPLEMENTED"',bench)
        self.assertIn('flat_pre_qkv_stream_ab.json',bench)
        self.assertIn("with tempfile.TemporaryDirectory(",bench)


if __name__ == "__main__":
    unittest.main()
