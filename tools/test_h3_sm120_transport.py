"""CPU-only tests for the no-kernel SM120 transport inventory."""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
import unittest

SRC = Path(__file__).with_name("probe_h3_sm120_transport.py")
spec = importlib.util.spec_from_file_location("htrc_sm120_transport", SRC)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Sm120TransportSourceGate(unittest.TestCase):
    def test_5070_ti_must_reject_multicast(self):
        d = module.classify_device("NVIDIA GeForce RTX 5070 Ti", 12, 0)
        self.assertEqual(d["vendor_tma_multicast"],
                         "UNSUPPORTED_BY_NVIDIA_CUTLASS")
        self.assertIn("NO_GO", d["decision"])
        self.assertEqual(d["production_promotion"],
                         "PROHIBITED_CAPABILITY_ONLY")

    def test_no_false_positive_on_other_sm120(self):
        d = module.classify_device("NVIDIA RTX PRO 6000 Blackwell", 12, 0)
        self.assertEqual(d["vendor_tma_multicast"], "UNKNOWN_FAIL_CLOSED")
        self.assertNotIn("NO_GO_HARDWARE", d["decision"])

    def test_no_cluster_or_unicast_claim_on_unknown_device(self):
        d = module.classify_device("Unnamed GPU", 10, 0)
        self.assertEqual(d["vendor_tma_multicast"], "UNKNOWN_FAIL_CLOSED")
        self.assertEqual(d["tma_unicast"], "NOT_EXECUTED_NOT_PROVEN")
        self.assertEqual(d["l2_reuse"], "NOT_MEASURED_NO_PHYSICAL_HBM_CLAIM")

    def test_no_compilation_or_kernel_calls(self):
        src = SRC.read_text(encoding="utf-8")
        tree = ast.parse(src)
        self.assertNotIn("@triton.jit", src)
        self.assertNotIn("torch.empty(", src)
        self.assertNotIn("torch.randn(", src)
        self.assertIn('"gpu_kernel_executed": False', src)
        self.assertIn('"cuda_graph_captured": False', src)
        self.assertIn('"benchmark_executed": False', src)
        self.assertFalse(any(isinstance(n, ast.Import) and
                             any(a.name == "torch" for a in n.names)
                             for n in tree.body))


if __name__ == "__main__":
    unittest.main()
