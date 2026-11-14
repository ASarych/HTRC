"""CPU-only admission contracts for isolated Tile32 resource ABBA."""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

SRC = Path(__file__).with_name("benchmark_h3_tile_resource_abba.py")
LOADER = importlib.util.spec_from_file_location("tile_resource_abba", SRC)
mod = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(mod)


class ResearchTileABBA(unittest.TestCase):
    def test_exact_candidate_whitelist(self):
        for item in mod.CANDIDATES:
            mod.validate_request(item, 15275, 4, 7, 5)
        for bad in ("none", "rb4", "late_q_stage1", "rb2_stage1", "fused_vb", ""):
            with self.assertRaises(ValueError):
                mod.validate_request(bad, 15275, 4, 7, 5)

    def test_pin_and_abba_bounds(self):
        for n, h, rounds, warmups in ((40759,4,7,5),(15275,8,7,5),
                                     (15275,4,6,5),(15275,4,7,4),
                                     (15275,4,13,5)):
            with self.assertRaises(ValueError):
                mod.validate_request("late_q",n,h,rounds,warmups)

    def test_compiled_resources(self):
        k=SimpleNamespace(n_regs=255,n_spills=0,
                          metadata=SimpleNamespace(shared=50688))
        self.assertEqual(mod.compiled_resources(k), {
            "registers_per_thread":255,"shared_bytes_per_cta":50688,
            "compiler_spills":0,"warps_per_cta":4,
        })
        with self.assertRaisesRegex(RuntimeError,"lacks register/shared"):
            mod.compiled_resources(SimpleNamespace(metadata=SimpleNamespace()))

    def test_two_ctas_exact_sm120_resource_limits(self):
        source={"registers_per_thread":255,"shared_bytes_per_cta":50688}
        data=mod.resource_estimate(source, 65536, 102400, 1536)
        self.assertEqual(data["source_resource_limits"], {
            "registers":2,"shared":2,"threads":12,
        })
        self.assertEqual(data["optimistic_resident_ctas_per_sm"],2)
        self.assertFalse(data["meets_both_3cta_necessary_bounds"])

    def test_both_resource_limits_must_pass_for_three_ctas(self):
        ok=mod.resource_estimate({"registers_per_thread":170,"shared_bytes_per_cta":34133},
                                 65536,102400,1536)
        self.assertTrue(ok["meets_both_3cta_necessary_bounds"])
        self.assertEqual(ok["optimistic_resident_ctas_per_sm"],3)
        for regs,smem in ((171,30000),(150,34134),(255,30000),(160,50688)):
            trial=mod.resource_estimate({"registers_per_thread":regs,
                                         "shared_bytes_per_cta":smem},65536,102400,1536)
            self.assertFalse(trial["meets_both_3cta_necessary_bounds"])

    def test_abba_requires_both_independent_wins(self):
        s=mod.abba_decision([0.44,0.46],[0.41,0.42])
        self.assertTrue(s["at_least_5pct_faster_and_consistent"])
        self.assertEqual(s["admission"],"PROMISING_TILE_ONLY")
        slow=mod.abba_decision([0.44,0.46],[0.45,0.40])
        self.assertFalse(slow["at_least_5pct_faster_and_consistent"])
        self.assertEqual(slow["admission"],"NO_GO_TILE_ONLY")
        for arr in (([1.0],[0.8,0.8]),([1.0,1.1],[float('nan'),0.5]),
                    ([1.0,1.1],[0.0,0.5])):
            with self.assertRaises(ValueError):
                mod.abba_decision(*arr)

    def test_no_production_activation_and_strict_partial_parity(self):
        s=SRC.read_text(encoding="utf-8")
        ast.parse(s)
        self.assertIn('batch.launch_prepared_tile_batch(',s)
        self.assertIn('"research_resource_variant": candidate',s)
        self.assertIn('"research_resource_variant":',s)
        self.assertIn('for name in ("pm", "pl", "acc")',s)
        self.assertIn('if not torch.equal(baseline[name], candidate[name]):',s)
        self.assertIn('for bi, bank in enumerate(banks):',s)
        self.assertIn('PASS_FIRST_REAL_CALL',s)
        self.assertIn('if not torch.equal(eager_b.cpu(), outputs[1])',s)
        self.assertNotIn('from comfy',s)

    def test_independent_four_block_graph_and_fixed_single_json(self):
        s=SRC.read_text(encoding="utf-8")
        self.assertIn('("control", args.candidate,',s)
        self.assertIn('args.candidate, "control")',s)
        self.assertIn('torch.cuda.CUDAGraph()',s)
        self.assertIn('for _ in range(args.warmups):',s)
        self.assertIn('for _ in range(args.rounds):',s)
        self.assertIn('graph.replay()',s)
        self.assertIn('args.output.write_text(json.dumps(report',s)
        self.assertIn('"timing_scope": "ONE_TILE_KERNEL_WINDOW_NOT_FULL_ATTENTION"',s)
        self.assertIn('"production_promotion": "PROHIBITED_RESEARCH_ONLY"',s)
        self.assertIn('torch.set_grad_enabled(False)',s)

    def test_rb2_independent_row_ownership(self):
        # The existing _tile_kernel uses RT=T//RB and program_id(2)*RT
        # as each CTA's row offset. Each split's 64 output rows must
        # have precisely one writer for control and RB2.
        for rb in (1, 2):
            owners={}
            for tile in range(128):
                for sp in range(4):
                    for plane in range(rb):
                        rt=64//rb
                        for row in range(plane*rt,(plane+1)*rt):
                            name=(tile,sp,row)
                            self.assertNotIn(name,owners)
                            owners[name]=plane
            self.assertEqual(len(owners),128*4*64)
            for tile in (0,31,32,127):
                for sp in range(4):
                    self.assertEqual(set(r for (t,p,r) in owners if t==tile and p==sp),set(range(64)))

    def test_rb2_geometry_and_both_resource_gate(self):
        self.assertIn("rb2",mod.CANDIDATES)
        mod.validate_request("rb2",15275,4,7,5)
        src=SRC.read_text(encoding="utf-8")
        self.assertIn('"candidate_grid": [128, splits, rb]',src)
        self.assertIn('"candidate_rows_per_cta": 64 // rb',src)
        self.assertIn('rb = 2 if args.candidate in ("rb2", "rb2_late_q") else 1',src)
        # Prevent falsely claiming three blocks with shared still >34KiB
        over=mod.resource_estimate({"registers_per_thread":170,
                                     "shared_bytes_per_cta":40960},65536,102400,1536)
        self.assertFalse(over["meets_both_3cta_necessary_bounds"])

    def test_combined_gate_accepts_only_positive_three_cta_resources(self):
        prepared = mod.resource_estimate(
            {"registers_per_thread":140,"shared_bytes_per_cta":32768},
            65536, 102400, 1536)
        result = mod.combined_resource_gate(
            {"registers_per_thread":140,"shared_bytes_per_cta":32768,
             "compiler_spills":0}, prepared,
            registers_per_sm=65536, shared_bytes_per_sm=102400)
        self.assertTrue(result["admitted_for_tile_timing"])
        self.assertEqual(result["status"],"ADMIT_TILE_TIMING_ONLY")
        self.assertEqual(result["required_shared_bytes_per_cta_at_most"],34133)
        self.assertEqual(result["required_registers_per_thread_at_most"],170)

    def test_combined_gate_rejects_previous_rb2_resource_result(self):
        previous = mod.resource_estimate(
            {"registers_per_thread":140,"shared_bytes_per_cta":37376},
            65536, 102400, 1536)
        result = mod.combined_resource_gate(
            {"registers_per_thread":140,"shared_bytes_per_cta":37376,
             "compiler_spills":0}, previous,
            registers_per_sm=65536, shared_bytes_per_sm=102400)
        self.assertEqual(result["status"],"NO_GO_RESOURCE_GATE")
        self.assertEqual(result["shared_bytes_excess"],3243)
        self.assertFalse(result["admitted_for_tile_timing"])

    def test_combined_gate_rejects_registers_and_spills(self):
        for regs,shared,spills in (
            (171,32768,0), (140,34134,0),
            (140,32768,1), (140,32768,None),
        ):
            with self.subTest(regs=regs,shared=shared,spills=spills):
                estimates=mod.resource_estimate(
                    {"registers_per_thread":regs,"shared_bytes_per_cta":shared},
                    65536,102400,1536)
                result=mod.combined_resource_gate(
                    {"registers_per_thread":regs,"shared_bytes_per_cta":shared,
                     "compiler_spills":spills}, estimates,
                    registers_per_sm=65536,shared_bytes_per_sm=102400)
                self.assertEqual(result["status"],"NO_GO_RESOURCE_GATE")
                self.assertFalse(result["admitted_for_tile_timing"])

    def test_no_tile_timing_when_combined_resource_gate_fails(self):
        src=SRC.read_text(encoding="utf-8")
        self.assertLess(src.index('if args.candidate == "rb2_late_q":'),
                        src.index('for block_idx, arm in enumerate('))
        self.assertLess(src.index('if not gate["admitted_for_tile_timing"]:'),
                        src.index('for block_idx, arm in enumerate('))
        self.assertIn('"admission": "UNRUN_RESOURCE_GATE"',src)
        self.assertIn('"samples_collected": 0',src)
        self.assertIn('return report\n            states = item_b["states"]',src)
        self.assertIn('return 0 if report["status"] in ("PASS_NUMERIC_TILE_ONLY", "NO_GO_RESOURCE_GATE") else 2',src)
        self.assertIn('rb = 2 if args.candidate in ("rb2", "rb2_late_q") else 1',src)

    def test_optional_source_gate_exact_research_variant_only(self):
        source=Path(__file__).parents[1] / 'h3' / 'htrc_flat_head_batch.py'
        if not source.exists():
            self.skipTest('Only standalone CPU helpers available without repository')
        s=source.read_text(encoding="utf-8")
        ast.parse(s)
        self.assertIn('research_resource_variant=None',s)
        self.assertIn('LATE_NEAR_Q=resource_late_near_q',s)
        self.assertIn('num_stages=stages,**vb_kernel_kwargs',s)
        self.assertIn('"late_q", "stage1", "rb2", "rb2_late_q"',s)
        self.assertIn('research_resource_variant in ("late_q", "rb2_late_q")',s)
        self.assertIn('research_resource_variant in ("rb2", "rb2_late_q")',s)
        self.assertIn('resource_rb=1',s)
        self.assertIn('resource_rb=2',s)
        self.assertIn('[(heads*tiles,splits,resource_rb)]',s)
        self.assertIn('TAIL=0,RB=resource_rb,I8=1',s)
        self.assertIn('if research_resource_variant is not None:',s)


if __name__ == "__main__":
    unittest.main()
