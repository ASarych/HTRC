"""CPU/source contracts for original-group masked G/VB far numeric gate."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_h3_far_exclusion import aligned_split, summarize_exclusions


class FarMaskedCPU(unittest.TestCase):
    def test_original_split_geometry_preserved(self):
        self.assertEqual(aligned_split(954,4,64),256)
        self.assertEqual(954//64 + 1,15)
        self.assertEqual(4*32*4,512)

    def test_partial_masks_leave_same_column_positions(self):
        near = [i for i in range(128) if i%2==0]
        exclude = [int(i in near) for i in range(128)]
        kept = [j for j, x in enumerate(exclude) if x==0]
        self.assertEqual(kept,list(range(1,128,2)))
        original = [list(range(start,start+64)) for start in (0,64)]
        per_block = [[x for x in col if exclude[x]==0] for col in original]
        self.assertEqual(per_block[0],list(range(1,64,2)))
        self.assertEqual(per_block[1],list(range(65,128,2)))
        self.assertEqual([len(row) for row in original],[64,64])
        self.assertEqual([len(row) for row in per_block],[32,32])

    def test_exclusion_validation_and_no_zero_blocks(self):
        mask = [int(i%2==0) for i in range(128)]
        selected = [i for i in range(128) if i%2==0]
        audit = summarize_exclusions([[mask]], [[selected]], splits=2, bn=64, expected_near=64)
        self.assertEqual(audit['fully_excluded_blocks'],0)
        self.assertEqual(audit['current_far_dot_blocks'],2)
        self.assertEqual(audit['eligible_far_leaves_per_center'],64)

    def test_independent_research_triton_original_bn64(self):
        source = (HERE/'htrc_far_masked_loads_triton.py').read_text(encoding='utf-8')
        ast.parse(source)
        for token in (
            'lo = sp * LPS', 'hi = tl.minimum(NB, lo + LPS)',
            'for n0 in range(lo, hi, BN):',
            'ex = tl.load(EXCL + t * NB + c, mask=cm, other=1)',
            'eligible = (ex == 0) & cm',
            'gmask = eligible', 'gmask = cm',
            'vmask = eligible', 'vmask = cm',
            'sc = tl.where(eligible[None, :], sc, float("-inf"))',
            'tl.dot(dq8, tl.trans(g))',
            'l_i = l_i * alpha + tl.sum(p, axis=1)',
            'acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)',
            'tl.store(PM + base * T + r, m_i)',
            'tl.store(PL + base * T + r, l_i)',
            'tl.store(PACC + (base * T + r)[:, None] * D + d[None, :], acc)',
        ):
            self.assertIn(token,source)
        self.assertNotIn('PACKED',source)
        self.assertNotIn('COUNTS',source)
        self.assertNotIn('torch.cuda.CUDAGraph',source)
        self.assertNotIn('import torch',source)

    def test_control_cloned_first_and_three_mask_variants(self):
        source=(HERE/'benchmark_h3_far_masked_loads_numeric.py').read_text(encoding='utf-8')
        ast.parse(source)
        for token in (
            '"mirror": (False, False)',
            '"mask_g": (True, False)',
            '"mask_vb": (False, True)',
            '"mask_both": (True, True)',
            'tr._tile_kernel[(HEADS*TILES,SPLITS,1)]',
            'PASS=1,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0',
            'a.view(torch.int32) != b.view(torch.int32)',
            '"parity_admission"]="NO_GO_RESEARCH_DENSE_MIRROR"',
            '"parity_admission"]="NO_GO_MASKED_FAR_BIT_EXACT"',
            '"no_cuda_event_timing":True',
            '"full_masked_attention":"UNRUN"',
        ):
            self.assertIn(token,source)
        self.assertNotIn('torch.cuda.CUDAGraph()',source)
        self.assertNotIn('start.elapsed_time',source)
        self.assertIn('fingerprint != EXPECTED_MASKS[bi]',source)
        self.assertLess(source.index('actual!=args.expect_head'),source.index('result=run_gpu()'))

    def test_no_production_source_modified_by_this_slice(self):
        # The only allowed GitHub changes are research-only files under tools/ or research/.
        doc=(HERE.parent/'research'/'H3_FAR_MASKED_LOADS_NUMERIC_ADMISSION_2026-11-15.md')
        self.assertTrue(doc.is_file())
        text=doc.read_text(encoding='utf-8')
        self.assertIn('no changes to h3/',text)
        self.assertIn('no speed measurement',text)

    def test_runner_exact_head_and_expected_negative(self):
        r=(HERE/'run_h3_far_masked_loads_numeric_safe.ps1').read_text(encoding='utf-8')
        for x in ('git clone --no-checkout --no-hardlinks',
                  'git -C $tmp switch --detach $ExactHead',
                  'test_h3_far_masked_loads_numeric_cpu.py',
                  'benchmark_h3_far_masked_loads_numeric.py',
                  'NO_GO_RESEARCH_DENSE_MIRROR','NO_GO_MASKED_FAR_BIT_EXACT',
                  'flat_far_masked_loads_numeric.json','speed_capture_blocks.Count -ne 0'):
            self.assertIn(x,r)

    def test_wrong_head_fails_without_importing_cuda(self):
        with tempfile.TemporaryDirectory(prefix='htrc-mask-far-smoke-') as temp:
            root=Path(temp);td=root/'tools';td.mkdir()
            import shutil
            for name in ('benchmark_h3_far_masked_loads_numeric.py',
                         'analyze_h3_far_exclusion.py','profile_h3_far_exclusion.py'):
                shutil.copy2(HERE/name,td/name)
            subprocess.run(['git','init','-q',str(root)],check=True,capture_output=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=HTRC-test',
                            '-c','user.email=test@example.invalid','commit','--allow-empty',
                            '-q','-m','init'],check=True,capture_output=True)
            output=root/'one.json'
            cmd=[sys.executable,'-B',str(td/'benchmark_h3_far_masked_loads_numeric.py'),
                 '--expect-head','0'*40,'--output',str(output)]
            result=subprocess.run(cmd,capture_output=True,text=True)
            import json
            row=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result.returncode,2)
            self.assertEqual(row['status'],'FAIL_CLOSED')
            self.assertEqual(row['speed_measurement'],'UNRUN_NUMERIC_GATE_ONLY')
            self.assertEqual(row['speed_capture_blocks'],[])
            self.assertFalse(row['completed'])
            self.assertIn('Exact HEAD mismatch',row['error'])

    def test_exact_repo_default_far_contract(self):
        root=HERE.parent
        t=root/'h3'/'htrc_flat_triton.py'
        b=root/'h3'/'htrc_flat_head_batch.py'
        if not t.is_file() or not b.is_file():
            self.skipTest('Whole exact-head HTRC checkout mandatory in Windows runner')
        s=t.read_text(encoding='utf-8')
        for x in ('for n0 in range(lo, hi, BN):',
                  'sc = tl.where(((ex == 0) & cm)[None, :], sc, float("-inf"))',
                  'acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)',
                  'g = tl.load(G + (gt * nb + c)[:, None] * D + d[None, :], mask=cm[:, None], other=0.0)',
                  'vb = tl.load(VB + (gt * nb + c)[:, None] * D + d[None, :],',
                  'PASS: tl.constexpr = 0'):
            self.assertIn(x,s)
        self.assertIn('FAR_PV_ACC=0,NEAR_PV_ACC=resource_near_pv_acc',
                      b.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
