"""No-GPU numerical gate contracts for a research full-Window masked-G Tile."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))


class FullMaskGSource(unittest.TestCase):
    def test_full_tile_far_near_order_and_bounded_mask(self):
        s=(HERE/'htrc_full_mask_g_triton.py').read_text(encoding='utf-8')
        ast.parse(s)
        for key in ('for n0 in range(lo, hi, BN):',
                    'for e0 in range(e_lo, e_hi, BN):',
                    'lo = sp * LPS','hi = tl.minimum(NB, lo + LPS)',
                    'e_lo = sp * EPS','e_hi = tl.minimum(e_n, e_lo + EPS)',
                    'eligible = (ex == 0) & cm','gmask = eligible',
                    'gmask = cm','mask=gmask[:, None]',
                    'mask=cm[:, None], other=0.0',
                    'tl.dot(dq8, tl.trans(g))',
                    'tl.dot(q8, tl.trans(kk))',
                    'l_i = l_i * alpha + tl.sum(p, axis=1)',
                    'acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)',
                    'acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)',
                    'tl.store(PM + base * T + r, m_i)',
                    'tl.store(PL + base * T + r, l_i)',
                    'tl.store(PACC + (base * T + r)[:, None] * D + d[None, :], acc)'):
            self.assertIn(key,s)
        self.assertLess(s.index('for n0 in range'),s.index('for e0 in range'))
        self.assertNotIn('PACKED',s)
        self.assertNotIn('COUNTS',s)
        self.assertNotIn('import torch',s)

    def test_mask_g_does_not_change_near_or_vb_inputs(self):
        s=(HERE/'htrc_full_mask_g_triton.py').read_text(encoding='utf-8')
        self.assertEqual(s.count('if MASK_G:'),1)
        self.assertIn('if MASK_G:\n            gmask = eligible\n        else:\n            gmask = cm',s)
        self.assertIn('vb = tl.load(VB + (t * NB + c)[:, None] * D + d[None, :],\n                     mask=cm[:, None]',s)
        self.assertIn('leaf = tl.load(IDX + t * KX + j // LEAF, mask=jm, other=0)',s)
        self.assertIn('ksc = tl.load(KS + key_base + tok, mask=jm, other=0.0)',s)

    def test_launch_uses_original_full_tile_and_merge(self):
        s=(HERE/'benchmark_h3_full_mask_g_numeric.py').read_text(encoding='utf-8')
        ast.parse(s)
        for key in ("batch.launch_prepared_tile_batch(states,cfg,tr,buffers=control)",
                    'batch.launch_prepared_merge_batch(states,cfg,tr,partial,tails=tail',
                    'output=out,merge_rows=32',
                    "_check_variant(torch,batch,states,cfg,tr,control,",
                    "'mirror',False", "'mask_g',True",
                    "item['output'],ref_merged",
                    'compare_far_partials(torch,base,scratch)',
                    'ref.view(torch.int32)!=candidate.view(torch.int32)',
                    "'NO_GO_FULL_RESEARCH_MIRROR'",
                    "'NO_GO_FULL_MASK_G_BIT_EXACT'",
                    "'PASS_FULL_MASK_G_NUMERIC_ONLY'",
                    "'UNRUN_NUMERIC_ONLY'", "'speed_capture_blocks':[]"):
            self.assertIn(key,s)
        self.assertLess(s.index("'mirror',False"),s.index("'mask_g',True"))
        self.assertLess(s.index('actual!=args.expect_head'),s.index('result=run_gpu()'))
        self.assertNotIn('torch.cuda.CUDAGraph()',s)
        self.assertNotIn('elapsed_time',s)

    def test_original_geometry_and_tail_are_pinned(self):
        s=(HERE/'benchmark_h3_full_mask_g_numeric.py').read_text(encoding='utf-8')
        self.assertIn('keys!=15264',s)
        self.assertIn('tuple(tail[0].shape)!=(HEADS,11,D)',s)
        self.assertIn('fingerprint!=EXPECTED_MASKS[bi]',s)
        self.assertIn("selection['current_far_dot_blocks']!=1920",s)
        self.assertIn("states['K'].view(HEADS*keys,D)",s)

    def test_error_report_is_one_json_with_speed_untouched(self):
        s=(HERE/'benchmark_h3_full_mask_g_numeric.py').read_text(encoding='utf-8')
        for key in ("'status':'FAIL_CLOSED'", "'completed':False",
                    "'no_cuda_event_timing':True", "'speed_capture_blocks':[]",
                    "args.output.write_text(json.dumps(report,indent=2)+'\\n'", 
                    'return 0 if report[\'status\']==\'PASS_FULL_MASK_G_NUMERIC_ONLY\' else 2'):
            self.assertIn(key,s)

    def test_runner_guards_exact_head_and_scientific_negative(self):
        s=(HERE/'run_h3_full_mask_g_numeric_safe.ps1').read_text(encoding='utf-8')
        for key in ('git clone --no-checkout --no-hardlinks',
                    'git -C $tmp switch --detach $ExactHead',
                    'test_h3_full_mask_g_numeric_cpu.py',
                    'benchmark_h3_full_mask_g_numeric.py',
                    'NO_GO_FULL_RESEARCH_MIRROR', 'NO_GO_FULL_MASK_G_BIT_EXACT',
                    'PASS_FULL_MASK_G_NUMERIC_ONLY',
                    'flat_full_mask_g_numeric.json',
                    '$r.speed_capture_blocks.Count -ne 0'):
            self.assertIn(key,s)

    def test_wrong_head_fails_closed_before_cuda(self):
        with tempfile.TemporaryDirectory(prefix='htrc-full-mask-g-cpu-') as tmp:
            root=Path(tmp)
            td=root/'tools';td.mkdir()
            for name in ('benchmark_h3_full_mask_g_numeric.py',
                         'benchmark_h3_far_masked_loads_numeric.py',
                         'analyze_h3_far_exclusion.py',
                         'profile_h3_far_exclusion.py'):
                shutil.copy2(HERE/name,td/name)
            subprocess.run(['git','init','-q',str(root)],check=True,capture_output=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=HTRC-test',
                            '-c','user.email=test@example.invalid','commit','--allow-empty',
                            '-q','-m','init'],check=True,capture_output=True)
            dest=root/'result.json'
            p=subprocess.run([sys.executable,'-B',str(td/'benchmark_h3_full_mask_g_numeric.py'),
                              '--expect-head','0'*40,'--output',str(dest)],
                              capture_output=True,text=True)
            report=json.loads(dest.read_text(encoding='utf-8'))
            self.assertEqual(p.returncode,2)
            self.assertEqual(report['status'],'FAIL_CLOSED')
            self.assertEqual(report['speed_measurement'],'UNRUN_NUMERIC_ONLY')
            self.assertFalse(report['completed'])
            self.assertEqual(report['speed_capture_blocks'],[])
            self.assertIn('Exact HEAD mismatch',report['error'])

    def test_full_repo_original_contract(self):
        original=HERE.parent/'h3'/'htrc_flat_triton.py'
        batch=HERE.parent/'h3'/'htrc_flat_head_batch.py'
        if not original.is_file() or not batch.is_file():
            self.skipTest('Requires full exact-head HTRC checkout; mandatory on Windows')
        s=original.read_text(encoding='utf-8')
        b=batch.read_text(encoding='utf-8')
        for key in ('if PASS != 1:', 'if PASS != 2:',
                    'for n0 in range(lo, hi, BN):','for e0 in range(e_lo, e_hi, BN_NEAR):',
                    'sc = tl.where(((ex == 0) & cm)[None, :], sc, float("-inf"))',
                    'acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)',
                    'FAR_PV_ACC: tl.constexpr = 0'):
            self.assertIn(key,s)
        self.assertIn('def launch_prepared_merge_batch(',b)
        self.assertIn('output=ws["_out"], merge_rows=32',
                      (HERE.parent/'h3'/'htrc_flat_directflow_live.py').read_text(encoding='utf-8'))


if __name__=='__main__':
    unittest.main()
