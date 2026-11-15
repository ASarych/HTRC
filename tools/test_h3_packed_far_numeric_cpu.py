"""CPU-only and source-contract tests for narrow far PASS=1 numeric prototype."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from analyze_h3_far_pack import reference_pack


class PackedFarContracts(unittest.TestCase):
    def test_preserve_original_split_membership(self):
        mask=[int(i%2==0) for i in range(128)]
        selected=[i for i in range(128) if i%2==0]
        ref=reference_pack([[mask]],[[selected]],splits=2,bn=64,expected_near=64)
        self.assertEqual(ref['indices'][0][0][0],list(range(1,64,2)))
        self.assertEqual(ref['indices'][0][0][1],list(range(65,128,2)))
        self.assertEqual(ref['packed_indices'],64)
        self.assertEqual(ref['packed_blocks'],2)

    def test_packed_groups_are_not_original_bn64_groups(self):
        # Shows a structural NUMERIC risk, not a GPU proof of mismatch.
        mask=[int(i%2==0) for i in range(128)]
        ref=reference_pack([[mask]],[[[i for i in range(0,128,2)]]],splits=1,bn=64)
        original=[list(range(0,64)),list(range(64,128))]
        packed=[ref['indices'][0][0][0][i:i+64] for i in range(0,64,64)]
        self.assertEqual(len(original),2)
        self.assertEqual(len(packed),1)
        self.assertEqual(len(packed[0]),64)
        self.assertNotEqual(original[0],packed[0])

    def test_pinned_geometry_split_aligned(self):
        from analyze_h3_far_exclusion import aligned_split
        self.assertEqual(aligned_split(954,4,64),256)
        self.assertEqual(954 * 16,15264)
        self.assertEqual(15275 - 15264,11)
        self.assertEqual(4*32*4,512)

    def test_packed_triton_compute_contract(self):
        s=(HERE/'htrc_packed_far_numeric_triton.py').read_text(encoding='utf-8')
        ast.parse(s)
        for needle in (
            'for offset in range(0,count,BN):',
            'valid = slot<count',
            'tl.load(PACKED+pos*LPS+slot,mask=valid,other=0)',
            'tl.dot(dq8,tl.trans(g))',
            'sdq[:,None]*sgc[None,:]',
            'p = tl.exp(sc-m_s[:,None])',
            'acc = acc*alpha[:,None] + tl.dot(p.to(tl.bfloat16),vb)',
            'tl.store(PM+outbase*T+r,m_i)',
            'tl.store(PL+outbase*T+r,l_i)',
            'tl.store(PACC+(outbase*T+r)[:,None]*D+d[None,:],acc)',
            'num_warps',
        ):
            if needle == 'num_warps': continue
            self.assertIn(needle,s)
        self.assertNotIn('tl.load(PACKED+pos*LPS+slot)',s)
        self.assertNotIn('import torch',s)

    def test_control_is_original_pass1_not_new_arithmetic(self):
        s=(HERE/'benchmark_h3_packed_far_numeric.py').read_text(encoding='utf-8')
        ast.parse(s)
        self.assertIn('tr._tile_kernel[(HEADS*TILES,SPLITS,1)]',s)
        self.assertIn('PASS=1,LATE_NEAR_Q=0,FAR_PREFETCH_LZ=0',s)
        self.assertIn('num_warps=4,num_stages=2',s)
        self.assertIn('source_head',s)
        self.assertIn('states["EXCL"]',s)

    def test_strict_raw_bit_parity_before_any_speed(self):
        s=(HERE/'benchmark_h3_packed_far_numeric.py').read_text(encoding='utf-8')
        self.assertIn('a.view(torch.int32) != b.view(torch.int32)',s)
        self.assertIn('far_partial_all_bit_exact',s)
        self.assertIn('"parity_admission"]="NO_GO_FAR_BIT_EXACT"',s)
        self.assertIn('"no_cuda_event_timing":True',s)
        self.assertNotIn('torch.cuda.CUDAGraph()',s)
        self.assertNotIn('start.elapsed_time',s)
        self.assertIn('far_partial_bit_exact',s)
        self.assertIn('"full_packed_attention":"UNRUN"',s)

    def test_pack_prefix_access_masks_suffix(self):
        s=(HERE/'benchmark_h3_packed_far_numeric.py').read_text(encoding='utf-8')
        self.assertIn('packed[valid].detach().cpu().tolist()',s)
        self.assertIn('valid = torch.arange(LPS',s)
        self.assertNotIn('packed.detach().cpu().tolist()',s)
        self.assertIn('if actual_valid != reference_valid:',s)
        self.assertIn('if fingerprint != EXPECTED_MASKS[bi]:',s)

    def test_runner_has_expected_negative_result_and_full_source_gate(self):
        s=(HERE/'run_h3_packed_far_numeric_safe.ps1').read_text(encoding='utf-8')
        for needle in (
            'git clone --no-checkout --no-hardlinks',
            'git -C $tmp switch --detach $ExactHead',
            'test_h3_packed_far_numeric_cpu.py',
            'benchmark_h3_packed_far_numeric.py',
            'NO_GO_FAR_BIT_EXACT',
            'FAIL_CLOSED',
            'speed_measurement -ne',
            'flat_packed_far_numeric.json',
        ):
            self.assertIn(needle,s)

    def test_wrong_head_fails_closed_without_cuda(self):
        with tempfile.TemporaryDirectory(prefix='htrc-packed-far-smoke-') as tmp:
            tmp=Path(tmp)
            # Place script in temp `tools` so ROOT is the newly initialized repo.
            dest=tmp/'tools';dest.mkdir()
            import shutil
            for file in ('benchmark_h3_packed_far_numeric.py',
                         'analyze_h3_far_pack.py','analyze_h3_far_exclusion.py',
                         'profile_h3_far_exclusion.py'):
                shutil.copy2(HERE/file,dest/file)
            subprocess.run(['git','init','-q',str(tmp)],check=True,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            subprocess.run(['git','-C',str(tmp),'-c','user.name=HTRC-test',
                            '-c','user.email=ht@example.invalid','commit',
                            '--allow-empty','-q','-m','init'],check=True,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            out=tmp/'one.json'
            command=[sys.executable,'-B',str(dest/'benchmark_h3_packed_far_numeric.py'),
                     '--expect-head','0'*40,'--output',str(out)]
            r=subprocess.run(command,capture_output=True,text=True)
            import json
            obj=json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(r.returncode,2)
            self.assertEqual(obj['status'],'FAIL_CLOSED')
            self.assertEqual(obj['speed_measurement'],'UNRUN_NUMERIC_GATE_ONLY')
            self.assertFalse(obj['completed'])
            self.assertIn('Exact HEAD mismatch',obj['error'])
            self.assertEqual(obj['speed_capture_blocks'],[])

    def test_full_repo_original_far_contract(self):
        root=HERE.parent
        p=root/'h3'/'htrc_flat_triton.py'
        b=root/'h3'/'htrc_flat_head_batch.py'
        if not p.exists() or not b.exists():
            self.skipTest('Needs exact whole HTRC checkout; mandatory on Windows')
        s=p.read_text(encoding='utf-8')
        l=b.read_text(encoding='utf-8')
        for needle in (
            'if PASS != 2:',
            'if PASS != 1:',
            'for n0 in range(lo, hi, BN):',
            'sc = tl.where(((ex == 0) & cm)[None, :], sc, float("-inf"))',
            'acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)',
        ):
            self.assertIn(needle,s)
        self.assertIn('Q_DIRECT=int(direct)',l)
        self.assertIn('BATCH_TILES_PER_HEAD=tiles',l)

if __name__ == '__main__':
    unittest.main()
