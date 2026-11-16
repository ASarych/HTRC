"""CPU-only checks for matched native fast/balanced quality pilot."""
from __future__ import annotations
import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmark_h3_fast_balanced_dense_quality_pilot import (
    interpret, run_gpu, MODES, ROUNDS, SEED, TOKENS, HEADS, TILES,
)
ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'tools'/'benchmark_h3_fast_balanced_dense_quality_pilot.py'
HELPER=ROOT/'tools'/'h3_flat_quality_ab.py'
RUNNER=ROOT/'tools'/'run_h3_fast_balanced_dense_quality_pilot_safe.ps1'


def quality(value):
    return {'valid':True,'relative_l2_percent':value,
            'per_query_rel_l2_p90_percent':value*1.6,
            'per_query_rel_l2_max_percent':value*2.3,
            'per_head_rel_l2_max_percent':value*1.2}


def case(mode,err,time_ms):
    return {
        'status':'MEASURED',
        'bank_count':2,'rng_seed':SEED,
        'mode':mode,'tokens':TOKENS,'heads':HEADS,
        'window_tiles':TILES,
        'sampled_query_positions':list(range(64)),
        'reference':{'name':'dense_exact_scaled_dot_product'},
        'results':{
            'materialized':{
                'status':'QUALITY_MEASURED',
                'quality':[quality(err),quality(err+1)],
                'different_qkv_banks_produced_bit_exact_same_output':False,
                'timing':{
                    'status':'TIMED_NOT_QUALITY_GATE',
                    'samples':[{'gpu_event_ms':time_ms,'synchronized_wall_ms':time_ms+2}
                               for _ in range(ROUNDS)],
                    'gpu_event_ms_median':time_ms,
                },
            }
        },
    }


class NearBudgetQualityCPU(unittest.TestCase):
    def test_syntax(self):
        ast.parse(PILOT.read_text('utf-8'))

    def test_pair_quality_delta_and_preliminary_speed(self):
        arms={'balanced':case('balanced',5,70),
              'fast':case('fast',8,50)}
        c=interpret(arms)
        self.assertEqual(len(c['bank_quality']),2)
        self.assertAlmostEqual(c['bank_quality'][0]['fast_minus_balanced_percentage_points'],3)
        self.assertAlmostEqual(c['bank_quality'][1]['fast_minus_balanced_percentage_points'],3)
        self.assertAlmostEqual(c['single_capture_unpaired_time']
                               ['preliminary_fast_reduction_percent'],(1-50/70)*100)
        self.assertTrue(c['single_capture_unpaired_time']
                        ['not_an_independent_capture_ABBA_speedup'])
        self.assertEqual(c['quality_budget_policy'],
                         'REPORT_ONLY_NO_AUTOMATIC_THRESHOLD')

    def test_invalid_mode_order_and_extra_vb_arm_rejected(self):
        arms={'balanced':case('balanced',5,70),
              'fast':case('fast',8,50)}
        bad=dict(reversed(list(arms.items())))
        with self.assertRaises(ValueError):interpret(bad)
        arms['balanced']['results']['split_stream']={}
        with self.assertRaises(ValueError):interpret(arms)

    def test_banks_and_synthetic_nonfinite_quality_rejected(self):
        a={'balanced':case('balanced',5,70),
           'fast':case('fast',8,50)}
        b=deepcopy(a)
        b['fast']['results']['materialized']['quality'][0]['valid']=False
        with self.assertRaises(ValueError):interpret(b)
        b=deepcopy(a)
        b['fast']['results']['materialized']['quality'][0]['relative_l2_percent']=float('nan')
        with self.assertRaises(ValueError):interpret(b)
        b=deepcopy(a)
        b['fast']['results']['materialized']['different_qkv_banks_produced_bit_exact_same_output']=True
        with self.assertRaises(ValueError):interpret(b)

    def test_runtime_or_missing_samples_fails_closed(self):
        a={'balanced':case('balanced',5,70),
           'fast':case('fast',8,50)}
        b=deepcopy(a)
        b['fast']['status']='PARTIAL_RUNTIME_FAILURE'
        with self.assertRaises(ValueError):interpret(b)
        b=deepcopy(a)
        b['fast']['results']['materialized']['timing']['samples'].pop()
        with self.assertRaises(ValueError):interpret(b)

    def test_pilot_uses_two_identically_seeded_modes_with_dense_oracle(self):
        pilot=PILOT.read_text('utf-8')
        helper=HELPER.read_text('utf-8')
        self.assertIn('MODES=(\'balanced\',\'fast\')',pilot)
        self.assertIn('variants=(),sample_queries=SAMPLED,seed=SEED',pilot)
        self.assertIn('torch.manual_seed(seed)',helper)
        self.assertIn('exact_dense_sampled_queries(*bank,positions,query_batch=8)',helper)
        self.assertIn('quality_against_exact(',helper)
        self.assertIn('if variants is None or any(',helper)
        self.assertLess(pilot.index('if actual!=args.expect_head:'),
                        pilot.index('        run_gpu(report)'))
        self.assertNotIn('import torch\n',pilot[:pilot.index('def run_gpu(report):')])
        self.assertIn("speed_admission':'UNRUN_INDEPENDENT_CAPTURE_ABBA'",pilot)

    def test_inference_helper_disables_grad_and_restores_state(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU torch unavailable')
        from h3_flat_quality_ab import _graph_attention_inference
        class FakeFlow:
            def graph_attention_bthd(self, *qkv, **kw):
                self_assert=(
                    not torch.is_grad_enabled()
                    and torch.is_inference_mode_enabled()
                    and kw['compare_only'] is True
                    and kw['mode']=='balanced'
                    and len(qkv)==3
                )
                if not self_assert:
                    raise AssertionError('HTRC Graph called with autograd enabled')
                return 'observed inference mode'
        previous_grad=torch.is_grad_enabled()
        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            result=_graph_attention_inference(
                torch,FakeFlow(),(1,2,3),heads=56,mode='balanced',
                tiles=16,budget_mib=1024,reserve_mib=256)
            self.assertEqual(result,'observed inference mode')
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(torch.is_grad_enabled(),previous_grad)

    def test_inference_helper_restores_grad_on_graph_error(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU torch unavailable')
        from h3_flat_quality_ab import _graph_attention_inference
        class FakeFlow:
            def graph_attention_bthd(self, *qkv, **kwargs):
                if torch.is_grad_enabled():
                    raise AssertionError('Expected inference mode even on failure')
                raise RuntimeError('synthetic graph refusal')
        with torch.enable_grad():
            with self.assertRaisesRegex(RuntimeError,'synthetic graph refusal'):
                _graph_attention_inference(
                    torch,FakeFlow(),(1,2,3),heads=56,mode='fast',
                    tiles=16,budget_mib=1024,reserve_mib=256)
            self.assertTrue(torch.is_grad_enabled())

    def test_preserve_both_modes_when_one_arm_has_runtime_error(self):
        report={'arms':{}}
        def fake_run_case(*,mode,on_update,**kw):
            self.assertEqual(kw['variants'],())
            case_result=case(mode,5 if mode=='balanced' else 8,
                             70 if mode=='balanced' else 50)
            if mode=='fast':
                case_result['status']='PARTIAL_RUNTIME_FAILURE'
                record=case_result['results']['materialized']
                record['status']='RUNTIME_FAILED'
                record['runtime_error']='RuntimeError: H3 Direct-Flow is inference-only'
            on_update('materialized',case_result['results']['materialized'])
            return case_result
        with patch('h3_flat_quality_ab.run_case',side_effect=fake_run_case):
            with self.assertRaisesRegex(
                ValueError,"fast: dense quality arm status='PARTIAL_RUNTIME_FAILURE'"):
                run_gpu(report)
        self.assertEqual(list(report['arms']),['balanced','fast'])
        self.assertEqual(report['arms']['balanced']['status'],'MEASURED')
        self.assertEqual(report['arms']['fast']['status'],'PARTIAL_RUNTIME_FAILURE')
        self.assertIn('inference-only',
                      report['arms']['fast']['results']['materialized']['runtime_error'])
        self.assertIsNone(report['active_mode'])
        self.assertNotIn('comparison',report)

    def test_interpret_error_names_failed_mode_and_internal_failure(self):
        arms={'balanced':case('balanced',5,70),'fast':case('fast',8,50)}
        fast=arms['fast']
        fast['status']='PARTIAL_RUNTIME_FAILURE'
        fast['results']['materialized']['status']='RUNTIME_FAILED'
        fast['results']['materialized']['runtime_error']='RuntimeError: synthetic CUDA failure'
        with self.assertRaises(ValueError) as caught:
            interpret(arms)
        self.assertIn("fast: dense quality arm status=",str(caught.exception))
        self.assertIn('synthetic CUDA failure',str(caught.exception))

    def test_exact_runner_and_one_json(self):
        p=RUNNER.read_text('utf-8')
        for needle in ('git -C $tmp switch --detach $ExactHead',
                       'test_h3_fast_balanced_dense_quality_pilot_cpu.py',
                       'test_h3_flat_scale_protocol.py',
                       'flat_fast_balanced_56head_dense_quality.json',
                       'QUALITY_MEASURED_SPEED_PRELIMINARY_ONLY',
                       'PROHIBITED_RESEARCH_ONLY'):
            self.assertIn(needle,p)
        self.assertLess(p.index('test_h3_fast_balanced_dense_quality_pilot_cpu.py'),
                        p.index('& $py -B $benchmark --expect-head'))
        self.assertNotIn('@traceArgs',p)
        self.assertIn('original error: $($arm.results.materialized.runtime_error)',p)
        self.assertIn("foreach($mode in @('balanced','fast')){",p)

if __name__=='__main__':
    unittest.main()
