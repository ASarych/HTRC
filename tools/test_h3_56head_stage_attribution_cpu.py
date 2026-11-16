"""CPU-only correctness and source contracts for 56-head HTRC stage diagnostic."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark_h3_56head_stage_attribution import (
    DIM, GROUP_CAP, HEADS, ROUNDS, SCHEMA, STAGES, TILES, TOKENS, WARMUPS,
    validate_request, verify_first_call, verify_graph_timing, verify_observer,
)


def full_gate():
    return {
        'status': 'PASS_FIRST_REAL_CALL', 'completed': True,
        'all_groups_bit_exact': True, 'strict_parity_required': True,
        'eager_first_call_diagnostic': 'EXECUTED',
        'heads': HEADS, 'tokens': TOKENS, 'mode': 'balanced',
        'vb_mode': 'materialized', 'group_policy': 'reuse_aware',
        'head_group_cap': GROUP_CAP, 'window_tiles': TILES,
        'fixed_exact_leaves': 512,
        'head_groups': [
            {'head_range': [i, min(i + GROUP_CAP, HEADS)], 'exact_flat_parity': True}
            for i in range(0, HEADS, GROUP_CAP)
        ], 'extra_reserved_mib': 216, 'graph_cache_group_sizes': [8],
    }


def stages():
    return [('htrc_flat_gpu_dispatch', {
        'heads': HEADS, 'tokens': TOKENS,
        'profile_gpu_stream_elapsed_ms': {k: 0.5 for k in STAGES},
        'profile_cpu_enqueue_ms': {k: 0.05 for k in STAGES},
        'query_staging': 'native_torch_mu', 'kv_staging': 'native_direct',
        'exact_leaves': 512, 'group_capacity': 11, 'query_windows': 15,
    })]


class FullOutputAdmission(unittest.TestCase):
    def test_fixed_geometry(self):
        self.assertEqual((TOKENS, HEADS, TILES, DIM, GROUP_CAP, ROUNDS, WARMUPS),
                         (15275, 56, 16, 128, 8, 7, 5))
        self.assertEqual(SCHEMA, 'htrc-56head-graph-eager-stage-attribution-v1')

    def test_first_call_strict_all_56_heads(self):
        result = verify_first_call(full_gate(), 1)
        self.assertEqual(len(result['head_ranges']), 7)
        self.assertEqual(result['head_ranges'][-1], [48, 56])
        self.assertTrue(result['full_output_exact'])

    def test_no_sampled_only_or_repeated_proxy(self):
        g = full_gate()
        for update in ({'strict_parity_required': False}, {'all_groups_bit_exact': None},
                       {'status': 'RECORDED_FIRST_CALL'}, {'head_group_cap': 7}):
            with self.subTest(update=update), self.assertRaises(AssertionError):
                verify_first_call({**g, **update}, 0)
        g['head_groups'][-1]['exact_flat_parity'] = False
        with self.assertRaises(AssertionError):
            verify_first_call(g, 0)

    def test_eager_observer_stage_scope_and_no_invented_graph_breakdown(self):
        d = verify_observer(stages())
        self.assertEqual(set(d['stage_gpu_stream_intervals_ms']), set(STAGES))
        self.assertEqual(d['scope'], 'EAGER_ONLY__NOT_GRAPH_KERNEL_TIMING')
        self.assertTrue(d['gpu_and_cpu_intervals_overlap_not_additive'])
        self.assertNotEqual(d['head_group_capacity'], GROUP_CAP)

    def test_wrong_observer_fails_closed(self):
        with self.assertRaises(AssertionError):
            verify_observer([])
        with self.assertRaises(AssertionError):
            verify_observer(stages() + stages())
        row = stages()
        del row[0][1]['profile_gpu_stream_elapsed_ms']['tile']
        with self.assertRaises(AssertionError):
            verify_observer(row)
        row = stages()
        row[0][1]['profile_cpu_enqueue_ms']['merge_tail'] = float('nan')
        with self.assertRaises(AssertionError):
            verify_observer(row)

    def test_warmed_complete_graph_timing(self):
        d = verify_graph_timing([{'gpu_ms': 70.0, 'sync_wall_ms': 75.0} for _ in range(ROUNDS)])
        self.assertEqual(d['gpu_event_median_ms'], 70.0)
        self.assertEqual(d['gpu_event_block_spread_pct'], 0.0)
        with self.assertRaises(AssertionError):
            verify_graph_timing([{'gpu_ms': 70, 'sync_wall_ms': 71}] * 6)
        with self.assertRaises(AssertionError):
            verify_graph_timing([{'gpu_ms': 0, 'sync_wall_ms': 71}] * 7)

    def test_explicit_quiet_and_pinned_cli(self):
        class Args:
            rounds = ROUNDS
            warmups = WARMUPS
            quiet_confirmed = True
            expect_head = 'a'*40
        validate_request(Args())
        a = Args()
        a.quiet_confirmed = False
        with self.assertRaises(ValueError):
            validate_request(a)
        a.quiet_confirmed = True
        a.rounds = 8
        with self.assertRaises(ValueError):
            validate_request(a)


class SourceAndFailureContracts(unittest.TestCase):
    def test_wrong_head_smoke_prevents_importing_cuda(self):
        with tempfile.TemporaryDirectory(prefix='htrc56-') as tmp:
            root = Path(tmp)
            tools = root / 'tools'
            tools.mkdir()
            shutil.copy2(HERE / 'benchmark_h3_56head_stage_attribution.py', tools)
            subprocess.run(['git', 'init', '-q', str(root)], capture_output=True, check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=HTRC-test',
                            '-c', 'user.email=test@example.invalid', 'commit',
                            '--allow-empty', '-q', '-m', 'init'], capture_output=True, check=True)
            output = root / 'fail.json'
            proc = subprocess.run([sys.executable, '-B',
                                   str(tools / 'benchmark_h3_56head_stage_attribution.py'),
                                   '--expect-head', '0'*40, '--quiet-confirmed',
                                   '--output', str(output)], capture_output=True, text=True)
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(report['status'], 'FAIL_CLOSED')
            self.assertFalse(report['completed'])
            self.assertEqual(report['numeric_gate'], 'UNRUN')
            self.assertIn('Exact HEAD mismatch', report['error'])
            self.assertIsNone(report['warmed_graph'])

    def test_source_retains_real_backends_and_full_parity(self):
        p = (HERE / 'benchmark_h3_56head_stage_attribution.py').read_text('utf-8')
        import ast
        ast.parse(p)
        for token in ('from h3 import htrc_flat_directflow_live as flow',
                      'from h3 import htrc_flat_live_backend as eager',
                      'compare_only=False', 'query_staging=\'native_torch_mu\'',
                      'kv_staging=\'native_direct\'',
                      'report[\'independent_full_output_bit_exact\']',
                      "'h3_model_or_video': 'UNRUN'",
                      "'production_promotion': 'PROHIBITED_RESEARCH_ONLY'",
                      "'not_a_graph_internal_stage_profile': True"):
            self.assertIn(token, p)
        self.assertLess(p.index('actual != args.expect_head:'), p.index('result = run_gpu(args, result)'))
        self.assertNotIn('import torch\n', p.split('def run_gpu(args, report):')[0])
        self.assertIn('torch.inference_mode()', p)

    def test_existing_graph_and_eager_stage_entrypoints_unchanged(self):
        root = HERE.parent
        graph = root / 'h3' / 'htrc_flat_directflow_live.py'
        eager = root / 'h3' / 'htrc_flat_live_backend.py'
        if not graph.exists() or not eager.exists():
            self.skipTest('Exact HTRC checkout required for source contract')
        graph_s = graph.read_text('utf-8')
        eager_s = eager.read_text('utf-8')
        self.assertIn('def graph_attention_bthd(', graph_s)
        self.assertIn('"all_groups_bit_exact":(', graph_s)
        self.assertIn('def flat_attention_bthd(', eager_s)
        self.assertIn('"profile_gpu_stream_elapsed_ms"', eager_s)
        for stage in STAGES:
            self.assertIn('phase_begin("'+stage+'")', eager_s)

    def test_runner_exact_clone_single_json_and_syntax(self):
        r = (HERE / 'run_h3_56head_stage_attribution_safe.ps1').read_text('utf-8')
        for required in ('git clone --no-checkout --no-hardlinks',
                         'git -C $tmp switch --detach $ExactHead',
                         'test_h3_56head_stage_attribution_cpu.py',
                         'flat_56head_stage_attribution.json',
                         '--quiet-confirmed --rounds 7 --warmups 5',
                         'IClosedOtherGpuWorkloads',
                         'PASS_TWO_BANK_FULL_OUTPUT_GRAPH_EAGER_BIT_EXACT',
                         'independent_full_output_bit_exact',
                         'PASS_ATTRIBUTION_ONLY'):
            self.assertIn(required, r)
        self.assertNotIn("benchmark_h3_56head_stage_attribution.py') `\n", r)
        exe = shutil.which('powershell.exe') or shutil.which('pwsh')
        if exe:
            path = str((HERE / 'run_h3_56head_stage_attribution_safe.ps1').resolve()).replace("'", "''")
            source = ("$tok=$null;$err=$null;"
                      "[void][System.Management.Automation.Language.Parser]::ParseFile("
                      + "'" + path + "',[ref]$tok,[ref]$err);"
                      "if($err.Count){$err|Out-String|Write-Host;exit 1}")
            proc = subprocess.run([exe, '-NoProfile', '-NonInteractive', '-Command', source],
                                  capture_output=True, text=True, timeout=20)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

if __name__ == '__main__':
    unittest.main()
