"""CPU-only tests for isolated unprofiled original Tile environment diagnostic."""
import ast
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from benchmark_h3_tile_quiet_environment import (
    BLOCKS_PER_PHASE, SAMPLES_PER_BLOCK, PHASES, POLL_MS, REPLAYS,
    _parse_smi_row, decide_wall_stability, validate_request,
)


def phase(value, jitter=0.001):
    return {'blocks': [{
        'samples_ms_per_replay': [value + jitter * i for i in range(SAMPLES_PER_BLOCK)],
        'median_ms_per_replay': value + jitter * j,
    } for j in range(BLOCKS_PER_PHASE)]}


class PureLogic(unittest.TestCase):
    def test_pinned_shape_and_explicit_quiet(self):
        self.assertEqual((PHASES, BLOCKS_PER_PHASE, SAMPLES_PER_BLOCK, REPLAYS, POLL_MS),
                         (('quiet_before', 'monitored', 'quiet_after'), 6, 5, 256, 200))
        class Args:
            warmups = 5
            quiet_confirmed = True
        validate_request(Args())
        a = Args()
        a.quiet_confirmed = False
        with self.assertRaises(ValueError):
            validate_request(a)
        a.quiet_confirmed = True
        a.warmups = 6
        with self.assertRaises(ValueError):
            validate_request(a)

    def test_all_stable_does_not_promote_speed(self):
        phases = {n: phase(0.4) for n in PHASES}
        d = decide_wall_stability(phases, {'samples': [{}] * 8})
        self.assertEqual(d['classification'], 'WALL_STABLE_IN_ALL_PHASES')
        self.assertEqual(d['status'], 'WALL_CONTROL_STABLE_DIAGNOSTIC_ONLY')
        self.assertTrue(d['not_gpu_event_stability_gate'])
        self.assertTrue(d['not_candidate_speed_gate'])

    def test_monitor_association_not_causality(self):
        phases = {'quiet_before': phase(0.4), 'quiet_after': phase(0.4),
                  'monitored': phase(0.4)}
        phases['monitored']['blocks'][5]['median_ms_per_replay'] = 0.47
        d = decide_wall_stability(phases, {'samples': [{}] * 8})
        self.assertEqual(d['classification'], 'MONITORED_PHASE_VARIATION_ONLY')
        self.assertTrue(d['no_causal_attribution'])
        self.assertEqual(d['status'], 'WALL_CONTROL_UNSTABLE_DIAGNOSTIC_ONLY')

    def test_quiet_interphase_drift_and_nan_rejected(self):
        phases = {'quiet_before': phase(0.4), 'monitored': phase(0.4),
                  'quiet_after': phase(0.46)}
        d = decide_wall_stability(phases, {'samples': [{}] * 8})
        self.assertEqual(d['classification'], 'QUIET_PHASE_VARIATION_OR_INTER_PHASE_DRIFT')
        phases['quiet_after']['blocks'][0]['samples_ms_per_replay'][0] = math.nan
        with self.assertRaises(ValueError):
            decide_wall_stability(phases, {'samples': []})

    def test_monitor_row_with_missing_wddm_fields(self):
        sample = _parse_smi_row('2026/11/15 19:20:01.000, P1, 2870, 13801, 62, 165.77, 75, N/A')
        self.assertEqual(sample['sm_clock_mhz'], 2870.0)
        self.assertIsNone(sample['memory_util_pct'])
        self.assertIsNone(_parse_smi_row('unexpected output'))


class SourceAndRunner(unittest.TestCase):
    def test_no_cuda_events_or_candidates(self):
        code = (HERE / 'benchmark_h3_tile_quiet_environment.py').read_text('utf-8')
        ast.parse(code)
        self.assertEqual(code.count('with torch.cuda.graph(sentinel):'), 1)
        self.assertNotIn('with torch.cuda.graph(fresh):', code)
        self.assertNotIn('torch.cuda.Event', code)
        self.assertNotIn('start.elapsed_time', code)
        self.assertNotIn('full_far_vb_cg_partial', code)
        self.assertNotIn('MASK_G=', code)
        self.assertIn('PASS_TWO_BANK_ORIGINAL_TILE_MERGE_BIT_EXACT', code)
        self.assertIn('torch.cuda.synchronize(device)', code)
        self.assertIn('receipt_monotonic_ns', code)
        self.assertIn("'candidate_timed': False", code)
        self.assertLess(code.index('actual != args.expect_head:'),
                        code.index('report.update(run_gpu(args))'))
        self.assertNotIn('import torch\n', code.split('def run_gpu(args):')[0])

    def test_one_json_and_runner_contract(self):
        code = (HERE / 'run_h3_tile_quiet_environment_safe.ps1').read_text('utf-8')
        for phrase in ('git clone --no-checkout --no-hardlinks',
                       'git -C $tmp switch --detach $ExactHead',
                       'test_h3_tile_quiet_environment_cpu.py',
                       'flat_tile_quiet_environment.json',
                       'Get-NetTCPConnection -LocalPort 8188',
                       'IClosedOtherGpuWorkloads',
                       "candidate_timed -ne $false"):
            self.assertIn(phrase, code)

    def test_runner_invocation_is_single_line_and_powershell_parses(self):
        # Guard against PowerShell treating --expect-head as a new statement.
        code = (HERE / 'run_h3_tile_quiet_environment_safe.ps1').read_text('utf-8')
        self.assertIn(
            "& $py -B (Join-Path $tmp 'tools\\benchmark_h3_tile_quiet_environment.py') --expect-head $ExactHead --quiet-confirmed --output $output --warmups 5",
            code)
        exe = shutil.which('powershell.exe') or shutil.which('pwsh')
        if exe is None:
            self.skipTest('PowerShell parser unavailable on this host')
        path = str((HERE / 'run_h3_tile_quiet_environment_safe.ps1').resolve())
        path = path.replace("'", "''")
        script = (
            "$tokens=$null; $parseErrors=$null; "
            "[void][System.Management.Automation.Language.Parser]::ParseFile("
            + "'" + path + "', [ref]$tokens, [ref]$parseErrors); "
            "if($parseErrors.Count -gt 0){"
            "$parseErrors | Out-String | Write-Host; exit 1}"
        )
        p = subprocess.run(
            [exe, '-NoProfile', '-NonInteractive', '-Command', script],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_wrong_head_fails_before_cuda(self):
        with tempfile.TemporaryDirectory(prefix='htrc-quiet-check-') as temp:
            root = Path(temp)
            tools = root / 'tools'
            tools.mkdir()
            for name in ('benchmark_h3_tile_quiet_environment.py',
                         'benchmark_h3_full_mask_g_numeric.py',
                         'benchmark_h3_far_masked_loads_numeric.py',
                         'analyze_h3_far_exclusion.py',
                         'profile_h3_far_exclusion.py'):
                shutil.copy2(HERE / name, tools / name)
            subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=HTRC-test',
                            '-c', 'user.email=test@example.invalid',
                            'commit', '--allow-empty', '-q', '-m', 'init'],
                           check=True, capture_output=True)
            output = root / 'one.json'
            p = subprocess.run([sys.executable, '-B',
                                str(tools / 'benchmark_h3_tile_quiet_environment.py'),
                                '--expect-head', '0'*40, '--quiet-confirmed',
                                '--output', str(output)], capture_output=True, text=True)
            data = json.loads(output.read_text('utf-8'))
            self.assertEqual(p.returncode, 2)
            self.assertEqual(data['status'], 'FAIL_CLOSED')
            self.assertFalse(data['completed'])
            self.assertEqual(data['phase_blocks'], {})
            self.assertFalse(data['candidate_timed'])
            self.assertIn('Exact HEAD mismatch', data['error'])


if __name__ == '__main__':
    unittest.main()
