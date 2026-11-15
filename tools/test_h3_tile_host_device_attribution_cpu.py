"""CPU-only gates for original Tile host/GPU startup attribution diagnostic."""
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
from benchmark_h3_tile_host_device_attribution import (
    BLOCKS, PHASES, REPLAYS_PER_SAMPLE, SAMPLES_PER_BLOCK, SPREAD_PCT,
    STATUS, spread_pct, summarize_protocol, validate_request,
)


def fake_phases(startup_spike=False, settled_spike=False):
    phases = {}
    for name in PHASES:
        paired = name != 'final_wall'
        blocks = []
        for i in range(BLOCKS[name]):
            wall = (0.43 if startup_spike and name == 'startup_host_gpu' and i == 1
                    else 0.44 if settled_spike and name == 'final_wall' and i == 4
                    else 0.409 + i * 0.00001)
            samples = [{
                'event_instrumented': paired,
                'wall_ms_per_replay': wall,
                'enqueue_ms_per_replay': 0.0009,
                'gpu_event_ms_per_replay': 0.408 if paired else None
            }] * SAMPLES_PER_BLOCK
            blocks.append({
                'samples': samples, 'wall_median_ms_per_replay': wall,
                'enqueue_median_ms_per_replay': 0.0009,
                'gpu_event_median_ms_per_replay': 0.408 if paired else None,
            })
        phases[name] = {'blocks': blocks}
    return phases


class PureLogic(unittest.TestCase):
    def test_pinned_protocol(self):
        self.assertEqual(PHASES, ('startup_host_gpu', 'settled_host_gpu', 'final_wall'))
        self.assertEqual(BLOCKS, {'startup_host_gpu': 12, 'settled_host_gpu': 12, 'final_wall': 6})
        self.assertEqual((REPLAYS_PER_SAMPLE, SAMPLES_PER_BLOCK, SPREAD_PCT), (256, 5, 3.0))
        class Args:
            quiet_confirmed = True
            warmups = 5
            measurement_protocol = 'host_device'
        validate_request(Args())
        a = Args()
        a.quiet_confirmed = False
        with self.assertRaises(ValueError):
            validate_request(a)
        a.quiet_confirmed = True
        a.measurement_protocol = 'candidate'
        with self.assertRaises(ValueError):
            validate_request(a)

    def test_stable_is_descriptive_only(self):
        s = summarize_protocol(fake_phases())
        self.assertEqual(s['classification'], 'INITIAL_PAIRED_AND_FINAL_UNPROFILED_WITHIN_THRESHOLD')
        self.assertEqual(s['status'], STATUS)
        self.assertTrue(s['no_proven_root_cause'])
        self.assertTrue(s['not_candidate_speed_gate'])

    def test_startup_transient_without_posthoc_exclusion(self):
        s = summarize_protocol(fake_phases(startup_spike=True))
        self.assertEqual(s['classification'], 'INITIAL_VARIATION_FINAL_UNPROFILED_STABLE')
        self.assertGreater(s['original_wall_initial_spread_pct'], 3)
        self.assertLess(s['original_wall_final_spread_pct'], 3)
        self.assertEqual(len(s['phases']['startup_host_gpu']['wall_block_medians_ms_per_replay']), 12)

    def test_final_phase_variation_refuses_stability(self):
        s = summarize_protocol(fake_phases(settled_spike=True))
        self.assertEqual(s['classification'], 'VARIATION_PERSISTS_IN_FINAL_UNPROFILED_PHASE')

    def test_native_sample_schema_matches_protocol_fixture(self):
        # Exercise the REAL _time_sample() payload, not a hand-written
        # alternative field name. Fake CUDA calls avoid GPU hardware.
        from benchmark_h3_tile_host_device_attribution import _time_sample

        class FakeGraph:
            def __init__(self):
                self.replays = 0

            def replay(self):
                self.replays += 1

        class FakeEvent:
            def record(self):
                pass

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 102.4

        class FakeTorch:
            class cuda:
                @staticmethod
                def synchronize(device):
                    pass

        graph = FakeGraph()
        paired = _time_sample(FakeTorch, graph, 'cuda:0', FakeEvent(), FakeEvent())
        self.assertEqual(graph.replays, REPLAYS_PER_SAMPLE)
        self.assertIs(paired['event_instrumented'], True)
        self.assertNotIn('instrumented', paired)
        self.assertGreater(paired['gpu_event_ms_per_replay'], 0)

        quiet = _time_sample(FakeTorch, graph, 'cuda:0')
        self.assertIs(quiet['event_instrumented'], False)
        self.assertIsNone(quiet['gpu_event_ms_per_replay'])
        self.assertEqual(graph.replays, REPLAYS_PER_SAMPLE * 2)

    def test_missing_sample_schema_is_valueerror_not_keyerror(self):
        p = fake_phases()
        del p['startup_host_gpu']['blocks'][0]['samples'][0]['event_instrumented']
        with self.assertRaisesRegex(ValueError, 'Instrumented sample mismatch'):
            summarize_protocol(p)

    def test_invalid_incomplete_and_instrument_mismatch(self):
        with self.assertRaises(ValueError):
            spread_pct([0.4, math.nan])
        with self.assertRaises(ValueError):
            summarize_protocol({'startup_host_gpu': fake_phases()['startup_host_gpu']})
        p = fake_phases()
        p['settled_host_gpu']['blocks'][0]['samples'][0]['event_instrumented'] = False
        with self.assertRaises(ValueError):
            summarize_protocol(p)


class SourceContract(unittest.TestCase):
    def test_original_math_source_only_and_sha_before_cuda(self):
        s = (HERE / 'benchmark_h3_tile_host_device_attribution.py').read_text('utf-8')
        ast.parse(s)
        old = (HERE / 'benchmark_h3_tile_quiet_environment.py').read_text('utf-8')
        ast.parse(old)
        self.assertIn("getattr(args, 'measurement_protocol', None) == 'host_device'", old)
        self.assertIn('collect_control_probe(', old)
        self.assertIn("sample.get('event_instrumented') is not paired", s)
        self.assertEqual(old.count('with torch.cuda.graph(sentinel):'), 1)
        for wanted in ("'current_pid': os.getpid()", "torch.cuda.Event(enable_timing=True)",
                       "stop_event.synchronize()", "start_event.elapsed_time(stop_event)",
                       "graph.replay()", "STATUS = 'RECORDED_ORIGINAL_TILE_HOST_DEVICE_DIAGNOSTIC_ONLY'",
                       "'candidate_timed': False", "'h3_model_or_video': 'UNRUN'"):
            self.assertIn(wanted, s)
        self.assertLess(s.index('actual != args.expect_head:'), s.index('result.update(run_gpu(args))'))
        self.assertNotIn('import torch\n', s.split('def main():')[0])
        self.assertNotIn('full_far_vb_cg_partial', s)
        self.assertNotIn('MASK_G=', s)

    def test_runner_has_one_line_invocation_and_parser(self):
        s = (HERE / 'run_h3_tile_host_device_attribution_safe.ps1').read_text('utf-8')
        self.assertIn('$failed.source_head -eq $ExactHead', s)
        self.assertIn('Write-Host $failed.traceback_tail', s)
        for wanted in (
            "benchmark_h3_tile_host_device_attribution.py') --expect-head $ExactHead",
            'test_h3_tile_host_device_attribution_cpu.py',
            'flat_tile_host_device_attribution.json',
            'git clone --no-checkout --no-hardlinks',
            'git -C $tmp switch --detach $ExactHead',
            'IClosedOtherGpuWorkloads',
        ):
            self.assertIn(wanted, s)
        exe = shutil.which('powershell.exe') or shutil.which('pwsh')
        if exe:
            path = str((HERE / 'run_h3_tile_host_device_attribution_safe.ps1').resolve())
            path = path.replace("'", "''")
            script = ("$tok=$null;$err=$null;"
                      "[void][System.Management.Automation.Language.Parser]::ParseFile("
                      + "'" + path + "',[ref]$tok,[ref]$err);"
                      "if($err.Count){$err|Out-String|Write-Host;exit 1}")
            run = subprocess.run([exe, '-NoProfile', '-NonInteractive', '-Command', script],
                                 capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_wrong_head_fails_without_cuda(self):
        with tempfile.TemporaryDirectory(prefix='htrc-host-device-') as tmp:
            root = Path(tmp)
            tools = root / 'tools'
            tools.mkdir()
            for name in ('benchmark_h3_tile_host_device_attribution.py',
                         'benchmark_h3_tile_quiet_environment.py',
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
            out = root / 'fail.json'
            run = subprocess.run(
                [sys.executable, '-B', str(tools / 'benchmark_h3_tile_host_device_attribution.py'),
                 '--expect-head', '0' * 40, '--quiet-confirmed', '--output', str(out)],
                capture_output=True, text=True)
            result = json.loads(out.read_text('utf-8'))
            self.assertEqual(run.returncode, 2)
            self.assertEqual(result['status'], 'FAIL_CLOSED')
            self.assertFalse(result['completed'])
            self.assertEqual(result['phase_blocks'], {})
            self.assertIn('Exact HEAD mismatch', result['error'])


if __name__ == '__main__':
    unittest.main()
