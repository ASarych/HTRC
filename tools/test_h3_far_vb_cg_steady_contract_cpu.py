"""Source/runner safety contract for opt-in steady far VB .cg A/B/B/A."""
from __future__ import annotations
import ast
from pathlib import Path
import shutil
import subprocess
import unittest

HERE = Path(__file__).resolve().parent


class SteadyABBAContract(unittest.TestCase):
    def test_gate_before_candidate_timing_and_opt_in_only(self):
        s = (HERE / 'benchmark_h3_far_vb_cg_tile_abba.py').read_text('utf-8')
        ast.parse(s)
        for exact in (
            "from htrc_tile_steady_gate import run_steady_precondition",
            "if getattr(args, 'steady_precondition', False):",
            "steady_report = run_steady_precondition(torch, steady_graph, device)",
            "steady_report['post_control_parity'] = steady_parity",
            "if not steady_report['decision']['admitted']:",
            "'INCONCLUSIVE_PRECONDITION'",
            "'speed_measurement': 'UNRUN_UNSTABLE_CONTROL'",
            "parser.add_argument('--steady-precondition', action='store_true')",
            "'steady_precondition_requested': bool(args.steady_precondition)",
            "'INCONCLUSIVE_PRECONDITION') else 2",
        ):
            self.assertIn(exact, s)
        self.assertLess(
            s.index("steady_report = run_steady_precondition("),
            s.index('for index, arm in enumerate(ARMS'))
        self.assertLess(
            s.index("if not steady_report['decision']['admitted']:"),
            s.index('for index, arm in enumerate(ARMS'))
        self.assertLess(s.index('actual != args.expect_head:'), s.index('report.update(run_gpu(args))'))
        self.assertIn("ARMS = ('control', 'far_vb_cg', 'far_vb_cg', 'control')", s)
        self.assertGreater(s.index("from htrc_tile_steady_gate import run_steady_precondition"),
                           s.index("def run_gpu(args):"))
        self.assertLess(s.index('actual != args.expect_head:'),
                        s.index('report.update(run_gpu(args))'))
        self.assertIn('MIN_REDUCTION_PCT = 5.0', s)
        self.assertIn('MAX_INTER_CAPTURE_SPREAD_PCT = 3.0', s)
        self.assertIn("return _launch_full_vb_cache(states, research_kernel, out, cache_vb='.cg')", s)
        self.assertNotIn('import torch\n', s.split('def run_gpu(args):')[0])

    def test_runner_single_json_before_gpu_cpu_suites_and_no_installed_changes(self):
        s = (HERE / 'run_h3_far_vb_cg_steady_tile_abba_safe.ps1').read_text('utf-8')
        for exact in (
            'git clone --no-checkout --no-hardlinks',
            'git -C $tmp switch --detach $ExactHead',
            'test_h3_far_vb_cg_tile_abba_cpu.py',
            'test_htrc_tile_steady_gate_cpu.py',
            'test_h3_far_vb_cg_steady_contract_cpu.py',
            'flat_full_far_vb_cg_steady_tile_abba.json',
            '--replays-per-sample 16 --steady-precondition',
            "report.status -eq 'INCONCLUSIVE_PRECONDITION'",
            "report.steady_precondition.blocks.Count -ne 24",
            "report.steady_precondition.decision.admitted -ne $false",
            "report.steady_precondition.decision.admitted -ne $true",
            'post_control_parity.full_window_bit_exact',
            'IClosedOtherGpuWorkloads',
            'Get-NetTCPConnection -LocalPort 8188',
        ):
            self.assertIn(exact, s)
        self.assertNotIn("benchmark_h3_far_vb_cg_tile_abba.py') `\n", s)
        self.assertLess(s.index('test_htrc_tile_steady_gate_cpu.py'),
                        s.index("benchmark_h3_far_vb_cg_tile_abba.py') --expect-head"))

    def test_runner_powershell_parser_if_available(self):
        exe = shutil.which('powershell.exe') or shutil.which('pwsh')
        if not exe:
            self.skipTest('PowerShell parser unavailable on this host')
        path = str((HERE / 'run_h3_far_vb_cg_steady_tile_abba_safe.ps1').resolve()).replace("'", "''")
        script = ("$tok=$null;$err=$null;"
                  "[void][System.Management.Automation.Language.Parser]::ParseFile("
                  + "'" + path + "',[ref]$tok,[ref]$err);"
                  "if($err.Count){$err|Out-String|Write-Host;exit 1}")
        p = subprocess.run(
            [exe, '-NoProfile', '-NonInteractive', '-Command', script],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


if __name__ == '__main__':
    unittest.main()
