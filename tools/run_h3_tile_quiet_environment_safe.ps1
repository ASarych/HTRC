param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [switch]$IClosedOtherGpuWorkloads
)

# One-Graph unprofiled wall stability with optional nvidia-smi telemetry.
# No production code, NVIDIA driver settings, or GPU power/clocks are changed.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_tile_quiet_environment.json'
$tmp = 'D:\HTRC-quiet-environment-' + ([guid]::NewGuid().ToString('N').Substring(0,10))
if (-not $IClosedOtherGpuWorkloads) {
    throw 'Confirm other GPU workloads are closed with -IClosedOtherGpuWorkloads'
}
if (-not (Test-Path -LiteralPath $repo -PathType Container)) {
    throw 'Original HTRC research checkout missing'
}
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw 'CUDA Python missing'
}
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before GPU diagnostic'
}
if (Get-Process -Name blender,ollama,ollama_llama_server -ErrorAction SilentlyContinue) {
    throw 'Close Blender/Ollama before GPU diagnostic'
}
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }
try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Git origin missing' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Git origin setup failed' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research branch fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Exact source commit unavailable' }
    if ((git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) { throw 'Exact HEAD mismatch' }

    foreach ($suite in @('test_h3_full_mask_g_numeric_cpu.py',
                         'test_h3_tile_control_stability_cpu.py',
                         'test_h3_tile_quiet_environment_cpu.py')) {
        & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p $suite -v
        if ($LASTEXITCODE -ne 0) { throw "CPU suite $suite failed; CUDA not started" }
    }
    & $py -B (Join-Path $tmp 'tools\benchmark_h3_tile_quiet_environment.py') --expect-head $ExactHead --quiet-confirmed --output $output --warmups 5
    if ($LASTEXITCODE -ne 0) { throw 'Research diagnostic FAIL_CLOSED: inspect ONE JSON' }
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No fresh ONE JSON produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.schema -ne 'htrc-tile-quiet-environment-v1' -or
        $report.source_head -ne $ExactHead -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
        $report.h3_model_or_video -ne 'UNRUN' -or
        $report.candidate_timed -ne $false -or
        $report.completed -ne $true -or
        $report.numeric_gate -ne 'PASS_TWO_BANK_ORIGINAL_TILE_MERGE_BIT_EXACT' -or
        $report.phase_blocks.quiet_before.blocks.Count -ne 6 -or
        $report.phase_blocks.monitored.blocks.Count -ne 6 -or
        $report.phase_blocks.quiet_after.blocks.Count -ne 6 -or
        $report.telemetry.available -ne $true -or
        $report.telemetry.samples.Count -lt 5 -or
        $report.sentinel_initial_parity.full_window_bit_exact -ne $true -or
        $report.status -notin @('WALL_CONTROL_STABLE_DIAGNOSTIC_ONLY',
                                'WALL_CONTROL_UNSTABLE_DIAGNOSTIC_ONLY')) {
        throw 'Incomplete source/numeric/telemetry report; inspect ONE JSON'
    }
    foreach ($name in @('quiet_before', 'monitored', 'quiet_after')) {
        foreach ($block in $report.phase_blocks.$name.blocks) {
            if ($block.samples.Count -ne 5) {
                throw 'Missing wall-clock samples'
            }
        }
        if ($report.phase_blocks.$name.post_phase_parity.full_window_bit_exact -ne $true) {
            throw 'Sentinel original bits changed'
        }
    }
    Write-Host "HTRC original Tile wall stability: $($report.status)"
    Write-Host "Classification: $($report.wall_stability.classification)"
    Write-Host "NVIDIA telemetry samples: $($report.telemetry.samples.Count)"
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
exit 0
