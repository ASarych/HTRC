param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [switch]$IClosedOtherGpuWorkloads
)

# Research-only: one original Tile, startup wall then instrumented GPU Event then wall.
# No driver/clock/power changes, candidate code, model or video inference.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_tile_host_device_attribution.json'
$tmp = 'D:\HTRC-host-device-' + ([guid]::NewGuid().ToString('N').Substring(0,10))
if (-not $IClosedOtherGpuWorkloads) {
    throw 'Close other GPU jobs, then explicitly provide -IClosedOtherGpuWorkloads'
}
if (-not (Test-Path -LiteralPath $repo -PathType Container)) {
    throw 'Original HTRC research checkout missing'
}
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw 'CUDA Python missing'
}
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close running ComfyUI server before GPU diagnostic'
}
if (Get-Process -Name blender,ollama,ollama_llama_server -ErrorAction SilentlyContinue) {
    throw 'Close Blender/Ollama before GPU diagnostic'
}
# This runs BEFORE launching the benchmark Python. A same-path client that
# appears later may be this experiment's own PID; the report records os.getpid().
$clientLines = @(& nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>$null)
if ($LASTEXITCODE -ne 0) {
    throw 'Cannot inspect GPU compute clients before starting diagnostic'
}
$comfyPythonClients = @($clientLines | Where-Object {
    $_ -like '*\ComfyUI-Installs\*\standalone-env\python*.exe*' -or
    $_ -like '*\ComfyUI-Installs\*\ComfyUI\.venv\Scripts\python*.exe*'
})
if ($comfyPythonClients.Count -gt 0) {
    throw "Background Comfy GPU client already present before benchmark: $($comfyPythonClients -join '; ')"
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
                         'test_h3_tile_quiet_environment_cpu.py',
                         'test_h3_tile_host_device_attribution_cpu.py')) {
        & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p $suite -v
        if ($LASTEXITCODE -ne 0) { throw "CPU suite $suite failed; CUDA not started" }
    }

    & $py -B (Join-Path $tmp 'tools\benchmark_h3_tile_host_device_attribution.py') --expect-head $ExactHead --quiet-confirmed --output $output
    if ($LASTEXITCODE -ne 0) { throw 'Research diagnostic FAIL_CLOSED: inspect ONE JSON' }
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No fresh ONE JSON produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.schema -ne 'htrc-tile-host-device-v1' -or
        $report.source_head -ne $ExactHead -or
        $report.status -ne 'RECORDED_ORIGINAL_TILE_HOST_DEVICE_DIAGNOSTIC_ONLY' -or
        $report.completed -ne $true -or
        $report.numeric_gate -ne 'PASS_TWO_BANK_ORIGINAL_TILE_MERGE_BIT_EXACT' -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
        $report.h3_model_or_video -ne 'UNRUN' -or
        $report.candidate_timed -ne $false -or
        $report.process_identity.current_pid -le 0 -or
        $report.sentinel_initial_parity.full_window_bit_exact -ne $true -or
        $report.phase_blocks.startup_wall.blocks.Count -ne 12 -or
        $report.phase_blocks.paired_host_gpu.blocks.Count -ne 12 -or
        $report.phase_blocks.settled_wall.blocks.Count -ne 6) {
        throw 'Invalid research-only numeric or measurement contract; inspect ONE JSON'
    }
    foreach ($name in @('startup_wall','paired_host_gpu','settled_wall')) {
        if ($report.phase_blocks.$name.post_phase_parity.full_window_bit_exact -ne $true) {
            throw "Original Tile numeric parity failed after $name"
        }
        $count = if ($name -eq 'settled_wall') { 6 } else { 12 }
        if ($report.phase_blocks.$name.blocks.Count -ne $count) { throw "Missing $name blocks" }
        foreach ($block in $report.phase_blocks.$name.blocks) {
            if ($block.samples.Count -ne 5) { throw "Missing $name samples" }
            foreach ($sample in $block.samples) {
                if ($sample.wall_ms_per_replay -le 0 -or $sample.enqueue_ms_per_replay -le 0) {
                    throw "Invalid $name CPU timing"
                }
                if ($name -eq 'paired_host_gpu') {
                    if ($sample.event_instrumented -ne $true -or $sample.gpu_event_ms_per_replay -le 0) {
                        throw 'Missing paired CUDA Event GPU timing'
                    }
                } elseif ($sample.event_instrumented -ne $false -or $null -ne $sample.gpu_event_ms_per_replay) {
                    throw 'Unexpected event instrumentation in wall-only phase'
                }
            }
        }
    }
    Write-Host "HTRC host/GPU original Tile diagnostic: $($report.status)"
    Write-Host "Classification: $($report.host_device_diagnostic.classification)"
    Write-Host "Self process PID: $($report.process_identity.current_pid)"
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
exit 0
