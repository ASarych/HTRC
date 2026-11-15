param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [ValidateRange(7,12)]
    [int]$Rounds = 7
)

# Isolated original Tile, four independent control graphs and one persistent
# original control sentinel. NEVER a candidate-speed or production admission.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_tile_control_stability.json'
$tmp = 'D:\HTRC-control-stability-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not (Test-Path -LiteralPath $repo -PathType Container)) {
    throw 'Original HTRC research checkout missing'
}
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw 'CUDA Python missing'
}
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before GPU diagnostic'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender before GPU diagnostic'
}
if (Test-Path -LiteralPath $tmp) { throw 'Disposable checkout already exists' }
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
                         'test_h3_tile_control_stability_cpu.py')) {
        & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p $suite -v
        if ($LASTEXITCODE -ne 0) { throw "CPU suite $suite failed; CUDA not started" }
    }

    & $py -B (Join-Path $tmp 'tools\benchmark_h3_tile_control_stability.py') `
        --expect-head $ExactHead --output $output --rounds $Rounds `
        --warmups 5 --replays-per-sample 16
    $code = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No fresh ONE JSON produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.schema -ne 'htrc-tile-control-stability-v1' -or
        $report.source_head -ne $ExactHead -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
        $report.h3_model_or_video -ne 'UNRUN' -or
        $report.candidate_timed -ne $false) {
        throw 'Invalid source or control-only contract; inspect ONE JSON'
    }
    if ($report.status -notin @('PASS_CONTROL_STABILITY_ONLY',
                                'UNSTABLE_CONTROL_ONLY')) {
        throw "Diagnostic FAIL_CLOSED: $($report.status) $($report.error)"
    }
    if ($code -ne 0 -or $report.completed -ne $true -or
        $report.numeric_gate -ne 'PASS_TWO_BANK_ORIGINAL_TILE_MERGE_BIT_EXACT' -or
        $report.banks.Count -ne 2 -or
        $report.capture_blocks.Count -ne 4 -or
        $report.control_stability.status -ne $report.status -or
        $report.speed_measurement -ne 'CONTROL_ONLY_SENTINEL_BRACKET_EVENTS' -or
        $report.sentinel_initial_parity.full_window_bit_exact -ne $true) {
        throw 'Incomplete control-only stability gate; inspect ONE JSON'
    }
    if ($report.banks[0].selection_mask_sha256 -eq $report.banks[1].selection_mask_sha256) {
        throw 'Independent QKV selection masks unexpectedly identical'
    }
    foreach ($bank in $report.banks) {
        if ($bank.original_full_tile_merge_graph_bit_exact -ne $true) {
            throw 'Baseline original Merge not exact'
        }
    }
    foreach ($block in $report.capture_blocks) {
        if ($block.arm -ne 'control' -or
            $block.independent_capture -ne $true -or
            $block.replays_per_gpu_event -ne 16 -or
            $block.samples_per_capture -ne $Rounds -or
            $block.pre_capture_validation.full_window_bit_exact -ne $true -or
            $block.post_warmup_validation.full_window_bit_exact -ne $true -or
            $block.post_timing_validation.full_window_bit_exact -ne $true -or
            $block.sentinel_post_timing_validation.full_window_bit_exact -ne $true) {
            throw 'Incomplete numerical or capture gate'
        }
        foreach ($phase in @('sentinel_before','fresh_control','sentinel_after')) {
            if ($block.$phase.samples_ms_per_tile.Count -ne $Rounds -or
                $block.$phase.median_ms_per_tile -le 0) {
                throw "Missing phase $phase timing"
            }
        }
    }

    Write-Host "HTRC original Tile stability: $($report.status)"
    Write-Host "Classification: $($report.control_stability.classification)"
    Write-Host "Fresh capture spread: $($report.control_stability.fresh_capture_spread_pct) %"
    Write-Host "Sentinel spread: $($report.control_stability.sentinel_all_spread_pct) %"
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
exit 0
