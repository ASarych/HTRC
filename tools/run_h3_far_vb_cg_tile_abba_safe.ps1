param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [ValidateRange(7,12)]
    [int]$Rounds = 7
)

# Research-only original vs far VB .cg full Tile-only A/B/B/A. No installed edits.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_full_far_vb_cg_tile_abba.json'
$tmp = 'D:\HTRC-far-vb-cg-abba-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'Original HTRC research checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'ComfyUI CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before research GPU benchmark'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender before research GPU benchmark'
}
if (Test-Path -LiteralPath $tmp) { throw 'Refuse to reuse detached checkout' }
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Git origin missing' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Git origin restore failed' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research branch fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Requested SHA missing from detached checkout' }
    if ((git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) { throw 'Exact HEAD mismatch' }

    foreach ($suite in @('test_h3_full_mask_g_numeric_cpu.py',
                         'test_h3_far_vb_cg_tile_abba_cpu.py')) {
        & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p $suite -v
        if ($LASTEXITCODE -ne 0) { throw "CPU/source gate $suite failed: CUDA not started" }
    }
    & $py -B (Join-Path $tmp 'tools\benchmark_h3_far_vb_cg_tile_abba.py') `
        --expect-head $ExactHead --output $output --rounds $Rounds `
        --warmups 5 --replays-per-sample 16
    $code = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No fresh ONE JSON produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.schema -ne 'htrc-far-vb-cg-tile-abba-v1' -or
        $report.source_head -ne $ExactHead -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
        $report.h3_model_or_video -ne 'UNRUN') {
        throw 'Invalid exact source or production gate: inspect ONE JSON'
    }
    if ($report.status -eq 'FAIL_CLOSED' -and
        $report.parity_admission -in @('NO_GO_BENCH_NUMERIC','NO_GO_CAPTURE_BIT_EXACT')) {
        if ($code -ne 2 -or $report.completed -ne $false -or
            $report.numeric_gate -eq 'UNRUN') {
            throw 'Invalid scientific numeric NO_GO; inspect report'
        }
        Write-Host 'HTRC: numeric NO_GO, no speed admitted'
    }
    elseif ($report.status -eq 'NO_GO_RESOURCE_GATE') {
        if ($code -ne 0 -or $report.completed -ne $true -or
            $report.numeric_gate -ne 'PASS_TWO_BANK_FULL_TILE_MERGE_BIT_EXACT' -or
            $report.resource_gate.admitted -ne $false -or
            $report.capture_blocks.Count -ne 0) {
            throw 'Invalid resource gate; no GPU timing should run'
        }
        Write-Host 'HTRC: resource NO_GO; no speed samples'
    }
    elseif ($report.status -in @('PASS_TILE_ONLY_SPEED_GATE',
                                 'NO_GO_TILE_ONLY','INCONCLUSIVE_TILE_ONLY')) {
        if ($code -ne 0 -or $report.completed -ne $true -or
            $report.numeric_gate -ne 'PASS_TWO_BANK_FULL_TILE_MERGE_BIT_EXACT' -or
            $report.resource_gate.admitted -ne $true -or
            $report.capture_blocks.Count -ne 4 -or
            $report.tile_only_speed.status -ne $report.status -or
            $report.speed_measurement -ne 'ONE_WINDOW_TILE_KERNEL_ABBA') {
            throw 'Invalid completed A/B/B/A or source numeric gates'
        }
        $order = @($report.capture_blocks | ForEach-Object { $_.arm }) -join ','
        if ($order -ne 'control,far_vb_cg,far_vb_cg,control' -or $report.banks.Count -ne 2) {
            throw 'Wrong capture order or QKV bank count'
        }
        foreach ($bank in $report.banks) {
            if ($bank.original_full_tile_merge_graph_bit_exact -ne $true -or
                $bank.variants.mirror.full_window_bit_exact -ne $true -or
                $bank.variants.far_vb_cg.full_window_bit_exact -ne $true) {
                throw 'Missing two-bank full numeric parity'
            }
        }
        foreach ($block in $report.capture_blocks) {
            if ($block.independent_capture -ne $true -or
                $block.gpu_event_ms_per_tile_samples.Count -ne $Rounds -or
                $block.pre_capture_validation.full_window_bit_exact -ne $true -or
                $block.post_warmup_validation.full_window_bit_exact -ne $true -or
                $block.post_timing_validation.full_window_bit_exact -ne $true) {
                throw 'Incomplete captured parity/timing gate'
            }
        }
        Write-Host "HTRC far VB .cg Tile A/B/B/A: $($report.status)"
        Write-Host "Control: $($report.tile_only_speed.control_median_ms) ms"
        Write-Host "Mask G: $($report.tile_only_speed.far_vb_cg_median_ms) ms"
    }
    else {
        throw "Unexpected benchmark failure: $($report.status); $($report.error)"
    }
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
exit 0
