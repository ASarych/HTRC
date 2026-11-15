param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead
)

# Research-only immutable source clone, fail-closed numeric gate; NO speed capture.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_far_masked_loads_numeric.json'
$tmp = 'D:\HTRC-far-masked-numeric-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'Original research checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'Pinned CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before the numeric gate'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender before the numeric gate'
}
if (Test-Path -LiteralPath $tmp) { throw 'Temporary checkout already exists' }
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }
try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Git origin not configured' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable local clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Restore GitHub origin failed' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research HEAD fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Exact commit unavailable' }
    if ((git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) {
        throw 'Exact HEAD mismatch'
    }
    & $py -B -m unittest discover -s (Join-Path $tmp 'tools') `
        -p 'test_h3_far_masked_loads_numeric_cpu.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'CPU/source tests failed; CUDA not started' }
    & $py -B (Join-Path $tmp 'tools\benchmark_h3_far_masked_loads_numeric.py') `
        --expect-head $ExactHead --output $output
    $code = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No new ONE JSON result produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.source_head -ne $ExactHead -or
        $report.banks.Count -ne 2 -or
        $report.speed_capture_blocks.Count -ne 0 -or
        $report.no_cuda_event_timing -ne $true -or
        $report.speed_measurement -ne 'UNRUN_NUMERIC_GATE_ONLY' -or
        $report.first_complete_graph_eager_bit_exact -ne $true -or
        $report.bank_b_complete_graph_eager_bit_exact -ne $true -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY') {
        throw 'Incomplete two-bank parity gate, source mismatch, or forbidden speed measurement'
    }
    if ($report.status -eq 'FAIL_CLOSED' -and
        $report.parity_admission -in @('NO_GO_RESEARCH_DENSE_MIRROR', 'NO_GO_MASKED_FAR_BIT_EXACT')) {
        if ($code -ne 2 -or $report.completed -ne $false) {
            throw 'Invalid scientific numeric NO_GO reporting'
        }
        $bad = @($report.banks | Where-Object {
            $_.variant_parity.mirror.all_bit_exact -ne $true -or
            $_.variant_parity.mask_both.all_bit_exact -ne $true
        })
        if ($bad.Count -lt 1) { throw 'No mismatch evidence for numeric NO_GO' }
        Write-Host "HTRC masked far numeric NO_GO: $($report.parity_admission)"
    }
    elseif ($report.status -eq 'PASS_MASKED_FAR_NUMERIC_ONLY') {
        if ($code -ne 0 -or $report.completed -ne $true -or
            $report.research_dense_mirror_bit_exact -ne $true -or
            $report.admitted_far_only_variants.Count -lt 1) {
            throw 'Invalid masked far-only numerical admission'
        }
        Write-Host "HTRC masked far-only PASS: $($report.admitted_far_only_variants -join ',')"
    }
    else {
        throw "Unexpected numeric FAIL_CLOSED: $($report.error)"
    }
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
