param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead
)

# Research-only original PASS1 vs compacted far QK/PV numeric test.
# Never modify ComfyUI, production, active source checkout or runtime.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_packed_far_numeric.json'
$tmp = 'D:\HTRC-packed-far-numeric-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'Original research checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before numerical experiment'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender before numerical experiment'
}
if (Test-Path -LiteralPath $tmp) { throw 'Temporary checkout already exists' }
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Original Git remote missing' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Local disposable clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Cannot set remote' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Cannot fetch exact research branch' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Exact HEAD absent from disposable checkout' }
    if ((git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) { throw 'Exact HEAD mismatch' }

    & $py -B -m unittest discover -s (Join-Path $tmp 'tools') `
        -p 'test_h3_packed_far_numeric_cpu.py' -v
    if ($LASTEXITCODE -ne 0) {
        throw 'Packed far CPU/source gates failed; GPU not started'
    }
    & $py -B (Join-Path $tmp 'tools\benchmark_h3_packed_far_numeric.py') `
        --expect-head $ExactHead --output $output
    $code = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No fresh single JSON result was produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.source_head -ne $ExactHead -or
        $report.banks.Count -ne 2 -or
        $report.speed_capture_blocks.Count -ne 0 -or
        $report.speed_measurement -ne 'UNRUN_NUMERIC_GATE_ONLY' -or
        $report.no_cuda_event_timing -ne $true -or
        $report.first_complete_graph_eager_bit_exact -ne $true -or
        $report.bank_b_complete_graph_eager_bit_exact -ne $true -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY') {
        throw 'Missing strict two-bank numeric/stop gate; inspect ONE JSON'
    }
    if ($report.status -eq 'FAIL_CLOSED' -and
        $report.parity_admission -eq 'NO_GO_FAR_BIT_EXACT') {
        if ($code -ne 2 -or $report.completed -ne $false -or
            $report.far_partial_bit_exact -ne $false) {
            throw 'Invalid numeric NO_GO exit/status'
        }
        $bad = @($report.banks | Where-Object { $_.far_partial_all_bit_exact -ne $true })
        if ($bad.Count -lt 1) { throw 'NO_GO lacks mismatch evidence' }
        Write-Host 'HTRC packed far NUMERIC NO_GO (no speed measured)'
    }
    elseif ($report.status -eq 'PASS_FAR_NUMERIC_ONLY') {
        if ($code -ne 0 -or $report.completed -ne $true -or
            $report.far_partial_bit_exact -ne $true -or
            $report.parity_admission -ne 'PASS_FAR_ONLY_NEXT_FULL_GATE_REQUIRED') {
            throw 'Invalid far-only numeric acceptance'
        }
        Write-Host 'HTRC packed far PASS far-only; full near+far still UNRUN'
    }
    else {
        throw "GPU diagnostic FAIL_CLOSED unexpectedly: $($report.error); inspect ONE JSON"
    }
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
