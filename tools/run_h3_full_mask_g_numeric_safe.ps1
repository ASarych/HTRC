param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead
)

# Isolated strict full-window numeric experiment; NOT a performance A/B.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_full_mask_g_numeric.json'
$tmp = 'D:\HTRC-full-mask-g-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'HTRC checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'ComfyUI CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before exact numeric gate'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender before exact numeric gate'
}
if (Test-Path -LiteralPath $tmp) { throw 'Temporary exact checkout collision' }
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Missing original remote' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable local clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Cannot restore GitHub remote' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research HEAD fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Exact HEAD checkout failed' }
    if ((git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) { throw 'Exact HEAD mismatch' }

    & $py -B -m unittest discover -s (Join-Path $tmp 'tools') `
        -p 'test_h3_full_mask_g_numeric_cpu.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'CPU/source contract failed; GPU not started' }

    & $py -B (Join-Path $tmp 'tools\benchmark_h3_full_mask_g_numeric.py') `
        --expect-head $ExactHead --output $output
    $code = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No fresh single JSON result created'
    }
    $r = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($r.schema -ne 'htrc-full-mask-g-numeric-v1' -or
        $r.source_head -ne $ExactHead -or
        $r.speed_measurement -ne 'UNRUN_NUMERIC_ONLY' -or
        $r.no_cuda_event_timing -ne $true -or
        $r.speed_capture_blocks.Count -ne 0 -or
        $r.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY') {
        throw 'Source/speed safety gate invalid; inspect ONE JSON'
    }
    if ($r.status -eq 'PASS_FULL_MASK_G_NUMERIC_ONLY') {
        if ($code -ne 0 -or $r.completed -ne $true -or
            $r.parity_admission -ne 'PASS_FULL_WINDOW_PARTIALS_AND_MERGE_ONLY' -or
            $r.banks.Count -ne 2 -or
            $r.mirror_full_numeric_bit_exact -ne $true -or
            $r.mask_g_full_numeric_bit_exact -ne $true -or
            $r.baseline_window_graph_parity -ne $true -or
            $r.first_complete_graph_eager_bit_exact -ne $true -or
            $r.bank_b_complete_graph_eager_bit_exact -ne $true) {
            throw 'Incomplete strict two-bank full PASS'
        }
        foreach ($bank in $r.banks) {
            if ($bank.baseline_full_tile_merge_equals_graph_window.bit_exact -ne $true -or
                $bank.variants.mirror.full_window_bit_exact -ne $true -or
                $bank.variants.mask_g.full_window_bit_exact -ne $true) {
                throw 'One window parity missing despite PASS status'
            }
        }
        Write-Host 'HTRC mask_g full Tile+Merge window: NUMERIC PASS (speed UNRUN)'
    }
    elseif ($r.status -eq 'FAIL_CLOSED' -and
        $r.parity_admission -in @('NO_GO_FULL_RESEARCH_MIRROR','NO_GO_FULL_MASK_G_BIT_EXACT')) {
        if ($code -ne 2 -or $r.completed -ne $false -or $r.banks.Count -ne 2) {
            throw 'Expected scientific NO_GO lacks complete evidence'
        }
        Write-Host "HTRC full mask_g NUMERIC NO_GO: $($r.parity_admission) (speed UNRUN)"
    }
    else {
        throw "GPU diagnostic failed: $($r.error); inspect ONE JSON"
    }
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
