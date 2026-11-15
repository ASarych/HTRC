param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead
)

# Research-only read-only Tile32 selection-mask probe; no new kernel or timed ABBA.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_far_exclusion_probe.json'
$tmp = 'D:\HTRC-far-exclusion-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'HTRC checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'ComfyUI CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before experiment'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender before experiment'
}
if (Test-Path -LiteralPath $tmp) { throw 'Refuse to reuse temporary worktree' }
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Git origin unavailable' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable local clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Origin setup failed' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research branch fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Exact research HEAD unavailable' }
    if ((git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) {
        throw 'Exact HEAD mismatch'
    }
    & $py -B -m unittest discover -s (Join-Path $tmp 'tools') `
        -p 'test_h3_far_exclusion_cpu.py' -v
    if ($LASTEXITCODE -ne 0) {
        throw 'Far EXCL CPU/source gate failed; GPU not started'
    }
    & $py -B (Join-Path $tmp 'tools\profile_h3_far_exclusion.py') `
        --expect-head $ExactHead --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "Far EXCL GPU diagnostic failed closed; inspect ONE JSON: $output"
    }
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No new ONE JSON was produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.status -ne 'PASS_FAR_EXCLUSION_GEOMETRY' -or
        $report.completed -ne $true -or
        $report.source_head -ne $ExactHead -or
        $report.banks.Count -ne 2 -or
        $report.all_mask_id_invariants_passed -ne $true -or
        $report.different_selection_masks_per_bank -ne $true -or
        $report.bank_b_complete_graph_eager_bit_exact -ne $true -or
        $report.no_new_kernel -ne $true -or
        $report.no_candidate_tile_kernel_timing -ne $true -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY') {
        throw 'Invalid diagnostic JSON or incomplete source/numeric gates'
    }
    foreach ($bank in $report.banks) {
        if ($bank.selection.verified_tile_centers -ne 128 -or
            $bank.selection.current_far_dot_blocks -ne 1920 -or
            $bank.selection.selected_near_leaves_per_center -ne 512) {
            throw 'Unexpected far mask shape or loop geometry'
        }
    }
    Write-Host "HTRC far selection: $($report.status)"
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
