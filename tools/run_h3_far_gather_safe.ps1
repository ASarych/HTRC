param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [ValidateRange(7,12)]
    [int]$Rounds = 7
)

# Research ONLY. Never edit installed ComfyUI, production or user's original clone.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_far_gather_cost.json'
$tmp = 'D:\HTRC-far-gather-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'HTRC checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'ComfyUI GPU Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before experimental run'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender before experimental run'
}
if (Test-Path -LiteralPath $tmp) { throw 'Refuse to reuse temporary checkout' }
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Git origin missing' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Origin restore failed' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research branch fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Exact research HEAD unavailable' }
    if ((git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) { throw 'Exact HEAD mismatch' }

    & $py -B -m unittest discover -s (Join-Path $tmp 'tools') `
        -p 'test_h3_far_gather_cpu.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Far gather CPU/source gate failed; GPU not started' }
    & $py -B (Join-Path $tmp 'tools\benchmark_h3_far_gather_cost.py') `
        --expect-head $ExactHead --output $output --rounds $Rounds
    if ($LASTEXITCODE -ne 0) { throw "GPU gather failed closed; inspect ONE JSON $output" }
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'No new single JSON produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.status -ne 'PASS_GATHER_COST_ONLY' -or
        $report.completed -ne $true -or
        $report.source_head -ne $ExactHead -or
        $report.banks.Count -ne 2 -or
        $report.capture_blocks.Count -ne 4 -or
        $report.first_complete_graph_eager_bit_exact -ne $true -or
        $report.bank_b_complete_graph_eager_bit_exact -ne $true -or
        $report.no_packed_attention_kernel -ne $true -or
        $report.no_far_softmax_parity_claim -ne $true -or
        $report.timings.this_is_not_equal_work -ne $true -or
        $report.memory.gpu_output_suffix_undefined -ne $true -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY') {
        throw 'Missing strict read-only gather gates in ONE JSON'
    }
    foreach ($bank in $report.banks) {
        if ($bank.pack_validation.valid_prefixes_bit_exact -ne $true -or
            $bank.gather_validation.control_dense.raw_values_bit_exact -ne $true -or
            $bank.gather_validation.candidate_packed.raw_values_bit_exact -ne $true -or
            $bank.geometry.packed_valid_source_leaves -ne 56576) {
            throw 'Incomplete bytewise gather gate or unexpected far geometry'
        }
    }
    Write-Host "HTRC far gather: $($report.status)"
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
