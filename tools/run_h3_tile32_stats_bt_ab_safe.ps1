param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [ValidateRange(2,12)]
    [int]$Rounds = 3
)

# Only a temporary exact-HEAD HTRC checkout is executed. No ComfyUI
# replacement, H3 checkpoint, VAE, production branch, or inference.
$ErrorActionPreference = "Stop"
$source = "D:\HTRC-graph-group-ab-20261113"
$branch = "experiment/flat-memory-to-speed-balanced-20261113"
$py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
$report = "D:\ComfyOutput\H3\math_probe\flat_tile32_stats_bt_ab.json"
$tag = [guid]::NewGuid().ToString("N").Substring(0,10)
$clean = "D:\HTRC-stats-BT-ab-$tag"

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "HTRC source missing: $source"
}
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw "ComfyUI CUDA Python missing: $py"
}
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw "Close ComfyUI before measuring GPU kernel A/B"
}
if (Test-Path -LiteralPath $clean) {
    throw "Temporary HTRC checkout already exists: $clean"
}
$before = if (Test-Path -LiteralPath $report -PathType Leaf) {
    (Get-Item -LiteralPath $report).LastWriteTimeUtc
} else {
    [datetime]::MinValue
}

try {
    $origin = (git -C $source remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $origin) {
        throw "Cannot resolve HTRC Git remote"
    }
    git clone --no-checkout --no-hardlinks $source $clean
    if ($LASTEXITCODE -ne 0) { throw "Clean HTRC clone failed" }
    git -C $clean remote set-url origin $origin
    if ($LASTEXITCODE -ne 0) { throw "HTRC remote update failed" }
    git -C $clean fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw "HTRC exact branch fetch failed" }
    git -C $clean switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw "HTRC exact HEAD checkout failed" }
    if ((git -C $clean rev-parse HEAD).Trim() -ne $ExactHead) {
        throw "HTRC checkout is not the requested exact HEAD"
    }
    & $py -B (Join-Path $clean "tools\check_h3_tile32_stats_bt_cpu.py")
    if ($LASTEXITCODE -ne 0) {
        throw "CPU admission failed. Model/GPU work not started"
    }
    & $py -B (Join-Path $clean "tools\benchmark_h3_tile32_stats_bt_ab_gpu.py") `
        --tokens 40759 --heads 8 --rounds $Rounds --output $report
    if ($LASTEXITCODE -ne 0) {
        throw "Stats BT16-vs-BT32 CUDA A/B failed closed. Inspect the single JSON"
    }
    if (-not (Test-Path -LiteralPath $report -PathType Leaf) -or
        (Get-Item -LiteralPath $report).LastWriteTimeUtc -le $before) {
        throw "No fresh stats BT CUDA report written"
    }
    $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    if ($result.status -ne "PASS_NUMERIC" -or
        $result.completed -ne $true -or
        $result.outputs.bank_0.bit_exact -ne $true -or
        $result.outputs.bank_1.bit_exact -ne $true -or
        $result.model_checkpoint_loaded -ne $false -or
        $result.video_or_vae_executed -ne $false) {
        throw "BT32 research output parity or scope gate failed"
    }
    Write-Host "HTRC stats BT A/B: PASSED numerical parity"
    Write-Host "GPU ratio BT16/BT32: $($result.gpu_speed_ratio_bt16_over_bt32)"
    Write-Host "Report (one JSON): $report"
}
finally {
    if (Test-Path -LiteralPath $clean) {
        Remove-Item -LiteralPath $clean -Recurse -Force -ErrorAction SilentlyContinue
    }
}
