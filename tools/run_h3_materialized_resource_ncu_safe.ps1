param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [ValidateRange(3,16)]
    [int]$MaxKernelNodes = 12
)

# Independent, read-only Nsight resource/time triage of materialized HTRC.
# Pins the same QKV/Graph geometry; one separate JSON, never H3 or video.
$ErrorActionPreference = 'Stop'
$source = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.BAT'
$out = 'D:\ComfyOutput\H3\math_probe\flat_materialized_resource_ncu.json'
$stamp = [guid]::NewGuid().ToString('N').Substring(0,10)
$clean = "D:\HTRC-resource-ncu-$stamp"

if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw 'HTRC checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'CUDA Python missing' }
if (-not (Test-Path -LiteralPath $ncu -PathType Leaf)) { throw 'Nsight Compute missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI first'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender first'
}
if (Test-Path -LiteralPath $clean) { throw 'Refuse to reuse a stale checkout' }

$envNames = @(
    'APL_HTRC_TILE32_STATS_BT',
    'APL_HTRC_DIRECTFLOW_TILES32_ADMISSION',
    'APL_HTRC_DIRECTFLOW_GROUP_POLICY',
    'APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP',
    'APL_HTRC_GRAPH_CAPTURE_SAFE_MU',
    'APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB',
    'APL_HTRC_DIRECTFLOW_MIN_FREE_MIB',
    'APL_HTRC_DIRECTFLOW_VB_MODE'
)
$previous = @{}
foreach ($key in $envNames) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
$before = if (Test-Path -LiteralPath $out -PathType Leaf) {
    (Get-Item -LiteralPath $out).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $origin = (git -C $source remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $origin) { throw 'Git origin unavailable' }
    git clone --no-checkout --no-hardlinks $source $clean
    if ($LASTEXITCODE -ne 0) { throw 'Temporary clone failed' }
    git -C $clean remote set-url origin $origin
    if ($LASTEXITCODE -ne 0) { throw 'Cannot set origin' }
    git -C $clean fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Git fetch failed' }
    git -C $clean switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Cannot checkout exact HEAD' }
    if ((git -C $clean rev-parse HEAD).Trim() -ne $ExactHead) {
        throw 'Wrong Git HEAD; fail closed'
    }

    [Environment]::SetEnvironmentVariable('APL_HTRC_TILE32_STATS_BT','16','Process')
    [Environment]::SetEnvironmentVariable('APL_HTRC_DIRECTFLOW_TILES32_ADMISSION','0','Process')
    [Environment]::SetEnvironmentVariable('APL_HTRC_DIRECTFLOW_GROUP_POLICY','legacy','Process')
    [Environment]::SetEnvironmentVariable('APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP',$null,'Process')
    [Environment]::SetEnvironmentVariable('APL_HTRC_GRAPH_CAPTURE_SAFE_MU','0','Process')
    [Environment]::SetEnvironmentVariable('APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB','768','Process')
    [Environment]::SetEnvironmentVariable('APL_HTRC_DIRECTFLOW_MIN_FREE_MIB','256','Process')
    [Environment]::SetEnvironmentVariable('APL_HTRC_DIRECTFLOW_VB_MODE','materialized','Process')

    & $py -B -m unittest discover -s (Join-Path $clean 'tools') `
        -p 'test_h3_materialized_resource_ncu_cpu.py' -v
    if ($LASTEXITCODE -ne 0) {
        throw 'Resource profiler CPU tests failed; GPU not run'
    }

    & $py -B (Join-Path $clean 'tools\profile_h3_flat_resource_ncu.py') `
        --ncu $ncu --expected-head $ExactHead --count $MaxKernelNodes --output $out
    if ($LASTEXITCODE -ne 0) { throw "Nsight resource triage failed; inspect ONE JSON: $out" }

    if (-not (Test-Path -LiteralPath $out -PathType Leaf) -or
        (Get-Item -LiteralPath $out).LastWriteTimeUtc -le $before) {
        throw 'No fresh resource JSON'
    }
    $report = Get-Content -LiteralPath $out -Raw | ConvertFrom-Json
    if ($report.status -ne 'PROFILE_RESOURCE_COUNTERS_COLLECTED' -or
        $report.completed -ne $true -or
        $report.hardware_profile -ne $true -or
        $report.strict_child_parity_marker -ne $true -or
        $report.source_head -ne $ExactHead -or
        $report.candidate -ne 'materialized_ONLY' -or
        $report.has_stats_and_tile -ne $true -or
        $report.production_promotion -ne 'PROHIBITED_RESOURCE_TRIAGE_ONLY') {
        throw 'Resource profile JSON gate failed'
    }
    Write-Host 'HTRC resource counters verified (NCU replay, not ABBA timing)'
    Write-Host "JSON: $out"
}
finally {
    foreach ($key in $envNames) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
    if (Test-Path -LiteralPath $clean) {
        Remove-Item -LiteralPath $clean -Recurse -Force -ErrorAction SilentlyContinue
    }
}
