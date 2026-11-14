param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [ValidateRange(1,16)]
    [int]$MaxKernelNodes = 12
)

# Bounded Nsight Compute study of the already-proven materialized path.
# One JSON, no H3 checkpoint, no video, no new CUDA kernel.
$ErrorActionPreference = "Stop"
$source = "D:\HTRC-graph-group-ab-20261113"
$branch = "experiment/flat-memory-to-speed-balanced-20261113"
$py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.BAT"
$out = "D:\ComfyOutput\H3\math_probe\flat_materialized_ncu.json"
$stamp = [guid]::NewGuid().ToString("N").Substring(0,10)
$clean = "D:\HTRC-ncu-materialized-$stamp"

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "HTRC source checkout missing"
}
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw "ComfyUI CUDA Python missing"
}
if (-not (Test-Path -LiteralPath $ncu -PathType Leaf)) {
    throw "Nsight Compute 2026.1.1 missing at inventory path"
}
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw "Close ComfyUI before profiler measurement"
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw "Close Blender before profiler measurement"
}
if (Test-Path -LiteralPath $clean) {
    throw "Refuse to reuse temporary checkout"
}

$envNames = @(
    "APL_HTRC_TILE32_STATS_BT",
    "APL_HTRC_DIRECTFLOW_TILES32_ADMISSION",
    "APL_HTRC_DIRECTFLOW_GROUP_POLICY",
    "APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP",
    "APL_HTRC_GRAPH_CAPTURE_SAFE_MU",
    "APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB",
    "APL_HTRC_DIRECTFLOW_MIN_FREE_MIB",
    "APL_HTRC_DIRECTFLOW_VB_MODE"
)
$previous = @{}
foreach ($name in $envNames) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$before = if (Test-Path -LiteralPath $out -PathType Leaf) {
    (Get-Item -LiteralPath $out).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $origin = (git -C $source remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $origin) { throw "Git origin not found" }
    git clone --no-checkout --no-hardlinks $source $clean
    if ($LASTEXITCODE -ne 0) { throw "Temporary clone failed" }
    git -C $clean remote set-url origin $origin
    if ($LASTEXITCODE -ne 0) { throw "Remote setup failed" }
    git -C $clean fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw "Git fetch failed" }
    git -C $clean switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw "Cannot detach exact HEAD" }
    if ((git -C $clean rev-parse HEAD).Trim() -ne $ExactHead) {
        throw "Exact HEAD mismatch"
    }

    [Environment]::SetEnvironmentVariable("APL_HTRC_TILE32_STATS_BT","16","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_TILES32_ADMISSION","0","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_GROUP_POLICY","legacy","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP",$null,"Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_GRAPH_CAPTURE_SAFE_MU","0","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB","768","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_MIN_FREE_MIB","256","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_VB_MODE","materialized","Process")

    & $py -B -m unittest discover -s (Join-Path $clean "tools") -p "test_h3_materialized_ncu_cpu.py" -v
    if ($LASTEXITCODE -ne 0) {
        throw "NCU CPU parser/source contracts failed; no GPU profile started"
    }
    & $py -B (Join-Path $clean "tools\profile_h3_flat_materialized_ncu.py") --ncu $ncu --count $MaxKernelNodes --output $out
    if ($LASTEXITCODE -ne 0) {
        throw "Nsight failed closed; inspect ONE JSON: $out"
    }
    if (-not (Test-Path -LiteralPath $out -PathType Leaf) -or
        (Get-Item -LiteralPath $out).LastWriteTimeUtc -le $before) {
        throw "Profiler did not write a fresh JSON"
    }
    $report = Get-Content -LiteralPath $out -Raw | ConvertFrom-Json
    if ($report.status -ne "PROFILE_COUNTERS_COLLECTED" -or
        $report.completed -ne $true -or
        $report.hardware_profile -ne $true -or
        $report.strict_child_parity_marker -ne $true -or
        $report.candidate -ne "materialized_ONLY" -or
        $report.tokens -ne 15275 -or $report.heads -ne 4 -or
        $report.window_tiles -ne 32 -or
        $report.production_promotion -ne "PROHIBITED_PROFILER_ONLY") {
        throw "Profiler JSON invalid or numeric gate incomplete"
    }
    Write-Host "HTRC materialized Nsight PROFILE_COUNTERS_COLLECTED"
    Write-Host "Profiled kernel nodes: $($report.profiled_launches)"
    Write-Host "One JSON: $out"
}
finally {
    foreach ($name in $envNames) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
    }
    if (Test-Path -LiteralPath $clean) {
        Remove-Item -LiteralPath $clean -Recurse -Force -ErrorAction SilentlyContinue
    }
}
