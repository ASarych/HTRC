param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [ValidateSet("split_stream","fused_vb")]
    [string]$Candidate = "split_stream",
    [ValidateRange(2,12)]
    [int]$Rounds = 5,
    [switch]$DiagnoseNumeric
)

# One isolated exact-HEAD CUDA A/B. No installed ComfyNodes edits,
# H3 weights, VAE, video, or silent production-mode promotion.
$ErrorActionPreference = "Stop"
$source = "D:\HTRC-graph-group-ab-20261113"
$branch = "experiment/flat-memory-to-speed-balanced-20261113"
$py = "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe"
$tag = [guid]::NewGuid().ToString("N").Substring(0,10)
$clean = "D:\HTRC-vb-probe-$tag"
if ($DiagnoseNumeric -and $Candidate -ne "fused_vb") {
    throw "Numeric diagnostic is restricted to fused_vb"
}

if ($Candidate -eq "split_stream") {
    $bench = "benchmark_h3_flat_streamed_vb_ab.py"
    $test = "test_h3_flat_streamed_vb_contract.py"
    $report = "D:\ComfyOutput\H3\math_probe\flat_streamed_vb_ab.json"
    $numericKeyA = "bank_0_materialized_vs_stream"
    $numericKeyB = "bank_1_materialized_vs_stream"
} else {
    $bench = "benchmark_h3_flat_fused_vb_ab.py"
    $test = "test_h3_flat_fused_vb_contract.py"
    $report = "D:\ComfyOutput\H3\math_probe\flat_fused_vb_ab.json"
    $numericKeyA = "bank_0_materialized_vs_fused_vb"
    $numericKeyB = "bank_1_materialized_vs_fused_vb"
}

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "HTRC source checkout missing: $source"
}
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw "ComfyUI CUDA Python missing: $py"
}
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw "Close ComfyUI before isolated GPU measurement"
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw "Close Blender before isolated GPU measurement"
}
if (Test-Path -LiteralPath $clean) {
    throw "Refuse to reuse existing temporary checkout: $clean"
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
$before = if (Test-Path -LiteralPath $report -PathType Leaf) {
    (Get-Item -LiteralPath $report).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $origin = (git -C $source remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $origin) {
        throw "Cannot identify HTRC Git remote"
    }
    git clone --no-checkout --no-hardlinks $source $clean
    if ($LASTEXITCODE -ne 0) { throw "Temporary clone failed" }
    git -C $clean remote set-url origin $origin
    if ($LASTEXITCODE -ne 0) { throw "Temporary remote setup failed" }
    git -C $clean fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw "HTRC branch fetch failed" }
    git -C $clean switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw "Exact HEAD checkout failed" }
    if ((git -C $clean rev-parse HEAD).Trim() -ne $ExactHead) {
        throw "Detached checkout HEAD mismatch"
    }

    # This four-head VB gate is NOT the prior special eight-head
    # STATS_BT Tile32 experiment. Its dedicated admission would
    # reject four heads, so disable only that research preflight.
    # Standard Graph allocator/free-memory guards still apply.
    [Environment]::SetEnvironmentVariable("APL_HTRC_TILE32_STATS_BT","16","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_TILES32_ADMISSION","0","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_GROUP_POLICY","legacy","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_RESEARCH_FORCE_GROUP",$null,"Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_GRAPH_CAPTURE_SAFE_MU","0","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_MAX_EXTRA_MIB","768","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_MIN_FREE_MIB","256","Process")
    [Environment]::SetEnvironmentVariable("APL_HTRC_DIRECTFLOW_VB_MODE",$null,"Process")

    & $py -B -m unittest discover -s (Join-Path $clean "tools") -p $test -v
    if ($LASTEXITCODE -ne 0) {
        throw "CPU contracts failed; GPU benchmark NOT started"
    }
    $gpuArgs = @(
        "--tokens", "15275", "--heads", "4", "--mode", "balanced",
        "--tiles", "32", "--rounds", "$Rounds",
        "--budget-mib", "1024", "--reserve-mib", "256",
        "--output", $report
    )
    if ($DiagnoseNumeric) {
        $gpuArgs += "--diagnose-numeric"
    }
    & $py -B (Join-Path $clean ("tools\" + $bench)) @gpuArgs
    if ($LASTEXITCODE -ne 0) {
        throw "GPU candidate failed closed; inspect ONE JSON at $report"
    }
    if (-not (Test-Path -LiteralPath $report -PathType Leaf) -or
        (Get-Item -LiteralPath $report).LastWriteTimeUtc -le $before) {
        throw "No fresh CUDA report written"
    }
    $result = Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
    if ($DiagnoseNumeric) {
        if ($result.completed -ne $true -or
            $result.status -notin @("DIAG_NUMERIC_MISMATCH","DIAG_MATCH_NOT_ACCEPTED") -or
            $result.comparisons.materialized_vs_eager.bit_exact -ne $true -or
            $result.compare_only_for_diagnosis -ne $true -or
            $result.strict_first_call_gate_changed -ne $false -or
            $result.performance_measured -ne $false -or
            $result.promotion_to_real_h3 -ne "PROHIBITED_DIAGNOSTIC_ONLY" -or
            $result.mode -ne "balanced" -or $result.tiles -ne 32 -or
            $result.tokens -ne 15275 -or $result.heads -ne 4) {
            throw "Diagnostic invalid or materialized control failed"
        }
        Write-Host "HTRC fused_vb DIAGNOSTIC ONLY; no numerical PASS"
        Write-Host "Fused-vs-eager maximum absolute error: $($result.comparisons.fused_vb_vs_eager.max_abs)"
        Write-Host "Fused-vs-eager mismatched elements: $($result.comparisons.fused_vb_vs_eager.mismatch_count)"
        Write-Host "One JSON: $report"
    } else {
        if ($result.completed -ne $true -or $result.status -ne "PASS_NUMERIC" -or
            $result.numeric.$numericKeyA -ne $true -or
            $result.numeric.$numericKeyB -ne $true -or
            $result.eager_graph_first_call_gates.materialized.first_call_bit_exact -ne $true -or
            $result.eager_graph_first_call_gates.$Candidate.first_call_bit_exact -ne $true -or
            $result.eager_graph_first_call_gates.materialized.bank_b_eager_bit_exact -ne $true -or
            $result.eager_graph_first_call_gates.$Candidate.bank_b_eager_bit_exact -ne $true -or
            $result.mode -ne "balanced" -or $result.window_tiles -ne 32 -or
            $result.tokens -ne 15275 -or $result.heads -ne 4) {
            throw "CUDA numeric gate or requested fixed geometry failed"
        }
        Write-Host "HTRC $Candidate full attention A/B: PASS_NUMERIC"
        Write-Host "Materialized Graph GPU ms: $($result.warm_graph_timings.materialized.gpu_event_ms_per_call)"
        Write-Host "Candidate Graph GPU ms: $($result.warm_graph_timings.$Candidate.gpu_event_ms_per_call)"
        Write-Host "One JSON: $report"
    }
}
finally {
    foreach ($name in $envNames) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
    }
    if (Test-Path -LiteralPath $clean) {
        Remove-Item -LiteralPath $clean -Recurse -Force -ErrorAction SilentlyContinue
    }
}
