param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [switch]$IClosedOtherGpuWorkloads,
    [switch]$GraphStageTrace
)

# Research-only full 56-head Graph and separate eager stage event attribution.
# No changes to installed ComfyUI, Torch, model, kernels, GPU clocks or drivers.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = if ($GraphStageTrace) {
    'D:\ComfyOutput\H3\math_probe\flat_56head_graph_stage_trace.json'
} else {
    'D:\ComfyOutput\H3\math_probe\flat_56head_stage_attribution.json'
}
$tmp = 'D:\HTRC-56-stage-' + ([guid]::NewGuid().ToString('N').Substring(0,10))
if (-not $IClosedOtherGpuWorkloads) { throw 'Confirm GPU workload closure with -IClosedOtherGpuWorkloads' }
if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'HTRC source checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'RTX CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI server before diagnostic'
}
if (Get-Process -Name blender,ollama,ollama_llama_server -ErrorAction SilentlyContinue) {
    throw 'Close Blender/Ollama before diagnostic'
}
$clientLines = @(& nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>$null)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect nvidia-smi clients before benchmark' }
$comfyPythonClients = @($clientLines | Where-Object {
    $_ -like '*\ComfyUI-Installs\*\standalone-env\python*.exe*' -or
    $_ -like '*\ComfyUI-Installs\*\ComfyUI\.venv\Scripts\python*.exe*'
})
if ($comfyPythonClients.Count -gt 0) {
    throw "Background Comfy GPU process already present: $($comfyPythonClients -join '; ')"
}
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }
try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Git origin missing' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Origin setup failed' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research branch fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0 -or (git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) {
        throw 'Exact source HEAD is unavailable'
    }

    & $py -B -m py_compile (Join-Path $tmp 'tools\benchmark_h3_56head_stage_attribution.py') (Join-Path $tmp 'tools\test_h3_56head_stage_attribution_cpu.py')
    if ($LASTEXITCODE -ne 0) { throw 'New Python files failed syntax preflight; CUDA not started' }
    foreach ($suite in @('test_h3_56head_stage_attribution_cpu.py',
                         'test_h3_flat_directflow_live_contract.py',
                         'test_h3_graph_group_plan.py',
                         'test_htrc_graph_stage_trace_cpu.py')) {
        & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p $suite -v
        if ($LASTEXITCODE -ne 0) { throw "CPU/source suite $suite failed; CUDA not started" }
    }
    # PowerShell unwraps the one-item array returned by an if expression.
    # A scalar string splatted with @name becomes separate CLI characters.
    # Use two explicit invocations so the opt-in flag is ONE argument.
    $benchmark = Join-Path $tmp 'tools\benchmark_h3_56head_stage_attribution.py'
    if ($GraphStageTrace) {
        & $py -B $benchmark --expect-head $ExactHead --output $output --quiet-confirmed --rounds 7 --warmups 5 --graph-stage-trace
    } else {
        & $py -B $benchmark --expect-head $ExactHead --output $output --quiet-confirmed --rounds 7 --warmups 5
    }
    $code = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'Fresh ONE JSON was not produced'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($code -ne 0) {
        Write-Host "Diagnostic failed status=$($report.status): $($report.error)"
        Write-Host $report.traceback_tail
        throw 'FAIL_CLOSED: see real Python error above'
    }
    $accepted = if ($GraphStageTrace) {
        'PASS_GRAPH_BOUNDARY_TRACE_DIAGNOSTIC_ONLY'
    } else { 'PASS_ATTRIBUTION_ONLY' }
    if ($report.schema -ne 'htrc-56head-graph-eager-stage-attribution-v1' -or
        $report.status -ne $accepted -or
        $report.completed -ne $true -or
        $report.source_head -ne $ExactHead -or
        $report.numeric_gate -ne 'PASS_TWO_BANK_FULL_OUTPUT_GRAPH_EAGER_BIT_EXACT' -or
        $report.strict_bank_checks.Count -ne 2 -or
        $report.warmed_graph.samples.Count -ne 7 -or
        $report.independent_full_output_bit_exact -ne $true -or
        $report.eager_stage_attribution.scope -ne 'EAGER_ONLY__NOT_GRAPH_KERNEL_TIMING' -or
        $report.h3_model_or_video -ne 'UNRUN' -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY') {
        throw 'Incomplete exact-source/numeric/stage diagnostic report'
    }
    if ($report.graph_stage_trace_requested -ne [bool]$GraphStageTrace) {
        throw 'Graph trace opt-in flag mismatch'
    }
    if ($GraphStageTrace) {
        $trace = $report.graph_stage_trace
        if ($trace.status -ne 'PASS_GRAPH_BOUNDARY_TRACE_BIT_EXACT' -or
            $trace.full_output_bit_exact -ne $true -or
            $trace.graph_replay_count -ne 105 -or
            $trace.groups -ne 7 -or
            $trace.raw_intervals.Count -ne 322 -or
            $trace.not_graph_internal_kernel_profile -ne $true) {
            throw 'Incomplete diagnostic-only Graph boundary trace'
        }
        foreach ($label in @('kv_refresh_quant','q_stage','graph_replay','output_write')) {
            if ($trace.stage_gpu_stream_intervals_ms.$label -le 0 -or
                $trace.stage_cpu_enqueue_intervals_ms.$label -le 0) {
                throw "Graph boundary stage $label missing"
            }
        }
        Write-Host "Graph stage trace event distortion >20%: $($trace.measurement_distorted_over_20pct)"
    }
    foreach ($bank in $report.strict_bank_checks) {
        if ($bank.full_output_exact -ne $true -or $bank.head_ranges.Count -ne 7) {
            throw 'Not all 56 original heads were verified'
        }
    }
    foreach ($stage in @('kv_workspace_setup','query_stats_selection','tile','merge_tail','output_copy')) {
        if ($null -eq $report.eager_stage_attribution.stage_gpu_stream_intervals_ms.$stage -or
            $null -eq $report.eager_stage_attribution.stage_cpu_enqueue_intervals_ms.$stage) {
            throw "Missing eager observer stage $stage"
        }
    }
    Write-Host "Graph full 56-head median: $($report.warmed_graph.gpu_event_median_ms) ms"
    Write-Host "Graph event spread: $($report.warmed_graph.gpu_event_block_spread_pct)%"
    Write-Host 'Eager stages and instrumented Graph boundaries are NOT Graph internal kernel timings'
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp -PathType Container) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
exit 0
