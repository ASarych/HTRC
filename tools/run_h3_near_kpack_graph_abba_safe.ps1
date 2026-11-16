param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [switch]$IClosedOtherGpuWorkloads
)

# One research-only full 56-head Graph A/B/B/A; no production modifications.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_56head_near_kpack_graph_abba.json'
$tmp = 'D:\HTRC-near56-abba-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not $IClosedOtherGpuWorkloads) {
    throw 'Confirm other GPU workloads closed with -IClosedOtherGpuWorkloads'
}
if (-not (Test-Path -LiteralPath $repo -PathType Container)) {
    throw 'HTRC source checkout missing'
}
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw 'RTX CUDA Python missing'
}
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before full Graph research'
}
if (Get-Process -Name blender,ollama,ollama_llama_server -ErrorAction SilentlyContinue) {
    throw 'Close Blender/Ollama before full Graph research'
}
$clientLines = @(& nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>$null)
if ($LASTEXITCODE -ne 0) {
    throw 'nvidia-smi GPU client preflight failed'
}
$comfyPythonClients = @($clientLines | Where-Object {
    $_ -like '*\ComfyUI-Installs\*\standalone-env\python*.exe*' -or
    $_ -like '*\ComfyUI-Installs\*\ComfyUI\.venv\Scripts\python*.exe*'
})
if ($comfyPythonClients.Count -gt 0) {
    throw "Background Comfy GPU client present: $($comfyPythonClients -join '; ')"
}
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Missing Git origin' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Cannot restore remote' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research HEAD fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0 -or (git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) {
        throw 'Exact HEAD checkout failed'
    }

    $pyFiles = @(
        'tools\htrc_near_kpack_triton.py',
        'tools\htrc_near_kpack_graph_adapter.py',
        'tools\benchmark_h3_near_kpack_numeric.py',
        'tools\benchmark_h3_near_kpack_graph_abba.py',
        'tools\test_h3_near_kpack_numeric_cpu.py',
        'tools\test_h3_near_kpack_graph_abba_cpu.py'
    ) | ForEach-Object { Join-Path $tmp $_ }
    & $py -B -m py_compile $pyFiles
    if ($LASTEXITCODE -ne 0) {
        throw 'Source syntax gate FAILED; CUDA not started'
    }
    foreach ($suite in @(
        'test_h3_near_kpack_numeric_cpu.py',
        'test_h3_near_kpack_graph_abba_cpu.py',
        'test_h3_56head_stage_attribution_cpu.py',
        'test_h3_flat_directflow_live_contract.py',
        'test_h3_graph_group_plan.py'
    )) {
        & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p $suite -v
        if ($LASTEXITCODE -ne 0) {
            throw "CPU/source suite $suite FAILED; CUDA not started"
        }
    }

    $benchmark = Join-Path $tmp 'tools\benchmark_h3_near_kpack_graph_abba.py'
    & $py -B $benchmark --expect-head $ExactHead --output $output --quiet-confirmed --rounds 7 --warmups 5
    $exitCode = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'Fresh ONE JSON was not produced'
    }
    $r = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($r.schema -ne 'htrc-56head-near-kpack-full-graph-abba-v1' -or
        $r.source_head -ne $ExactHead -or
        $r.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
        $r.h3_model_or_video -ne 'UNRUN' -or
        $r.independent_model_resident_vram -ne 'UNRUN' -or
        $r.shared_packed_k8_and_scale_bytes_per_captured_session -ne 138412032) {
        throw 'Source/prod/scratch report contract FAILED; inspect ONE JSON'
    }
    if ($r.status -eq 'FAIL_CLOSED') {
        if ($exitCode -ne 2 -or $r.completed -ne $false) {
            throw 'FAIL_CLOSED JSON has inconsistent process status'
        }
        Write-Host "Strict numerical/capture blocker: $($r.error)"
        Write-Host $r.traceback_tail
        throw 'HTRC FAIL_CLOSED: candidate timing or numerical gate not admitted'
    }
    $accepted = @('PASS_NUMERIC_PROMISING_FULL_GRAPH',
                  'PASS_NUMERIC_NO_GO_FULL_GRAPH',
                  'PASS_NUMERIC_INCONCLUSIVE_CAPTURE')
    if ($r.status -notin $accepted -or
        $exitCode -ne 0 -or
        $r.completed -ne $true -or
        $r.numeric_gate -ne 'PASS_TWO_BANK_FULL_56HEAD_AND_RAW_PM_PL_PACC_BIT_EXACT' -or
        $r.candidate_speed_measurement -ne 'MEASURED_FULL_56HEAD_GRAPH_INCLUDES_PACK' -or
        $r.two_distinct_bf16_qkv_banks -ne $true -or
        $r.preflight_candidate_raw_partials.Count -ne 2 -or
        $r.capture_blocks.Count -ne 4 -or
        $r.abba_decision.capture_order.Count -ne 4) {
        throw 'Incomplete full 56-head numeric or A/B/B/A evidence'
    }
    $expectedArms = @('control','packed','packed','control')
    for ($i=0; $i -lt 4; $i++) {
        $b = $r.capture_blocks[$i]
        if ($b.arm -ne $expectedArms[$i] -or
            $b.samples.Count -ne 7 -or
            $b.first_call_all_56_heads_strict_eager_bit_exact -ne $true -or
            $b.separate_bank_B_full_output_bit_exact -ne $true -or
            $b.pre_timing_full_output_bit_exact -ne $true -or
            $b.post_timing_full_output_bit_exact -ne $true -or
            $b.runtime_graph_capture_hard_768mib_gate -ne 'PASSED' -or
            $b.gpu_event_median_ms -le 0) {
            throw "Incomplete A/B/B/A block $i"
        }
        if ($b.arm -eq 'packed' -and
            ($b.raw_candidate_numeric.Count -ne 2 -or
             $b.pre_timing_raw_partials -ne 'PASS_TWO_BANK_PM_PL_PACC_AND_MERGE' -or
             $b.scratch.allocated_once_for_both_geometries -ne $true)) {
            throw "Candidate block $i lacks strict two-bank raw partial gate"
        }
    }
    foreach ($row in $r.preflight_candidate_raw_partials) {
        if ($row.all_partial_bits_exact -ne $true -or
            $row.merge_bit_exact -ne $true) {
            throw 'Candidate preflight PM/PL/PACC or Merge not bit-exact'
        }
        foreach ($field in @('pm','pl','acc')) {
            if ($row.pm_pl_pacc.$field.bit_exact -ne $true) {
                throw "Raw-field parity missing: $field"
            }
        }
    }
    if ($r.abba_decision.decision -eq 'PROMISING_RESEARCH_ONLY' -and
        $r.status -ne 'PASS_NUMERIC_PROMISING_FULL_GRAPH') {
        throw 'Contradictory speed decision'
    }
    Write-Host "Numeric: $($r.numeric_gate)"
    Write-Host "Controls: $($r.abba_decision.control_capture_medians_ms)"
    Write-Host "Packed: $($r.abba_decision.packed_capture_medians_ms)"
    Write-Host "Full Graph median saving: $($r.abba_decision.median_full_graph_reduction_pct)%"
    Write-Host "Worst pair saving: $($r.abba_decision.worst_pair_full_graph_reduction_pct)%"
    Write-Host "Decision: $($r.abba_decision.decision)"
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp -PathType Container) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
