param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [switch]$IClosedOtherGpuWorkloads
)

# ONE strict 4-head Tile32 numerical-only PASS1/PASS2 experiment.
# NO candidate speed, no production path or ComfyUI edits.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_tile_far_near_pass12_numeric.json'
$tmp = 'D:\HTRC-pass12-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not $IClosedOtherGpuWorkloads) {
    throw 'Confirm closed GPU workloads with -IClosedOtherGpuWorkloads'
}
if (-not (Test-Path -LiteralPath $repo -PathType Container)) {
    throw 'HTRC source checkout missing'
}
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw 'RTX CUDA Python missing'
}
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI server before numerical research'
}
if (Get-Process -Name blender,ollama,ollama_llama_server -ErrorAction SilentlyContinue) {
    throw 'Close Blender/Ollama before numerical research'
}
$gpuClients = @(& nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>$null)
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi preflight failed' }
if (@($gpuClients | Where-Object {
    $_ -like '*\ComfyUI-Installs\*\standalone-env\python*.exe*' -or
    $_ -like '*\ComfyUI-Installs\*\ComfyUI\.venv\Scripts\python*.exe*'
}).Count -gt 0) {
    throw 'Background Comfy GPU Python still active'
}
$prior = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $origin = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $origin) { throw 'Git origin missing' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable clone failed' }
    git -C $tmp remote set-url origin $origin
    if ($LASTEXITCODE -ne 0) { throw 'Origin reset failed' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research SHA fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0 -or (git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) {
        throw 'Exact detached HEAD unavailable'
    }

    $syntax = @(
        'tools\htrc_two_pass_tile_adapter.py',
        'tools\benchmark_h3_two_pass_numeric.py',
        'tools\test_h3_two_pass_numeric_cpu.py'
    ) | ForEach-Object { Join-Path $tmp $_ }
    & $py -B -m py_compile $syntax
    if ($LASTEXITCODE -ne 0) {
        throw 'Python syntax preflight FAILED; CUDA not started'
    }
    foreach ($suite in @(
        'test_h3_two_pass_numeric_cpu.py',
        'test_h3_near_kpack_numeric_cpu.py',
        'test_h3_flat_directflow_live_contract.py',
        'test_h3_graph_group_plan.py'
    )) {
        & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p $suite -v
        if ($LASTEXITCODE -ne 0) {
            throw "CPU/source suite $suite FAILED; CUDA not started"
        }
    }

    $benchmark = Join-Path $tmp 'tools\benchmark_h3_two_pass_numeric.py'
    & $py -B $benchmark --expect-head $ExactHead --output $output
    $code = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $prior) {
        throw 'Fresh ONE JSON was not produced'
    }
    $r = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($r.schema -ne 'htrc-tile-far-near-pass12-numeric-v1' -or
        $r.source_head -ne $ExactHead -or
        $r.candidate_speed_measurement -ne 'UNRUN_NUMERIC_ONLY' -or
        $r.capture_blocks.Count -ne 0 -or
        $r.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
        $r.h3_model_or_video -ne 'UNRUN' -or
        $r.candidate_complete_56head_graph -ne 'UNRUN') {
        throw 'Exact numeric-only research report contract FAILED'
    }
    if ($r.status -eq 'PASS_TWO_PASS_NUMERIC_ONLY') {
        if ($code -ne 0 -or $r.completed -ne $true -or
            $r.banks.Count -ne 2 -or
            $r.two_distinct_bf16_qkv_banks -ne $true -or
            $r.first_complete_graph_eager_bit_exact -ne $true -or
            $r.second_complete_graph_eager_bit_exact -ne $true -or
            $r.candidate_both_banks_bit_exact -ne $true -or
            $r.parity_admission -ne 'PASS_TWO_BANK_PM_PL_PACC_MERGE_BIT_EXACT') {
            throw 'Incomplete two-bank PASS1/PASS2 numeric PASS'
        }
        foreach ($bank in $r.banks) {
            if ($bank.candidate_full_window_bit_exact -ne $true -or
                $bank.candidate_merged_output.bit_exact -ne $true -or
                $bank.original_window_graph_bit_exact.bit_exact -ne $true) {
                throw 'Missing original/updated Merge parity'
            }
            foreach ($field in @('pm','pl','acc')) {
                if ($bank.candidate_partial_pm_pl_pacc.$field.bit_exact -ne $true) {
                    throw "Missing raw-bit $field candidate parity"
                }
            }
        }
        Write-Host 'Two-bank original PASS1/PASS2: NUMERIC PASS; speed UNRUN'
    } elseif ($r.status -eq 'NO_GO_TWO_PASS_NUMERIC') {
        if ($code -ne 0 -or $r.completed -ne $true -or
            $r.banks.Count -ne 2 -or $r.candidate_both_banks_bit_exact -ne $false) {
            throw 'Incomplete two-bank numeric NO_GO report'
        }
        Write-Host 'Two-pass Tile: NUMERIC NO_GO; do NOT test speed'
    } elseif ($r.status -eq 'FAIL_CLOSED') {
        Write-Host "Diagnostic blocker: $($r.error)"
        Write-Host $r.traceback_tail
        throw 'HTRC FAIL_CLOSED: inspect ONE JSON; speed UNRUN'
    } else {
        throw 'Unknown two-pass report status'
    }
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp -PathType Container) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
