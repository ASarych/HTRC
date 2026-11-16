param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [switch]$IClosedOtherGpuWorkloads
)

# Research-only numeric gate for near K8+FP32 scale layout; speed UNRUN.
# Disposable detached checkout; never edits HTRC production or ComfyUI.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_near_kpack_numeric.json'
$tmp = 'D:\HTRC-near-kpack-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not $IClosedOtherGpuWorkloads) {
    throw 'Confirm closed GPU workloads with -IClosedOtherGpuWorkloads'
}
if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'HTRC checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'RTX CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI server before exact numerical research'
}
if (Get-Process -Name blender,ollama,ollama_llama_server -ErrorAction SilentlyContinue) {
    throw 'Close Blender/Ollama before exact numerical research'
}
$clientLines = @(& nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>$null)
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi preflight failed' }
$comfyPythonClients = @($clientLines | Where-Object {
    $_ -like '*\ComfyUI-Installs\*\standalone-env\python*.exe*' -or
    $_ -like '*\ComfyUI-Installs\*\ComfyUI\.venv\Scripts\python*.exe*'
})
if ($comfyPythonClients.Count -gt 0) { throw 'Background Comfy GPU Python still active' }
$old = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Missing Git origin' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Disposable clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Cannot set origin' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Research ref unavailable' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0 -or (git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) {
        throw 'Exact source HEAD is unavailable'
    }

    & $py -B -m py_compile (Join-Path $tmp 'tools\benchmark_h3_near_kpack_numeric.py') (Join-Path $tmp 'tools\htrc_near_kpack_triton.py') (Join-Path $tmp 'tools\test_h3_near_kpack_numeric_cpu.py')
    if ($LASTEXITCODE -ne 0) { throw 'Python syntax preflight FAILED; CUDA not started' }
    & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p 'test_h3_near_kpack_numeric_cpu.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Near-K8 CPU/source gate FAILED; CUDA not started' }

    $bench = Join-Path $tmp 'tools\benchmark_h3_near_kpack_numeric.py'
    & $py -B $bench --expect-head $ExactHead --output $output
    $code = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old) {
        throw 'Fresh ONE JSON was not produced'
    }

    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.schema -ne 'htrc-near-kpack-numeric-v1' -or
        $report.source_head -ne $ExactHead -or
        $report.candidate_speed_measurement -ne 'UNRUN_NUMERIC_GATE_ONLY' -or
        $report.speed_capture_blocks.Count -ne 0 -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
        $report.h3_model_or_video -ne 'UNRUN') {
        throw 'Invalid exact-source/no-speed research contract; inspect ONE JSON'
    }
    if ($report.status -eq 'PASS_NEAR_KPACK_NUMERIC_ONLY') {
        if ($code -ne 0 -or $report.completed -ne $true -or
            $report.parity_admission -ne 'PASS_TWO_BANK_WINDOW_PARTIALS_AND_MERGE' -or
            $report.banks.Count -ne 2 -or
            $report.full_window_candidate_all_banks_bit_exact -ne $true -or
            $report.first_complete_graph_eager_bit_exact -ne $true -or
            $report.second_complete_graph_eager_bit_exact -ne $true) {
            throw 'Incomplete two-bank candidate numeric PASS'
        }
        foreach ($bank in $report.banks) {
            if ($bank.candidate_full_window_bit_exact -ne $true -or
                $bank.baseline_graph_window_output_bit_exact.bit_exact -ne $true -or
                $bank.candidate_merged_output.bit_exact -ne $true) {
                throw 'Missing strict per-bank full Tile/Merge parity'
            }
            foreach ($field in @('pm','pl','acc')) {
                if ($bank.candidate_partial_pm_pl_pacc.$field.bit_exact -ne $true) {
                    throw "Missing raw-bit $field parity"
                }
            }
        }
        Write-Host 'Near-K8+scale packed full Tile/Merge: TWO-BANK NUMERIC PASS; speed UNRUN'
    } elseif ($report.status -eq 'FAIL_CLOSED' -and
              $report.parity_admission -eq 'NO_GO_NEAR_KPACK_BIT_EXACT') {
        if ($code -ne 2 -or $report.banks.Count -ne 2) {
            throw 'Incomplete numeric NO_GO evidence'
        }
        Write-Host 'Near-K8+scale candidate: NUMERIC NO_GO; speed UNRUN'
    } else {
        Write-Host "Diagnostic status: $($report.status) error=$($report.error)"
        Write-Host $report.traceback_tail
        throw 'Runtime preflight or CUDA error; inspect ONE JSON'
    }
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp -PathType Container) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
