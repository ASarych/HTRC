param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [switch]$IClosedOtherGpuWorkloads
)

# Research-only fast-vs-balanced dense-quality pilot. One 56-head JSON;
# no production mode changes and no closed split/fused VB re-runs.
$ErrorActionPreference='Stop'
$repo='D:\HTRC-graph-group-ab-20261113'
$branch='experiment/flat-memory-to-speed-balanced-20261113'
$py='C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output='D:\ComfyOutput\H3\math_probe\flat_fast_balanced_56head_dense_quality.json'
$tmp='D:\HTRC-near-quality-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not $IClosedOtherGpuWorkloads) { throw 'Use -IClosedOtherGpuWorkloads only after closing other GPU tasks' }
if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'HTRC checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) { throw 'Close ComfyUI port 8188' }
if (Get-Process -Name blender,ollama,ollama_llama_server -ErrorAction SilentlyContinue) { throw 'Close Blender/Ollama' }
$gpu=@(& nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>$null)
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed' }
if (@($gpu | Where-Object {
    $_ -like '*\ComfyUI-Installs\*\standalone-env\python*.exe*' -or
    $_ -like '*\ComfyUI-Installs\*\ComfyUI\.venv\Scripts\python*.exe*'
}).Count -gt 0) { throw 'Background Comfy Python GPU client is still running' }
$old=if(Test-Path -LiteralPath $output -PathType Leaf){
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
}else{[datetime]::MinValue}

try {
    $remote=(git -C $repo remote get-url origin).Trim()
    if($LASTEXITCODE -ne 0 -or -not $remote){throw 'Missing git origin'}
    git clone --no-checkout --no-hardlinks $repo $tmp
    if($LASTEXITCODE -ne 0){throw 'Disposable clone failed'}
    git -C $tmp remote set-url origin $remote
    if($LASTEXITCODE -ne 0){throw 'Origin configuration failed'}
    git -C $tmp fetch origin $branch
    if($LASTEXITCODE -ne 0){throw 'Research branch fetch failed'}
    git -C $tmp switch --detach $ExactHead
    if($LASTEXITCODE -ne 0 -or (git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead){
        throw 'Exact source head not available'
    }
    $syntax=@(
        'tools\h3_flat_quality_ab.py',
        'tools\benchmark_h3_fast_balanced_dense_quality_pilot.py',
        'tools\test_h3_fast_balanced_dense_quality_pilot_cpu.py'
    ) | ForEach-Object { Join-Path $tmp $_ }
    & $py -B -m py_compile $syntax
    if($LASTEXITCODE -ne 0){throw 'Python syntax gate failed; CUDA not started'}
    foreach($suite in @(
        'test_h3_fast_balanced_dense_quality_pilot_cpu.py',
        'test_h3_flat_scale_protocol.py',
        'test_h3_flat_directflow_live_contract.py',
        'test_h3_graph_group_plan.py'
    )){
        & $py -B -m unittest discover -s (Join-Path $tmp 'tools') -p $suite -v
        if($LASTEXITCODE -ne 0){throw "CPU/source suite $suite FAILED; CUDA not started"}
    }

    $benchmark=Join-Path $tmp 'tools\benchmark_h3_fast_balanced_dense_quality_pilot.py'
    & $py -B $benchmark --expect-head $ExactHead --output $output
    $code=$LASTEXITCODE
    if(-not (Test-Path -LiteralPath $output -PathType Leaf) -or
       (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $old){
        throw 'Fresh ONE JSON not produced'
    }
    $r=Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if($r.schema -ne 'htrc-56head-fast-balanced-dense-quality-pilot-v1' -or
       $r.source_head -ne $ExactHead -or
       $r.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
       $r.speed_admission -ne 'UNRUN_INDEPENDENT_CAPTURE_ABBA' -or
       $r.h3_model_or_video -ne 'UNRUN'){
        throw 'Wrong exact-source pilot report schema / safety contract'
    }
    if($r.status -eq 'FAIL_CLOSED'){
        Write-Host $r.error
        Write-Host $r.traceback_tail
        throw 'Pilot FAIL_CLOSED; inspect ONE JSON'
    }
    if($code -ne 0 -or $r.status -ne 'QUALITY_MEASURED_SPEED_PRELIMINARY_ONLY' -or
       $r.completed -ne $true -or $r.comparison.bank_quality.Count -ne 2 -or
       $r.arms.balanced.results.materialized.status -ne 'QUALITY_MEASURED' -or
       $r.arms.fast.results.materialized.status -ne 'QUALITY_MEASURED'){
        throw 'Incomplete two-mode/two-bank quality pilot'
    }
    foreach($mode in @('balanced','fast')){
        $a=$r.arms.$mode
        if($a.results.materialized.quality.Count -ne 2 -or
           $a.results.materialized.timing.samples.Count -ne 7 -or
           $a.reference.queries_sampled -ne 64 -or
           $a.results.materialized.timing.gpu_event_ms_median -le 0){
            throw "Missing quality/timing evidence: $mode"
        }
    }
    Write-Host 'Dense FP32 sampled-query quality: two-mode pilot complete'
    Write-Host "Balanced error %: $($r.comparison.bank_quality.balanced_relative_l2_percent)"
    Write-Host "Fast error %: $($r.comparison.bank_quality.fast_relative_l2_percent)"
    Write-Host 'Speed comparison is PRELIMINARY, not independent A/B/B/A'
    Write-Host "ONE JSON: $output"
}
finally{
    if(Test-Path -LiteralPath $tmp -PathType Container){
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
