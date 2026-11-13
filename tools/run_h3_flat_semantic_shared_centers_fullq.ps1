# Single-command, read-only full-Q GPU A/B for MiniMax H3 Flat.
# Downloads pinned research script + pinned existing HTRC kernels.
# Does not modify ComfyUI, models, repositories or production backends.
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$SourceRef = "32443521ddf3cd474020cbeba8ef97bdc5b21186"
$BenchmarkRef = "3facc340959c29ec7eb8bea2d51497b24e85661e"
$CacheRoot = "D:\ComfyCache"
$WorkRoot = Join-Path $CacheRoot "htrc_semantic_shared_centers_ab"
$OutputRoot = "D:\ComfyOutput\H3\math_probe"
$PreferredCapture = Join-Path $OutputRoot "20260926-201504\attention_block_00_call_01.pt"
$Out = Join-Path $OutputRoot "flat_semantic_shared_centers_fullq_ab.json"

foreach ($Folder in @(
    $WorkRoot,
    $OutputRoot,
    (Join-Path $CacheRoot "cuda"),
    (Join-Path $CacheRoot "tmp"),
    (Join-Path $CacheRoot "torch_extensions"),
    (Join-Path $CacheRoot "torchinductor"),
    (Join-Path $CacheRoot "triton")
)) {
    New-Item -ItemType Directory -Path $Folder -Force | Out-Null
}
$env:CUDA_CACHE_PATH = Join-Path $CacheRoot "cuda"
$env:TEMP = Join-Path $CacheRoot "tmp"
$env:TMP = Join-Path $CacheRoot "tmp"
$env:TORCH_EXTENSIONS_DIR = Join-Path $CacheRoot "torch_extensions"
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $CacheRoot "torchinductor"
$env:TRITON_CACHE_DIR = Join-Path $CacheRoot "triton"

$CandidatePython = @(
    "C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe",
    (Join-Path $env:LOCALAPPDATA "Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe")
)
$Installs = Join-Path $env:LOCALAPPDATA "Comfy-Desktop\ComfyUI-Installs"
if (Test-Path $Installs -PathType Container) {
    $CandidatePython += @(Get-ChildItem -Path $Installs -Filter "python.exe" -File -Recurse -ErrorAction SilentlyContinue | Where-Object {
        $_.FullName -like "*\.venv\Scripts\python.exe"
    } | Select-Object -ExpandProperty FullName)
}
$Cmd = Get-Command python -ErrorAction SilentlyContinue
if ($Cmd -and $Cmd.Source) { $CandidatePython += $Cmd.Source }
$Python = $null
foreach ($Candidate in ($CandidatePython | Select-Object -Unique)) {
    if (-not $Candidate -or -not (Test-Path $Candidate -PathType Leaf)) { continue }
    & $Candidate -c "import torch, triton; assert torch.cuda.is_available()" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $Python = $Candidate
        break
    }
}
if (-not $Python) {
    throw "ComfyUI CUDA Python with torch and triton was not found. Check the .venv under Comfy-Desktop."
}

if (Test-Path $PreferredCapture -PathType Leaf) {
    $Capture = $PreferredCapture
} else {
    Write-Host "Finding saved full-Q attention capture under $OutputRoot ..." -ForegroundColor Cyan
    $Candidates = @(Get-ChildItem -Path $OutputRoot -Filter "attention_block_00_call_01.pt" -File -Recurse -ErrorAction SilentlyContinue)
    if ($Candidates.Count -eq 0) {
        throw "Full-Q capture not found under $OutputRoot (wanted attention_block_00_call_01.pt). No model run was started."
    }
    $Capture = ($Candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

$UrlRoot = "https://raw.githubusercontent.com/ASarych/HTRC"
$Files = @(
    @{ Name = "htrc_flat.py"; Url = "$UrlRoot/$SourceRef/h3/htrc_flat.py" },
    @{ Name = "htrc_flat_triton.py"; Url = "$UrlRoot/$SourceRef/h3/htrc_flat_triton.py" },
    @{ Name = "benchmark_h3_flat_semantic_shared_centers_fullq.py"; Url = "$UrlRoot/$BenchmarkRef/tools/benchmark_h3_flat_semantic_shared_centers_fullq.py" }
)
foreach ($File in $Files) {
    $Dest = Join-Path $WorkRoot $File.Name
    Write-Host "Fetching $($File.Name) ..." -ForegroundColor DarkCyan
    Invoke-WebRequest -UseBasicParsing -Uri $File.Url -OutFile $Dest
    if (-not (Test-Path $Dest -PathType Leaf)) { throw "Download failed: $Dest" }
}

$Runner = Join-Path $WorkRoot "benchmark_h3_flat_semantic_shared_centers_fullq.py"
& $Python -m py_compile $Runner (Join-Path $WorkRoot "htrc_flat.py") (Join-Path $WorkRoot "htrc_flat_triton.py")
if ($LASTEXITCODE -ne 0) { throw "Python syntax preflight failed" }

Write-Host "Python : $Python" -ForegroundColor Cyan
Write-Host "Capture: $Capture" -ForegroundColor Cyan
Write-Host "Result : $Out" -ForegroundColor Cyan
Write-Host "Running full-Q semantic shared-centers A/B (read-only, 4 physical heads)..." -ForegroundColor Green

$BenchArgs = @("--capture", $Capture, "--out", $Out, "--heads", "1,16,27,55",
    "--groups", "4,8,16", "--exact-leaves", "256", "--tile-chunk", "32",
    "--eval-queries", "256", "--rounds", "2", "--warmups", "1")
& $Python -B $Runner @BenchArgs

if ($LASTEXITCODE -ne 0) {
    throw "GPU A/B failed (exit $LASTEXITCODE); inspect partial JSON if present: $Out"
}
Write-Host "DONE: upload this ONE JSON to chat: $Out" -ForegroundColor Green
