param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExactHead,
    [ValidateSet('late_q','stage1','rb2','rb2_late_q')]
    [string]$Candidate = 'rb2_late_q',
    [ValidateRange(7,12)]
    [int]$Rounds = 7
)

# Research only. Disposable exact-HEAD checkout, one JSON, no real H3/video.
$ErrorActionPreference = 'Stop'
$repo = 'D:\HTRC-graph-group-ab-20261113'
$branch = 'experiment/flat-memory-to-speed-balanced-20261113'
$py = 'C:\Users\Misya\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Misyau\ComfyUI\.venv\Scripts\python.exe'
$output = 'D:\ComfyOutput\H3\math_probe\flat_tile_resource_abba.json'
$tmp = 'D:\HTRC-tile-resource-' + ([guid]::NewGuid().ToString('N').Substring(0,10))

if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw 'HTRC checkout missing' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'ComfyUI CUDA Python missing' }
if (Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Close ComfyUI before experiment'
}
if (Get-Process -Name blender -ErrorAction SilentlyContinue) {
    throw 'Close Blender before experiment'
}
if (Test-Path -LiteralPath $tmp) { throw 'Refuse to reuse temporary worktree' }

$original = if (Test-Path -LiteralPath $output -PathType Leaf) {
    (Get-Item -LiteralPath $output).LastWriteTimeUtc
} else { [datetime]::MinValue }

try {
    $remote = (git -C $repo remote get-url origin).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $remote) { throw 'Original Git origin unavailable' }
    git clone --no-checkout --no-hardlinks $repo $tmp
    if ($LASTEXITCODE -ne 0) { throw 'Temporary local clone failed' }
    git -C $tmp remote set-url origin $remote
    if ($LASTEXITCODE -ne 0) { throw 'Origin setup failed' }
    git -C $tmp fetch origin $branch
    if ($LASTEXITCODE -ne 0) { throw 'Remote branch fetch failed' }
    git -C $tmp switch --detach $ExactHead
    if ($LASTEXITCODE -ne 0) { throw 'Exact research HEAD unavailable' }
    if ((git -C $tmp rev-parse HEAD).Trim() -ne $ExactHead) {
        throw 'Pinned HEAD mismatch'
    }

    & $py -B -m unittest discover -s (Join-Path $tmp 'tools') `
        -p 'test_h3_tile_resource_abba_cpu.py' -v
    if ($LASTEXITCODE -ne 0) {
        throw 'CPU source and resource contract failed; GPU has NOT started'
    }

    & $py -B (Join-Path $tmp 'tools\benchmark_h3_tile_resource_abba.py') `
        --candidate $Candidate --rounds $Rounds --warmups 5 `
        --expect-head $ExactHead --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "Research experiment failed closed: $output"
    }
    if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or
        (Get-Item -LiteralPath $output).LastWriteTimeUtc -le $original) {
        throw 'Experiment did not produce a new ONE JSON'
    }
    $report = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
    if ($report.completed -ne $true -or
        $report.candidate -ne $Candidate -or
        $report.source_head -ne $ExactHead -or
        $report.production_promotion -ne 'PROHIBITED_RESEARCH_ONLY' -or
        $report.bank_partial_parity.Count -ne 2) {
        throw 'Experimental JSON invalid, partial, or missing numeric gates'
    }
    if ($Candidate -eq 'rb2_late_q') {
        if ($report.status -eq 'NO_GO_RESOURCE_GATE') {
            if ($report.resource_gate.admitted_for_tile_timing -ne $false -or
                $report.abba_blocks.Count -ne 0 -or
                $report.tile_only_speed.admission -ne 'UNRUN_RESOURCE_GATE') {
                throw 'Blocked resource gate unexpectedly included timing'
            }
        }
        elseif ($report.status -eq 'PASS_NUMERIC_TILE_ONLY') {
            if ($report.resource_gate.admitted_for_tile_timing -ne $true -or
                $report.abba_blocks.Count -ne 4) {
                throw 'Combined candidate timed without approved compiler resources'
            }
        }
        else { throw 'Unexpected combined experiment status' }
    }
    elseif ($report.status -ne 'PASS_NUMERIC_TILE_ONLY' -or
            $report.abba_blocks.Count -ne 4) {
        throw 'Existing one-knob variant must complete four-block ABBA'
    }
    if ($report.full_graph_first_call_eager_bit_exact -ne $true -or
        $report.full_graph_bank_b_eager_bit_exact -ne $true) {
        throw 'Missing full Graph/eager parity confirmation'
    }
    Write-Host "HTRC Tile32 research completed: $($report.status)"
    Write-Host "Kernel-only decision: $($report.tile_only_speed.admission)"
    if ($Candidate -eq 'rb2_late_q') {
        Write-Host "Compiler resource gate: $($report.resource_gate.status)"
    }
    Write-Host "ONE JSON: $output"
}
finally {
    if (Test-Path -LiteralPath $tmp) {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
}
