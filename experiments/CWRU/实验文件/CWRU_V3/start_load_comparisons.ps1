param(
    [switch]$Plan,
    [switch]$Status,
    [switch]$Pause,
    [switch]$Resume,
    [string]$Output = '',
    [string]$Method = '',
    [ValidateSet('small', 'full')]
    [string]$Sample = 'small',
    [string]$Ratio = 'all'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$taskPython = 'D:\anaconda\envs\pytorch_GPU\python.exe'
$taskRunner = Join-Path $PSScriptRoot 'run_load_comparisons.py'
$resultsRoot = Join-Path $projectRoot 'experiments\CWRU\实验结果\CWRU_V3'
$allMethods = @('WDCNN', 'DRSN_CW', 'PG_RDGFN_adapted', 'AW_Hybrid_adapted')
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw "Python environment not found: $taskPython"
}
# -Method all runs every method that is not complete yet for the chosen -Sample, one after another.
# -Sample small: 10/20/50% x 3 subsets (180 models per method); -Sample full: full training pool (20 per method).
# -Ratio only applies to -Sample small (all, 0.1, 0.2, 0.5).
$selectedModes = @($Plan, $Status, $Pause, $Resume | Where-Object { $_ }).Count
if ($selectedModes -gt 1) { throw 'Choose only one of -Plan, -Status, -Pause, or -Resume.' }
if ($Method -eq '') {
    if ($selectedModes -eq 0) {
        throw 'Add -Method all (or a single method). Optional: -Sample full|small and -Ratio 0.1/0.2/0.5.'
    }
    if ($Plan) { $targets = $allMethods } else { $targets = @() }
} elseif ($Method -eq 'all') {
    $targets = $allMethods
} else {
    if ($allMethods -notcontains $Method) {
        throw "Unknown method: $Method. Use all, WDCNN, DRSN_CW, PG_RDGFN_adapted or AW_Hybrid_adapted."
    }
    $targets = @($Method)
}
Set-Location -LiteralPath $projectRoot
$taskEnv = Split-Path -Parent $taskPython
$env:PATH = "$taskEnv;$taskEnv\Library\bin;$taskEnv\Scripts;" + $env:PATH
$env:PYTHONUTF8 = '1'

function Invoke-Task([string[]]$taskArgs) {
    & $taskPython -X utf8 -B $taskRunner @taskArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Get-ScopeRuns([string]$name, [string]$sample) {
    # Only directories written by this entry are considered; they carry scope.method and scope.sample.
    if (-not (Test-Path -LiteralPath $resultsRoot)) { return @() }
    $found = @()
    foreach ($dir in Get-ChildItem -LiteralPath $resultsRoot -Directory -Filter 'load_comparison_*') {
        $cfgPath = Join-Path $dir.FullName 'config.json'
        if (-not (Test-Path -LiteralPath $cfgPath)) { continue }
        try { $cfg = Get-Content -LiteralPath $cfgPath -Raw | ConvertFrom-Json } catch { continue }
        if ($null -eq $cfg.scope) { continue }
        $storedSample = if ($cfg.scope.sample) { [string]$cfg.scope.sample } else { 'small' }
        if ($cfg.scope.method -ne $name -or $storedSample -ne $sample) { continue }
        $found += [pscustomobject]@{ Path = $dir.FullName; Stage = [string]$cfg.stage; Started = [string]$cfg.started_at }
    }
    return $found
}

$scope = @()
if ($Method -and $Method -ne 'all') { $scope += @('--method', $Method) }
$scope += @('--sample', $Sample)
$scope += @('--ratio', $Ratio)

if ($Plan) {
    foreach ($name in $targets) {
        $runs = @(Get-ScopeRuns $name $Sample)
        $complete = @($runs | Where-Object { $_.Stage -eq 'complete' })
        $pending = @($runs | Where-Object { $_.Stage -ne 'complete' })
        if ($complete.Count -gt 0) {
            Write-Host "[skip] $name ($Sample) : already complete -> $($complete[0].Path)"
            continue
        }
        if ($pending.Count -gt 0) {
            Write-Host "[stop] $name ($Sample) : unfinished run (stage=$($pending[0].Stage)) -> $($pending[0].Path)"
            continue
        }
        Invoke-Task @('plan', '--method', $name, '--sample', $Sample, '--ratio', $Ratio)
    }
} elseif ($Status) {
    if (-not $Output) { throw 'Provide -Output with the experiment directory for -Status.' }
    Invoke-Task @('status', '--output', $Output)
} elseif ($Pause) {
    if (-not $Output) { throw 'Provide -Output with the running experiment directory for -Pause.' }
    Invoke-Task @('pause', '--output', $Output)
} elseif ($Resume) {
    if (-not $Output) { throw 'Provide -Output with the paused experiment directory for -Resume.' }
    Invoke-Task (@('resume', '--output', $Output) + $scope)
} else {
    if ($Output) {
        throw '-Output is only used with -Status/-Pause/-Resume; batch runs create a new directory per method.'
    }
    $summary = @()
    $failed = $false
    $index = 0
    foreach ($name in $targets) {
        $index += 1
        $runs = @(Get-ScopeRuns $name $Sample)
        $complete = @($runs | Where-Object { $_.Stage -eq 'complete' })
        if ($complete.Count -gt 0) {
            Write-Host "[$index/$($targets.Count)] $name ($Sample) : already complete, skip -> $($complete[0].Path)"
            $summary += "$name=skipped"
            continue
        }
        $pending = @($runs | Where-Object { $_.Stage -ne 'complete' })
        if ($pending.Count -gt 0) {
            Write-Host "[$index/$($targets.Count)] $name ($Sample) : STOP, unfinished run exists (stage=$($pending[0].Stage)) -> $($pending[0].Path)"
            Write-Host '    Inspect that directory and use -Resume -Output <dir>; no new run was started.'
            $summary += "$name=unfinished"
            $failed = $true
            continue
        }
        $newDir = Join-Path $resultsRoot ('load_comparison_' + $Sample + '_' + $name + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
        Write-Host ''
        Write-Host "[$index/$($targets.Count)] $name ($Sample) -> $newDir"
        & $taskPython -X utf8 -B $taskRunner run --output $newDir --method $name --sample $Sample --ratio $Ratio
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            Write-Host "[$index/$($targets.Count)] $name FAILED with exit code $code; directory kept: $newDir"
            $summary += "$name=failed"
            $failed = $true
        } else {
            Write-Host "[$index/$($targets.Count)] $name complete -> $newDir"
            $summary += "$name=complete"
        }
    }
    Write-Host ''
    Write-Host ('summary: ' + ($summary -join '; '))
    if ($failed) { exit 1 }
}
