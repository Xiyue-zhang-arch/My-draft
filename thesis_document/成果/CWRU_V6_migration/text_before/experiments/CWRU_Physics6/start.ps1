param([ValidateSet('plan','check','run')][string]$Command='plan', [string]$Output='')
$ErrorActionPreference='Stop'
$taskProject=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskPython='D:\anaconda\envs\pytorch_GPU\python.exe'
$taskEnvironment=Split-Path -Parent $taskPython
$env:PATH="$taskEnvironment;$taskEnvironment\Library\bin;$taskEnvironment\Scripts;"+$env:PATH
$env:PYTHONUTF8='1'
Set-Location -LiteralPath $taskProject
$taskArguments=@('-X','utf8','-B','-m','experiments.CWRU_Physics6.run',$Command)
if ($Output) { $taskArguments+=@('--output',$Output) }
& $taskPython @taskArguments
exit $LASTEXITCODE
