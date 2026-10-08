# Dot-sourced by deploy.ps1 and start.ps1: reads rig/rig.env into $Rig and defines Run.

$RigEnvFile = Join-Path $PSScriptRoot 'rig.env'
if (-not (Test-Path $RigEnvFile)) {
    Write-Host "rig\rig.env not found. Copy rig.env.example to rig.env and fill it in." -ForegroundColor Red
    exit 1
}
$Rig = @{}
foreach ($line in Get-Content $RigEnvFile) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$') {
        $Rig[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'")
    }
}
foreach ($required in 'ILL_HEAD_NODE', 'ILL_REMOTE_PROJECT') {
    if (-not $Rig[$required]) {
        Write-Host "$required is not set in rig\rig.env." -ForegroundColor Red
        exit 1
    }
}
$LocalProject = Split-Path -Parent $PSScriptRoot
$HeadNode = $Rig['ILL_HEAD_NODE']
$RemoteProject = $Rig['ILL_REMOTE_PROJECT']

function Run($label, $cmd) {
    Write-Host "`n--- $label ---"
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Write-Host "OK" -ForegroundColor Green
}
