# Start a session on the head node from this workstation.
#
#   powershell -ExecutionPolicy Bypass -File rig\start.ps1
#
# Runs rig/start.sh on the head node's local display (DISPLAY=:0), so the head node must
# have a logged-in graphical session. Head node and checkout path come from rig\rig.env.
. (Join-Path $PSScriptRoot '_rig_env.ps1')

$Screen = $Rig['ILL_HEAD_SCREEN']
if (-not $Screen) { $Screen = '1680x1050' }

Run "Starting the experiment on the head node" `
    "ssh $HeadNode `"cd '$RemoteProject' && chmod +x rig/*.sh 'Builds/Linux Main/MainExperiment.x86_64' 2>/dev/null; DISPLAY=:0 xrandr --fb $Screen 2>/dev/null; DISPLAY=:0 rig/start.sh`""

Write-Host "`n--- Done ---" -ForegroundColor Green
