# Copy freshly built players from this workstation to the head node, then push the
# Windows build on to the render nodes (rig/dist.sh on the head node).
#
#   powershell -ExecutionPolicy Bypass -File rig\deploy.ps1
#
# Expects Builds\Windows Main and Builds\Linux Main in the local checkout. The head node
# and its checkout path come from rig\rig.env.
. (Join-Path $PSScriptRoot '_rig_env.ps1')

Run "Preparing the head node" `
    "ssh $HeadNode `"cd '$RemoteProject' && chmod +x rig/*.sh && rm -rf 'Builds/Windows Main' 'Builds/Linux Main' && mkdir -p Builds`""

Run "Copying the Windows build to the head node" `
    "scp -r `"$LocalProject\Builds\Windows Main`" `"${HeadNode}:$RemoteProject/Builds/`""

Run "Copying the Linux build to the head node" `
    "scp -r `"$LocalProject\Builds\Linux Main`" `"${HeadNode}:$RemoteProject/Builds/`""

Run "Setting Linux build permissions" `
    "ssh $HeadNode `"chmod +x '$RemoteProject/Builds/Linux Main/MainExperiment.x86_64'`""

Run "Distributing to the render nodes" `
    "ssh $HeadNode `"cd '$RemoteProject' && rig/dist.sh`""

Write-Host "`n--- Done. Builds deployed; start the session on the head node (rig/start.sh). ---" -ForegroundColor Green
