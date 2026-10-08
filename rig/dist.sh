#!/bin/bash
# Copy the Windows build from the head node to every render node.
#
# The render nodes are Windows machines: SSH lands in a Windows shell, so the remote
# commands go through powershell.exe and only the transfer itself is scp.
#
# Each node gets a full clean replace: the target folder is wiped, recreated and then
# filled with the fresh build. Copying over an old build leaves stale files behind, and
# a render node running a mix of two builds shows flicker and split scenes.
#
# Run on the head node:
#   rig/ill.py stop     # the running app locks its files
#   rig/dist.sh
#
# Hosts, users, key and target folder come from rig/rig.env.
. "$(dirname "$0")/_load_env.sh"

SRC="Builds/Windows Main"
DEST="$ILL_RENDER_DEST"

if [ ! -d "$SRC" ]; then
    echo "ERROR: '$SRC' not found. Build the Windows player first (rig/build.sh windows)." >&2
    exit 1
fi

SRC_COUNT=$(find "$SRC" -type f | wc -l)
echo "========================================================"
echo " Distributing build to render nodes (Windows)"
echo " Source : $SRC ($SRC_COUNT files)"
echo " Target : <node>:$DEST"
echo "========================================================"

OVERALL_OK=1
SSH_OPTS=(-n -o BatchMode=yes -o ConnectTimeout=15 -i "$ILL_RENDER_KEY")

for pair in $ILL_RENDER_NODES; do
    node="${pair%%:*}"
    user="${pair#*:}"
    remote="$user@$node"
    echo ""
    echo "--------------------------------------------------------"
    echo " Node: $node  (user: $user)"
    echo "--------------------------------------------------------"

    # 1) Wipe and recreate the target folder. -n, -NonInteractive and the trailing
    #    "exit" stop the Windows OpenSSH channel from hanging open afterwards.
    echo -n " [1/3] Wiping and recreating $DEST ... "
    if ssh "${SSH_OPTS[@]}" "$remote" "powershell -NoProfile -NonInteractive -Command \"Remove-Item -Recurse -Force '$DEST' -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path '$DEST' | Out-Null; exit\""; then
        echo "OK"
    else
        echo "FAILED (could not reach $node or wipe the folder)"
        OVERALL_OK=0
        continue
    fi

    # 2) Copy the fresh build.
    echo -n " [2/3] Transferring $SRC_COUNT files ... "
    if scp -q -r -i "$ILL_RENDER_KEY" "$SRC"/* "$remote:$DEST/"; then
        echo "OK"
    else
        echo "FAILED (scp error; is the app still running? run rig/ill.py stop)"
        OVERALL_OK=0
        continue
    fi

    # 3) Check that the key build files arrived. Test-Path only (no pipes), so it works
    #    on any PowerShell version and survives the bash -> ssh -> Windows quoting.
    echo -n " [3/3] Verifying key files on the node ... "
    CHECK=$(ssh "${SSH_OPTS[@]}" "$remote" "powershell -NoProfile -NonInteractive -Command \"if ((Test-Path '$DEST/MainExperiment.exe') -and (Test-Path '$DEST/MainExperiment_Data/Managed/Assembly-CSharp.dll') -and (Test-Path '$DEST/MainExperiment_Data/sharedassets1.assets')) { 'ALLOK' } else { 'MISSING' }; exit\"" 2>/dev/null | tr -d '[:space:]')
    if [ "$CHECK" = "ALLOK" ]; then
        echo "OK"
        echo " => $node deployed."
    else
        echo "WARNING (could not confirm the key files; got '$CHECK')"
        echo " => check $node by hand; the transfer itself reported OK."
    fi
done

echo ""
echo "========================================================"
if [ "$OVERALL_OK" -eq 1 ]; then
    echo " DONE: every render node wiped, refilled and verified."
else
    echo " DONE WITH ERRORS: at least one node failed, see above."
fi
echo "========================================================"

[ "$OVERALL_OK" -eq 1 ]
