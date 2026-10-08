#!/bin/bash
# Headless Unity build of the experiment.
#
#   rig/build.sh            # Linux head-node build  -> Builds/Linux Main/
#   rig/build.sh windows    # Windows render-node build -> Builds/Windows Main/
#                           # (needs the editor's Windows build support)
#
# The editor path comes from ILL_UNITY_EDITOR in rig/rig.env. The build log is written
# to build.log in the project root.
. "$(dirname "$0")/_load_env.sh"

if [ ! -x "$ILL_UNITY_EDITOR" ]; then
    echo "ERROR: Unity editor not found at '$ILL_UNITY_EDITOR' (ILL_UNITY_EDITOR in rig/rig.env)." >&2
    exit 1
fi

case "${1:-linux}" in
    linux)   target=(-buildLinux64Player "$PROJECT_DIR/Builds/Linux Main/MainExperiment") ;;
    windows) target=(-buildWindows64Player "$PROJECT_DIR/Builds/Windows Main/MainExperiment.exe") ;;
    *) echo "usage: rig/build.sh [linux|windows]" >&2; exit 2 ;;
esac

echo "Starting Unity build (${1:-linux})..."
"$ILL_UNITY_EDITOR" -batchmode -quit -nographics \
    -projectPath "$PROJECT_DIR" \
    "${target[@]}" \
    -logFile "$PROJECT_DIR/build.log"
status=$?
[ $status -eq 0 ] && echo "Build finished." || echo "Build FAILED (exit $status); see build.log." >&2
exit $status
