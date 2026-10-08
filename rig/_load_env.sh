# Sourced by the rig scripts: loads rig/rig.env and moves to the project root.
RIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$RIG_DIR")"
if [ ! -f "$RIG_DIR/rig.env" ]; then
    echo "ERROR: $RIG_DIR/rig.env not found. Copy rig.env.example to rig.env and fill it in." >&2
    exit 1
fi
# shellcheck source=/dev/null
. "$RIG_DIR/rig.env"
cd "$PROJECT_DIR" || exit 1
