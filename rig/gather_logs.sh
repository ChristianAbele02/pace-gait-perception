#!/bin/bash
# Fetch the Unity player log (output_log.txt) from every render node into ./RenderLogs/.
#
# The log sits in the player's LocalLow folder, named after the Unity companyName and
# productName in ProjectSettings ("DefaultCompany" / "Hurdling Demo").
. "$(dirname "$0")/_load_env.sh"

DEST="$PROJECT_DIR/RenderLogs"
mkdir -p "$DEST"

n=0
for pair in $ILL_RENDER_NODES; do
    n=$((n + 1))
    node="${pair%%:*}"
    user="${pair#*:}"
    echo "--- Fetching from $node ---"
    REMOTE_PATH="C:\\Users\\$user\\AppData\\LocalLow\\DefaultCompany\\Hurdling Demo\\output_log.txt"
    ssh -n -i "$ILL_RENDER_KEY" "$user@$node" "type \"$REMOTE_PATH\"" \
        > "$DEST/output_log_node${n}.txt" 2> "$DEST/output_log_node${n}.err"
    if [ -s "$DEST/output_log_node${n}.txt" ]; then
        echo "  Saved $(wc -l < "$DEST/output_log_node${n}.txt") lines"
    else
        echo "  No log retrieved (see output_log_node${n}.err)"
        cat "$DEST/output_log_node${n}.err"
    fi
done

echo
echo "Logs in $DEST:"
ls -la "$DEST"
