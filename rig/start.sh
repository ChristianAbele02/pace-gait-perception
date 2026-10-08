#!/bin/bash
# Start a session: launch the render-node builds, then the head-node build.
#
# Run on the head node from a graphical terminal (the Linux player needs a display;
# a plain SSH session fails with "Can't open display").
. "$(dirname "$0")/_load_env.sh"
export LANG=C

if [ -n "$ILL_PYTHON_VENV" ] && [ -f "$ILL_PYTHON_VENV/bin/activate" ]; then
    # shellcheck source=/dev/null
    . "$ILL_PYTHON_VENV/bin/activate"
fi

python3 "$RIG_DIR/ill.py" stop
python3 "$RIG_DIR/ill.py" start

width="${ILL_HEAD_SCREEN%x*}"
height="${ILL_HEAD_SCREEN#*x}"
"Builds/Linux Main/MainExperiment.x86_64" -logFile ~/unity_main.log \
    -screen-width "${width:-1680}" -screen-height "${height:-1050}"
