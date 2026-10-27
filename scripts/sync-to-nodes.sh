#!/bin/bash
# Mac-side helper: copy this repo to the given hosts and (re)install the package into the H3 venv.
# Stops at the first host that fails (rsync or install), so the two nodes never silently run different revisions.
# Do this while the TP2 units are stopped if the Python package changed: the editable install is read at process start.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
VENV=${H3_VENV:-/home/knapcio/h3-base-20260916/runtime/.venv}
DEST=${H3_TP2_DEST:-/home/knapcio/h3-tp2}
rev=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo uncommitted)
for h in "$@"; do
  rsync -a --delete --exclude .git --exclude results "$HERE/" "$h:$DEST/" || { echo "rsync to $h failed"; exit 1; }
  ssh -o BatchMode=yes "$h" "echo '$rev' > $DEST/.synced-rev && $VENV/bin/pip install -q -e $DEST && $VENV/bin/python -c 'import h3_tp2.executor, h3_tp2.agent, h3_tp2.remote_worker; print(\"h3_tp2 ok on \" + __import__(\"socket\").gethostname())'" || { echo "install on $h failed"; exit 1; }
done
