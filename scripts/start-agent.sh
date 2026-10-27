#!/bin/bash
# Worker (rank 1): the agent that launches the diffusion worker on request. Run on the worker Spark.
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$H3_VENV/bin/python" -m h3_tp2.agent --bind "$WORKER_IP:$AGENT_PORT" --python "$H3_VENV/bin/python" --log-dir "$HOME/h3-tp2-logs"
