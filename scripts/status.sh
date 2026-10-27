#!/bin/bash
# Quick state of both ranks (run on the head; needs ssh to the worker by alias or IP in WORKER_SSH).
source "$(dirname "$0")/env.sh"
WORKER_SSH=${WORKER_SSH:-knapcio@$WORKER_IP}
echo "== head $(hostname)"; systemctl --user is-active h3-tp2-head.service 2>/dev/null; curl -s -m 5 -o /dev/null -w "health %{http_code}\n" "http://$H3_API_HOST:$H3_API_PORT/health"; awk '/MemAvailable/{printf "MemAvailable %d MiB\n",$2/1024}' /proc/meminfo
echo "== worker"; ssh -o ConnectTimeout=5 "$WORKER_SSH" 'systemctl --user is-active h3-tp2-agent.service; pgrep -fc "h3_tp2[.]remote_worker|DiffusionWorker-[0-9]" | sed "s/^/rank1 procs: /"; awk "/MemAvailable/{printf \"MemAvailable %d MiB\n\",\$2/1024}" /proc/meminfo'
