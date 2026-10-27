#!/bin/bash
# memwatch.sh start <csv> | stop <csv>
# 2 s samples of MemAvailable / Cached / swap-used (MiB). The sampler is fully detached (own session, all three
# standard streams redirected) so a `ssh host memwatch.sh start` returns at once instead of holding the channel open.
# `stop` kills the sampler and prints the sample count and the minimum MemAvailable.
case "${1:-}" in
  start)
    f=$2; : > "$f"
    setsid bash -c 'f=$1; while :; do awk -v t="$(date +%s)" '"'"'/MemAvailable/{a=$2}/^Cached:/{c=$2}/SwapTotal/{st=$2}/SwapFree/{sf=$2}END{printf "%s,%d,%d,%d\n",t,a/1024,c/1024,(st-sf)/1024}'"'"' /proc/meminfo >> "$f"; sleep 2; done' _ "$f" </dev/null >/dev/null 2>&1 &
    echo $! > "$f.pid";;
  stop)
    f=$2
    [ -f "$f.pid" ] && { pkill -g "$(cat "$f.pid")" 2>/dev/null; kill "$(cat "$f.pid")" 2>/dev/null; rm -f "$f.pid"; }
    [ -s "$f" ] && awk -F, 'NR==1||$2<m{m=$2} {n++} END{printf "samples=%d min_MemAvailable_MiB=%d\n", n, m}' "$f" || echo "samples=0 min_MemAvailable_MiB=unknown";;
  *) echo "usage: memwatch.sh start|stop <csv>" >&2; exit 2;;
esac
