#!/bin/bash
# Unified-memory guard for GB10 (after our spark-06 incident and tonyd2wild's h3_guard.sh): when MemAvailable stays
# below the floor, kill the rank before the NVIDIA driver exhausts memory and the whole host livelocks.
#   MEMGUARD_ROLE=head    SIGKILL the head unit and restart it (systemd would otherwise wait TimeoutStopSec under pressure)
#   MEMGUARD_ROLE=worker  SIGKILL the rank-1 process (the agent reports 'dead', the head fails closed and restarts)
#   MEMGUARD_ACTION=...   explicit command instead of a role
THRESH_KB=${MEMGUARD_THRESH_KB:-4194304}   # 4 GiB
HITS_NEEDED=${MEMGUARD_HITS:-2}
INTERVAL=${MEMGUARD_INTERVAL:-2}
COOLDOWN=${MEMGUARD_COOLDOWN:-180}
ROLE=${MEMGUARD_ROLE:-}
case "$ROLE" in
  head)   ACTION='systemctl --user kill -s KILL h3-tp2-head.service; sleep 2; systemctl --user restart h3-tp2-head.service';;
  worker) ACTION='p=$(cat "$HOME/h3-tp2-logs/rank1.pid" 2>/dev/null); if [ -n "$p" ]; then kill -KILL -- "-$p"; else pkill -KILL -f "h3_tp2[.]remote_worker"; fi';;   # the agent records the rank-1 process group
  *)      ACTION=${MEMGUARD_ACTION:?set MEMGUARD_ROLE=head|worker or MEMGUARD_ACTION};;
esac
avail() { local k v _; while read -r k v _; do [ "$k" = "MemAvailable:" ] && { echo "$v"; return; }; done < /proc/meminfo; echo 0; }
hits=0
echo "memguard: role=${ROLE:-custom} thresh=${THRESH_KB} kB hits=${HITS_NEEDED} interval=${INTERVAL}s action='${ACTION}'"
while :; do
  a=$(avail)
  if [ "$a" -lt "$THRESH_KB" ]; then
    hits=$((hits+1)); echo "low MemAvailable ${a} kB (hit ${hits}/${HITS_NEEDED})"
    if [ "$hits" -ge "$HITS_NEEDED" ]; then
      echo "TRIP: MemAvailable ${a} kB < ${THRESH_KB} kB -> ${ACTION}"
      bash -c "$ACTION"; rc=$?
      sleep 10; b=$(avail)
      if [ "$b" -lt "$THRESH_KB" ]; then echo "action rc=$rc but MemAvailable still ${b} kB: retrying after ${INTERVAL}s"; hits=$((HITS_NEEDED-1)); sleep "$INTERVAL"; continue; fi
      echo "recovered: MemAvailable ${b} kB (action rc=$rc); cooldown ${COOLDOWN}s"; hits=0; sleep "$COOLDOWN"; continue
    fi
  else
    hits=0
  fi
  sleep "$INTERVAL"
done
