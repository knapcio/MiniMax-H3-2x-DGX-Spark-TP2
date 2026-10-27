#!/bin/bash
# Run on each Spark before a boot. Exit 0 only when everything a boot needs is in place:
# fabric links ACTIVE with a RoCE v2 GID and an IPv4 netdev per HCA, >= 110 GiB MemAvailable (after --compact),
# the venv, the package and the weights present. Print versions to compare across the two nodes.
set -uo pipefail
source "$(dirname "$0")/env.sh"
rc=0
say() { printf '%-28s %s\n' "$1" "$2"; }
bad() { echo "  !! $1"; rc=1; }
# fabric
for hca in ${IB_HCA//,/ }; do
  st=$(cat /sys/class/infiniband/$hca/ports/1/state 2>/dev/null || echo missing)
  rate=$(cat /sys/class/infiniband/$hca/ports/1/rate 2>/dev/null || echo "?")
  typ=$(cat /sys/class/infiniband/$hca/ports/1/gid_attrs/types/$IB_GID_INDEX 2>/dev/null || echo "?")
  gid=$(cat /sys/class/infiniband/$hca/ports/1/gids/$IB_GID_INDEX 2>/dev/null || echo "?")
  dev=$(ls /sys/class/infiniband/$hca/device/net 2>/dev/null | head -1)
  ip4=$(ip -4 -br addr show dev "${dev:-none}" 2>/dev/null | awk '{print $3}')
  say "$hca" "$st, $rate, gid$IB_GID_INDEX=$typ ($gid) netdev=${dev:-?} ${ip4:-no-ipv4}"
  [[ "$st" == *ACTIVE* ]] || bad "$hca port not ACTIVE"
  [[ "$typ" == "RoCE v2" ]] || bad "$hca gid index $IB_GID_INDEX is not RoCE v2"
  [ -n "$ip4" ] || bad "$hca netdev has no IPv4 address (NCCL_IB_ADDR_FAMILY=AF_INET needs one)"
done
say "fabric if $FABRIC_IF" "$(ip -br addr show $FABRIC_IF 2>/dev/null | awk '{print $2, $3}')"
ip -br addr show "$FABRIC_IF" 2>/dev/null | grep -q UP || bad "$FABRIC_IF is not UP"
# memory (compaction first, then the floor)
FLOOR_MIB=112640   # 110 GiB
if [ "${1:-}" == "--compact" ]; then
  sync
  if echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 && echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1; then
    say "compaction" "done"
  else
    bad "compaction needs sudo: sudo's ticket is per parent shell, so run it yourself first: sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; echo 1 | sudo tee /proc/sys/vm/compact_memory"
  fi
fi
avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
say "MemAvailable" "${avail} MiB"
[ "$avail" -ge "$FLOOR_MIB" ] || bad "less than 110 GiB free: stop other GPU work first"
# software identity (compare across nodes)
say "kernel/driver" "$(uname -r) / $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
v=$($H3_VENV/bin/pip show vllm-omni 2>/dev/null | awk '/^Version/{print $2}'); say "vllm-omni" "${v:-missing}"; [ -n "$v" ] || bad "vllm-omni not installed in $H3_VENV"
p=$($H3_VENV/bin/python -c 'import h3_tp2;print(h3_tp2.__version__)' 2>/dev/null); say "h3_tp2" "${p:-missing} (rev $(cat "$(dirname "$0")/../.synced-rev" 2>/dev/null || echo unknown))"; [ -n "$p" ] || bad "h3_tp2 not installed in $H3_VENV"
if [ -d "$H3_MODEL" ]; then say "weights" "$(du -s --apparent-size "$H3_MODEL" | cut -f1) KB at $H3_MODEL"; else bad "weights missing at $H3_MODEL"; fi
say "control-channel hmac" "$([ -n "${H3_TP2_SECRET:-}" ] && echo on || echo off)"
[ $rc = 0 ] && echo "preflight: OK" || echo "preflight: FAIL"
exit $rc
