#!/usr/bin/env bash
# After the guarded quality queue only: install the CPU-tested prompt fix and
# disabled-by-default VAE trace on both nodes, smoke-test, then test the duration
# precedence through a real 16 s request. No host reboot or agent restart.
set -Eeuo pipefail

results=${H3_PATCH_RESULTS:-"$HOME/h3-tp2-results"}
txn=${H3_PATCH_TXN:-"$HOME/h3-tp2-results/file-txn-fc14580.py"}
live=${H3_PATCH_LIVE:-"$HOME/h3-tp2"}
stage=${H3_PATCH_STAGE:-"$HOME/h3-tp2-next-fc14580"}
backup=${H3_PATCH_BACKUP:-"$HOME/h3-tp2-backup-fc14580"}
bench=${H3_PATCH_BENCH:-"$HOME/h3-tp2/scripts/bench-matrix.sh"}
worker=${H3_PATCH_WORKER:-10.100.96.5}
meminfo=${H3_PATCH_MEMINFO:-/proc/meminfo}
proc_root=${H3_PATCH_PROC_ROOT:-/proc}
min_mem=${H3_PATCH_MIN_MEM_KB:-12582912}
wait_attempts=${H3_PATCH_WAIT_ATTEMPTS:-2880}
idle_attempts=${H3_PATCH_IDLE_ATTEMPTS:-720}
ready_attempts=${H3_PATCH_READY_ATTEMPTS:-100}
wait_s=${H3_PATCH_WAIT_S:-60}
idle_s=${H3_PATCH_IDLE_S:-10}
ready_s=${H3_PATCH_READY_S:-15}
txn_sha=b2904bd67bfa3985d2eedd8fa4974a97e4d0a41f27cb56dbe32fe726792dd2c1
head=h3-tp2-head.service
phase=waiting
cd "$results"
exec 9>"$results/prompt-patch-fc14580.lock"
flock -n 9 || { echo 'patch queue already running'; exit 2; }
for path in "$txn" "$live" "$stage" "$backup"; do
  [[ "$path" =~ ^/[a-zA-Z0-9._/-]+$ ]] || { echo 'unsupported path characters'; exit 1; }
done

mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a gen.log; }
health() { curl -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18093/health || true; }
wait_ready() {
  for ((j=0; j<ready_attempts; j++)); do
    [ "$(health)" = 200 ] && return 0
    sleep "$ready_s"
  done
  return 1
}
local_txn() { python3 "$txn" "$1" --live "$live" --stage "$stage" --backup "$backup"; }
remote_txn() {
  ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" \
    "python3 '$txn' '$1' --live '$live' --stage '$stage' --backup '$backup'"
}
restore_local() { local_txn verify >/dev/null 2>&1 || { local_txn restore && local_txn verify; }; }
restore_worker() { remote_txn verify >/dev/null 2>&1 || { remote_txn restore && remote_txn verify; }; }
on_exit() {
  local rc=$?
  ((rc)) || return 0
  mark "H3-AUTO-HOLD prompt-patch-fc14580 phase=$phase rc=$rc" || true
  case "$phase" in
    stopped|starting)
      # Restore both filesets before starting the old head. A newly started
      # process may already have accepted a request, so it must be verifiably
      # idle and have no async jobs before any second stop.
      if [ "$phase" = starting ]; then
        if systemctl --user is-active --quiet "$head"; then
          if ! wait_idle || ! no_jobs; then
            echo 'rollback: new head is active but not verifiably idle; leaving it untouched'
            return
          fi
          systemctl --user stop "$head" || { echo 'rollback: cannot stop new head'; return; }
        fi
      fi
      if restore_worker && restore_local; then
        if post_stop_idle && systemctl --user start "$head" && wait_ready; then
          mark 'H3-AUTO-PATCH-ROLLBACK-OK' || true
        else
          echo 'rollback: files restored; GPU did not become idle or head did not become healthy'
        fi
      else
        echo 'rollback incomplete; head remains stopped to avoid mixed revisions'
      fi
      ;;
    stopping)
      # No files were changed; restore service if the stop partially succeeded.
      systemctl --user is-active --quiet "$head" || systemctl --user start "$head" || true
      ;;
    healthy)
      # A smoke failure can leave an active request. Do not interrupt it or
      # overwrite either rank; leave the service and the failure on HOLD.
      echo 'head remains running; inspect failed smoke before any rollback'
      ;;
  esac
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

valid_case() { [ -s "$1/$2.mp4" ] && grep -Fxq "$2 ok" "$1/STATUS"; }
gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -q '^H3-AUTO-HOLD quality-20261030 ' gen.log 2>/dev/null; then
    echo 'quality queue held; no patch deployment'; exit 1
  fi
  finished=0
  if grep -q '^H3-AUTO-SKIP-300 ' gen.log 2>/dev/null; then finished=1; fi
  if grep -q '^H3-AUTO-QUALITY-DONE ' gen.log 2>/dev/null && \
     valid_case quality300-960-stream d300_960x576_s20_seed1101_fl2va; then finished=1; fi
  if ((finished)) && \
     valid_case quality16-960-stream d16_960x576_s20_seed1101_fl2va && \
     flock -n "$results/quality-20261030.lock" true; then
    gate=1; break
  fi
  sleep "$wait_s"
done
((gate)) || { echo 'quality queue did not finish with its verified 960 control'; exit 1; }

no_jobs() {
  curl -fsS -m 5 'http://127.0.0.1:18093/v1/videos?limit=1' | \
    python3 -c 'import json,sys; x=json.load(sys.stdin); sys.exit(0 if x.get("data") == [] and x.get("has_more") is False else 1)'
}
idle_sample() {
  local metrics gpu0 gpu1 mem0 mem1
  [ "$(health)" = 200 ] || return 1
  systemctl --user is-active --quiet "$head" || return 1
  systemctl --user is-active --quiet h3-tp2-memguard-head.service || return 1
  ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" \
    'systemctl --user is-active --quiet h3-tp2-agent.service && systemctl --user is-active --quiet h3-tp2-memguard-worker.service' || return 1
  metrics=$(curl -fsS -m 5 http://127.0.0.1:18093/metrics) || return 1
  printf '%s\n' "$metrics" | awk '
    /^vllm_omni:num_requests_running\{/ {r++; if ($NF != 0) bad=1}
    /^vllm_omni:num_requests_waiting\{/ {w++; if ($NF != 0) bad=1}
    END {exit !(r > 0 && w > 0 && !bad)}' || return 1
  gpu0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
  gpu1=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" \
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
  mem0=$(awk '/^MemAvailable:/ {print $2}' "$meminfo") || return 1
  mem1=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" \
    "awk '/^MemAvailable:/ {print \$2}' /proc/meminfo") || return 1
  [[ "$gpu0" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ && "$gpu1" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ &&
     "$mem0" =~ ^[0-9]+$ && "$mem1" =~ ^[0-9]+$ ]] || return 1
  ((gpu0 < 10 && gpu1 < 10 && mem0 >= min_mem && mem1 >= min_mem))
}
wait_idle() {
  local consecutive=0
  for ((j=0; j<idle_attempts; j++)); do
    if idle_sample; then consecutive=$((consecutive+1)); else consecutive=0; fi
    ((consecutive >= 3)) && return 0
    sleep "$idle_s"
  done
  return 1
}
post_stop_idle() {
  local gpu0 gpu1 consecutive=0
  for ((j=0; j<idle_attempts; j++)); do
    gpu0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
    gpu1=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
    if [[ "$gpu0" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ && "$gpu1" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] && \
       ((gpu0 < 10 && gpu1 < 10)); then consecutive=$((consecutive+1)); else consecutive=0; fi
    ((consecutive >= 3)) && return 0
    sleep "$idle_s"
  done
  return 1
}

[[ "$(sha256sum "$txn" | awk '{print $1}')" = "$txn_sha" ]]
[[ "$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "sha256sum '$txn'" | awk '{print $1}')" = "$txn_sha" ]]
local_txn verify
remote_txn verify
[ ! -e "$backup" ]
ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "test ! -e '$backup'"
wait_idle
no_jobs || { echo 'async jobs exist or cannot be enumerated; refusing restart'; exit 1; }
mark 'H3-AUTO-PATCH-START fc14580'
phase=stopping
systemctl --user stop "$head"
phase=stopped
post_stop_idle
remote_txn apply
local_txn apply
remote_txn verify-new
local_txn verify-new
phase=starting
systemctl --user start "$head"
wait_ready
phase=healthy
pid=$(systemctl --user show "$head" -p MainPID --value)
python3 - "$proc_root" "$pid" <<'PY'
from pathlib import Path
import sys
env = set((Path(sys.argv[1]) / sys.argv[2] / 'environ').read_bytes().split(b'\0'))
assert b'H3_TP2_VAE_BLEND_TRACE=1' not in env, 'experimental trace unexpectedly enabled'
for item in (b'H3_TP2_REQUEST_TIMEOUT_S=28800', b'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=28800',
             b'VLLM_OMNI_VIDEO_SYNC_TIMEOUT=29400'):
    assert item in env, 'expected server timeout missing'
PY
mark 'H3-AUTO-PATCH-HEAD-READY fc14580'

run_case() {
  local name=$1 spec=$2 tag=$3 limit=$4
  shift 4
  local out="$results/$name"
  [ ! -e "$out" ] || return 1
  wait_idle
  mark "H3-AUTO-START $name"
  OUT="$out" timeout --signal=TERM --kill-after=30s "$limit" bash "$bench" "$spec" "$@" > "$results/$name.log" 2>&1
  valid_case "$out" "$tag"
  mark "H3-AUTO-OK $name"
}
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL EXTRA_JSON || true
export TASK=t2va PROMPT='A calm landscape, one steady camera, soft natural ambience.'
run_case patch-smoke5 5:960:576:10:0 d5_960x576_s10_seed0 3600
export TASK=fl2va
export PROMPT='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, their shadows glide over the plain, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road walk on. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'
export EXTRA_JSON='{"long_video":true,"long_video_mode":"continuation","duration_seconds":30,"target":{"duration_seconds":16}}'
started=$(date '+%F %T')
run_case patch-duration16 16:1344:768:20:1101 d16_1344x768_s20_seed1101_fl2va 7500 "$results/koninck.png"
python3 - "$results/patch-duration16/d16_1344x768_s20_seed1101_fl2va.mp4.timing.json" <<'PY'
import json,sys
with open(sys.argv[1]) as f: timing = json.load(f)
assert timing['effective_duration'] == 16 and timing['extra_params']['duration_seconds'] == 30
assert timing['extra_params']['target']['duration_seconds'] == 16
PY
journalctl --user -u "$head" --since "$started" --no-pager | grep -q 'continuation prompts generated for 2 windows (16.0 s)'
mark 'H3-AUTO-PATCH-DONE fc14580'
