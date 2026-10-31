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
wait_attempts=${H3_PATCH_WAIT_ATTEMPTS:-5760}
idle_attempts=${H3_PATCH_IDLE_ATTEMPTS:-720}
ready_attempts=${H3_PATCH_READY_ATTEMPTS:-100}
wait_s=${H3_PATCH_WAIT_S:-60}
idle_s=${H3_PATCH_IDLE_S:-10}
ready_s=${H3_PATCH_READY_S:-15}
txn_sha=b2904bd67bfa3985d2eedd8fa4974a97e4d0a41f27cb56dbe32fe726792dd2c1
head=h3-tp2-head.service
phase=waiting
cd "$results"
exec 9>"$results/prompt-patch-aspect-20261031.lock"
flock -n 9 || { echo 'patch queue already running'; exit 2; }
for path in "$txn" "$live" "$stage" "$backup"; do
  [[ "$path" =~ ^/[a-zA-Z0-9._/-]+$ ]] || { echo 'unsupported path characters'; exit 1; }
done

mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a aspect-events-20261031.log; }
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
ensure_stopped() {
  local state pid
  state=$(systemctl --user show "$head" -p ActiveState --value) || return 1
  pid=$(systemctl --user show "$head" -p MainPID --value) || return 1
  [[ "$state" = inactive || "$state" = failed ]] && [[ "$pid" = 0 ]]
}
on_exit() {
  local rc=$? state
  ((rc)) || return 0
  mark "H3-ASPECT-HOLD prompt-patch-fc14580 phase=$phase rc=$rc" || true
  case "$phase" in
    stopped|starting)
      # Restore both filesets before starting the old head. A newly started
      # process may already have accepted a request, so it must be verifiably
      # idle and have no async jobs before any second stop.
      if [ "$phase" = starting ]; then
        state=$(systemctl --user show "$head" -p ActiveState --value) || return
        case "$state" in
          active)
            if ! wait_idle || ! no_jobs; then
              echo 'rollback: new head is active but not verifiably idle; leaving it untouched'
              return
            fi
            systemctl --user stop "$head" || { echo 'rollback: cannot stop new head'; return; }
            ;;
          inactive|failed)
            # Cancel any queued systemd auto-restart before touching files.
            systemctl --user stop "$head" || { echo 'rollback: cannot quiesce stopped head'; return; }
            ;;
          *) echo "rollback: head state $state is not safely stopped; leaving it untouched"; return;;
        esac
      fi
      ensure_stopped || { echo 'rollback: head is not safely stopped; leaving files untouched'; return; }
      if restore_worker && restore_local; then
        if post_stop_idle && systemctl --user start "$head" && wait_ready; then
          mark 'H3-ASPECT-PATCH-ROLLBACK-OK' || true
        else
          echo 'rollback: files restored; GPU did not become idle or head did not become healthy'
        fi
      else
        echo 'rollback incomplete; head remains stopped to avoid mixed revisions'
      fi
      ;;
    stopping)
      # No files were changed; restore service if the stop partially succeeded.
      state=$(systemctl --user show "$head" -p ActiveState --value) || return
      case "$state" in inactive|failed) systemctl --user start "$head" || true;; esac
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
  if grep -qE '^H3-ASPECT-HOLD (quality5-960x544-20261031|quality16-960x544-20261031|quality-20261031|long300-20261031) ' aspect-events-20261031.log 2>/dev/null; then
    echo 'aspect predecessor held; no patch deployment'; exit 1
  fi
  finished=0
  if grep -q '^H3-ASPECT-LONG300-SKIPPED ' aspect-events-20261031.log 2>/dev/null; then finished=1; fi
  if grep -q '^H3-ASPECT-LONG300-DONE ' aspect-events-20261031.log 2>/dev/null && \
     valid_case long300-960x544-20261031 d300_960x544_s20_seed1101_fl2va; then finished=1; fi
  if ((finished)) && \
     grep -q '^H3-ASPECT-OK quality16-960x544-20261031 ' aspect-events-20261031.log && \
     grep -q '^H3-ASPECT-OK quality60-960x544-20261031 ' aspect-events-20261031.log && \
     valid_case quality16-960x544-20261031 d16_960x544_s20_seed1101_fl2va && \
     valid_case quality60-960x544-20261031 d60_960x544_s20_seed1101_fl2va && \
     flock -n "$results/quality-aspect-20261031.lock" true && \
     flock -n "$results/long300-aspect-20261031.lock" true; then
    gate=1; break
  fi
  sleep "$wait_s"
done
((gate)) || { echo 'aspect controls and fallback did not finish safely'; exit 1; }

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
expected_timeouts=8h
if grep -q '^H3-ASPECT-LONG300-DONE ' aspect-events-20261031.log; then expected_timeouts=16h; fi
check_expected_pid=$(systemctl --user show "$head" -p MainPID --value)
python3 - "$proc_root" "$check_expected_pid" "$expected_timeouts" <<'PY'
from pathlib import Path
import sys
env=dict(x.split(b'=',1) for x in (Path(sys.argv[1])/sys.argv[2]/'environ').read_bytes().split(b'\0') if b'=' in x)
values=(b'57600',b'57600',b'58200') if sys.argv[3]=='16h' else (b'28800',b'28800',b'29400')
for key,value in zip((b'H3_TP2_REQUEST_TIMEOUT_S',b'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT',b'VLLM_OMNI_VIDEO_SYNC_TIMEOUT'),values):
    if env.get(key) != value: raise SystemExit('unexpected server timeout before patch')
if env.get(b'H3_TP2_VAE_BLEND_TRACE',b'0')==b'1': raise SystemExit('experimental trace unexpectedly enabled')
PY
mark 'H3-ASPECT-PATCH-START fc14580'
phase=stopping
systemctl --user stop "$head"
phase=stopped
ensure_stopped
post_stop_idle
ensure_stopped
remote_txn apply
ensure_stopped
local_txn apply
remote_txn verify-new
local_txn verify-new
phase=starting
systemctl --user start "$head"
wait_ready
phase=healthy
check_head_timeouts() {
  local pid
  pid=$(systemctl --user show "$head" -p MainPID --value) || return 1
  python3 - "$proc_root" "$pid" "$1" <<'PY'
from pathlib import Path
import sys
env = dict(x.split(b'=',1) for x in (Path(sys.argv[1])/sys.argv[2]/'environ').read_bytes().split(b'\0') if b'=' in x)
values = (b'57600',b'57600',b'58200') if sys.argv[3] == '16h' else (b'28800',b'28800',b'29400')
for key,value in zip((b'H3_TP2_REQUEST_TIMEOUT_S',b'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT',b'VLLM_OMNI_VIDEO_SYNC_TIMEOUT'),values):
    if env.get(key) != value: raise SystemExit('unexpected server timeout')
if env.get(b'H3_TP2_VAE_BLEND_TRACE',b'0') == b'1': raise SystemExit('experimental trace unexpectedly enabled')
PY
}
check_head_timeouts "$expected_timeouts"

mark 'H3-ASPECT-PATCH-HEAD-READY fc14580'

run_case() {
  local name=$1 spec=$2 tag=$3 limit=$4
  shift 4
  local out="$results/$name"
  [ ! -e "$out" ] || return 1
  wait_idle
  mark "H3-ASPECT-START $name"
  OUT="$out" timeout --signal=TERM --kill-after=30s "$limit" bash "$bench" "$spec" "$@" > "$results/$name.log" 2>&1
  valid_case "$out" "$tag"
  mark "H3-ASPECT-OK $name"
}
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL EXTRA_JSON || true
export TASK=t2va PROMPT='A calm landscape, one steady camera, soft natural ambience.'
run_case aspect-patch-smoke5 5:960:576:10:0 d5_960x576_s10_seed0 3600
export TASK=fl2va
export PROMPT='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, their shadows glide over the plain, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road walk on. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'
export EXTRA_JSON='{"long_video":true,"long_video_mode":"continuation","duration_seconds":30,"target":{"duration_seconds":16}}'
started=$(date '+%F %T')
run_case aspect-patch-duration16 16:960:544:20:1101 d16_960x544_s20_seed1101_fl2va 7500 "$results/koninck.png"
python3 - "$results/aspect-patch-duration16/d16_960x544_s20_seed1101_fl2va.mp4.timing.json" <<'PY'
import json,sys
with open(sys.argv[1]) as f: timing = json.load(f)
assert timing['effective_duration'] == 16 and timing['extra_params']['duration_seconds'] == 30
assert timing['extra_params']['target']['duration_seconds'] == 16
PY
journalctl --user -u "$head" --since "$started" --no-pager | grep -q 'continuation prompts generated for 2 windows (16.0 s)'
mark 'H3-ASPECT-PATCH-DONE fc14580'
