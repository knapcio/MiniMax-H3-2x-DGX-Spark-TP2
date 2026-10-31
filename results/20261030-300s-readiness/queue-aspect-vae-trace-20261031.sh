#!/usr/bin/env bash
# Deferred, scoped diagnostic only. Wait for the prompt patch and the long-300
# decision, temporarily turn
# on its VAE blend trace for a smoke and the matched 16 s aspect-corrected case, then
# restore the normal head. Never touch the model or the persistent env file.
set -Eeuo pipefail

results=${H3_TRACE_RESULTS:-"$HOME/h3-tp2-results"}
live=${H3_TRACE_LIVE:-"$HOME/h3-tp2"}
stage=${H3_TRACE_STAGE:-"$HOME/h3-tp2-next-fc14580"}
backup=${H3_TRACE_BACKUP:-"$HOME/h3-tp2-backup-fc14580"}
txn=${H3_TRACE_TXN:-"$results/file-txn-fc14580.py"}
bench=${H3_TRACE_BENCH:-"$live/scripts/bench-matrix.sh"}
metrics=${H3_TRACE_METRICS:-"$live/scripts/clip-metrics.sh"}
metrics_python=${H3_TRACE_PYTHON:-"$HOME/h3-long/.venv/bin/python"}
flicker=${H3_TRACE_FLICKER:-"$results/luma-flicker-20261030.py"}
worker=${H3_TRACE_WORKER:-10.100.96.5}
env_file=${H3_TRACE_ENV_FILE:-"$HOME/.h3-tp2.env"}
proc_root=${H3_TRACE_PROC_ROOT:-/proc}
meminfo=${H3_TRACE_MEMINFO:-/proc/meminfo}
dropin=${H3_TRACE_DROPIN:-"${XDG_RUNTIME_DIR:?}/systemd/user/h3-tp2-head.service.d/zz-h3-aspect-vae-trace-20261031.conf"}
wait_attempts=${H3_TRACE_WAIT_ATTEMPTS:-5760}  # 96 h
idle_attempts=${H3_TRACE_IDLE_ATTEMPTS:-720}   # 2 h
ready_attempts=${H3_TRACE_READY_ATTEMPTS:-100}
wait_s=${H3_TRACE_WAIT_S:-60}
idle_s=${H3_TRACE_IDLE_S:-10}
ready_s=${H3_TRACE_READY_S:-15}
head=h3-tp2-head.service
phase=waiting
needs_restore=0
restore_attempted=0
cd "$results"
exec 9>"$results/vae-trace-aspect-20261031.lock"
flock -n 9 || { echo 'trace queue already running'; exit 2; }

mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a aspect-events-20261031.log; }
valid_case() { [ -s "$1/$2.mp4" ] && grep -Fxq "$2 ok" "$1/STATUS"; }
health() { curl -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18093/health || true; }
wait_ready() {
  for ((j=0; j<ready_attempts; j++)); do
    [ "$(health)" = 200 ] && return 0
    sleep "$ready_s"
  done
  return 1
}
no_jobs() {
  curl -fsS -m 5 'http://127.0.0.1:18093/v1/videos?limit=1' | \
    python3 -c 'import json,sys; x=json.load(sys.stdin); sys.exit(0 if x.get("data") == [] and x.get("has_more") is False else 1)'
}
idle_sample() {
  local m gpu0 gpu1 mem0 mem1
  [ "$(health)" = 200 ] || return 1
  systemctl --user is-active --quiet "$head" || return 1
  systemctl --user is-active --quiet h3-tp2-memguard-head.service || return 1
  ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" \
    'systemctl --user is-active --quiet h3-tp2-agent.service && systemctl --user is-active --quiet h3-tp2-memguard-worker.service' || return 1
  m=$(curl -fsS -m 5 http://127.0.0.1:18093/metrics) || return 1
  printf '%s\n' "$m" | awk '
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
  ((gpu0 < 10 && gpu1 < 10 && mem0 >= 12582912 && mem1 >= 12582912))
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
ensure_stopped() {
  local state pid
  state=$(systemctl --user show "$head" -p ActiveState --value) || return 1
  pid=$(systemctl --user show "$head" -p MainPID --value) || return 1
  [[ "$state" = inactive || "$state" = failed ]] && [[ "$pid" = 0 ]]
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
check_trace_env() {
  local pid
  pid=$(systemctl --user show "$head" -p MainPID --value) || return 1
  python3 - "$proc_root" "$pid" "$1" <<'PY'
from pathlib import Path
import sys
env = dict(x.split(b'=', 1) for x in (Path(sys.argv[1]) / sys.argv[2] / 'environ').read_bytes().split(b'\0') if b'=' in x)
value = env.get(b'H3_TP2_VAE_BLEND_TRACE', b'0')
if (value == b'1') != (sys.argv[3] == 'on'):
    raise SystemExit('unexpected VAE trace environment')
PY
}
check_own_dropin() {
  python3 - "$dropin" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if p.is_symlink() or p.read_bytes() != b'[Service]\nEnvironment=H3_TP2_VAE_BLEND_TRACE=1\n':
    raise SystemExit('trace drop-in changed')
PY
}
remove_own_dropin() {
  python3 - "$dropin" <<'PY'
import os, sys
from pathlib import Path
p = Path(sys.argv[1])
if p.is_symlink() or p.read_bytes() != b'[Service]\nEnvironment=H3_TP2_VAE_BLEND_TRACE=1\n':
    raise SystemExit('trace drop-in changed')
os.unlink(p)
PY
}
restore_original() {
  local state
  mark 'H3-ASPECT-TRACE-RESTORING vae-trace'
  state=$(systemctl --user show "$head" -p ActiveState --value) || return 1
  case "$state" in
    active)
      # If stop failed before the drop-in existed, do not stop an unchanged head.
      if [ ! -e "$dropin" ] && [ ! -L "$dropin" ]; then check_trace_env off; return $?; fi
      check_own_dropin || { echo 'restore: drop-in changed; leaving active head untouched'; return 1; }
      wait_idle && no_jobs || { echo 'restore: active head not verifiably idle or async jobs exist'; return 1; }
      check_own_dropin || { echo 'restore: drop-in changed; leaving active head untouched'; return 1; }
      systemctl --user stop "$head" || return 1
      ;;
    inactive|failed) systemctl --user stop "$head" || return 1;;
    *) echo "restore: head state $state is not safely stopped"; return 1;;
  esac
  ensure_stopped && post_stop_idle && ensure_stopped || return 1
  if [ -e "$dropin" ] || [ -L "$dropin" ]; then
    remove_own_dropin || return 1
    systemctl --user daemon-reload || return 1
  fi
  systemctl --user start "$head" && wait_ready && check_trace_env off || return 1
  mark 'H3-ASPECT-TRACE-RESTORED vae-trace'
}
on_exit() {
  local rc=$? restore_rc=0
  trap - EXIT
  if ((needs_restore && !restore_attempted)); then
    restore_attempted=1
    restore_original || restore_rc=$?
  fi
  if ((rc || restore_rc)); then
    mark "H3-ASPECT-HOLD vae-trace phase=$phase rc=$rc restore_rc=$restore_rc" || true
  fi
  if ((rc)); then exit "$rc"; fi
  if ((restore_rc)); then exit "$restore_rc"; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -qE '^H3-ASPECT-HOLD (quality-20261031|prompt-patch-fc14580|long300-20261031) ' aspect-events-20261031.log 2>/dev/null; then
    echo 'predecessor held; no trace'; exit 1
  fi
  if grep -q '^H3-ASPECT-PATCH-DONE fc14580 ' aspect-events-20261031.log 2>/dev/null && \
     grep -qE '^H3-ASPECT-LONG300-(DONE|SKIPPED) ' aspect-events-20261031.log 2>/dev/null && \
     valid_case aspect-patch-smoke5 d5_960x576_s10_seed0 && \
     valid_case aspect-patch-duration16 d16_960x544_s20_seed1101_fl2va && \
     flock -n "$results/prompt-patch-aspect-20261031.lock" true && \
     flock -n "$results/long300-aspect-20261031.lock" true; then gate=1; break; fi
  sleep "$wait_s"
done
((gate)) || { echo 'patch or long300 predecessor did not finish'; exit 1; }

# Check deployed source, detector, outputs and config before even stopping the head.
for path in "$txn" "$live" "$stage" "$backup"; do
  [[ "$path" =~ ^/[a-zA-Z0-9._/-]+$ ]] || { echo 'unsupported path characters'; exit 1; }
done
[[ "$(sha256sum "$txn" | awk '{print $1}')" = b2904bd67bfa3985d2eedd8fa4974a97e4d0a41f27cb56dbe32fe726792dd2c1 ]] || { echo 'transaction helper changed'; exit 1; }
[[ "$(sha256sum "$flicker" | awk '{print $1}')" = 15f6267530a8e6051a2f5d27400798adc99d0fe8fe1f115cfd9319d2934c5a63 ]] || { echo 'flicker detector changed'; exit 1; }
worker_sha=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "sha256sum '$txn'") || exit 1
[[ "${worker_sha%% *}" = b2904bd67bfa3985d2eedd8fa4974a97e4d0a41f27cb56dbe32fe726792dd2c1 ]] || { echo 'worker transaction helper changed'; exit 1; }
python3 "$txn" verify-new --live "$live" --stage "$stage" --backup "$backup"
ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" \
  "python3 '$txn' verify-new --live '$live' --stage '$stage' --backup '$backup'"
if [[ -e "$dropin" || -L "$dropin" || -L "$(dirname "$dropin")" ]]; then
  echo 'foreign trace drop-in or symlink'; exit 1
fi
[[ -r "$env_file" ]] || { echo 'environment unavailable'; exit 1; }
if grep -q 'H3_TP2_VAE_BLEND_TRACE' "$env_file"; then
  echo 'trace already configured'; exit 1
else
  grep_rc=$?
  ((grep_rc == 1)) || { echo 'cannot inspect the environment'; exit 1; }
fi
drop_paths=$(systemctl --user show "$head" -p DropInPaths --value) || exit 1
if [[ -n "$drop_paths" ]]; then echo 'head has existing drop-ins'; exit 1; fi
if [[ -e "$results/aspect-trace-smoke5" || -e "$results/aspect-trace-duration16" ]]; then
  echo 'trace output already exists'; exit 1
fi
[[ -s "$results/koninck.png" ]] || { echo 'painting is missing'; exit 1; }
image_sha=$(sha256sum "$results/koninck.png" | awk '{print $1}')
grep -Fxq "$image_sha" "$results/quality16-960x544-20261031/source.sha256" || { echo 'source differs from control'; exit 1; }
wait_idle
no_jobs
check_trace_env off
mark 'H3-ASPECT-TRACE-START vae-trace'
phase=stopping
needs_restore=1
systemctl --user stop "$head"
phase=stopped
ensure_stopped || { echo 'head did not stop'; exit 1; }
post_stop_idle || { echo 'GPUs did not become idle'; exit 1; }
ensure_stopped || { echo 'head changed state after stop'; exit 1; }
mkdir -p "$(dirname "$dropin")"
python3 - "$dropin" <<'PY'
import os, sys
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, 'wb') as f:
    f.write(b'[Service]\nEnvironment=H3_TP2_VAE_BLEND_TRACE=1\n')
    f.flush(); os.fsync(f.fileno())
PY
systemctl --user daemon-reload
phase=starting
systemctl --user start "$head"
wait_ready
check_trace_env on
phase=healthy
mark 'H3-ASPECT-TRACE-HEAD-READY vae-trace'

run_case() {
  local name=$1 spec=$2 tag=$3 limit=$4
  shift 4
  local out="$results/$name" start
  [[ ! -e "$out" ]] || return 1
  [[ "$(sha256sum "$results/koninck.png" | awk '{print $1}')" = "$image_sha" ]] || return 1
  wait_idle && no_jobs || return 1
  start=$(date '+%F %T')
  mark "H3-ASPECT-START $name"
  OUT="$out" timeout --signal=TERM --kill-after=30s "$limit" bash "$bench" "$spec" "$@" > "$results/$name.log" 2>&1 || return 1
  valid_case "$out" "$tag" || return 1
  journalctl --user -u "$head" -a --since "$start" --no-pager -o short | \
    grep -E 'h3_tp2 VAE temporal blend (trace skipped|tracing unavailable|[0-9]+ approx-frame)' > "$out/vae-trace.txt" || return 1
  grep -qE 'h3_tp2 VAE temporal blend [0-9]+ approx-frame .* old .* new .* blended ' "$out/vae-trace.txt" || return 1
  ! grep -qE 'trace skipped|tracing unavailable' "$out/vae-trace.txt" || return 1
  mark "H3-ASPECT-OK $name"
}
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL EXTRA_JSON || true
export TASK=t2va PROMPT='A calm landscape, one steady camera, soft natural ambience.'
run_case aspect-trace-smoke5 5:960:576:10:0 d5_960x576_s10_seed0 3600
for case in aspect-patch-smoke5 aspect-trace-smoke5; do
  ffmpeg -v error -xerror -i "$results/$case/d5_960x576_s10_seed0.mp4" -map 0:v:0 -an \
    -c:v rawvideo -f hash -hash sha256 - > "$results/aspect-trace-smoke5/$case-decoded-video.sha256"
done
if cmp -s aspect-trace-smoke5/aspect-patch-smoke5-decoded-video.sha256 aspect-trace-smoke5/aspect-trace-smoke5-decoded-video.sha256; then
  echo 'decoded videos match' > aspect-trace-smoke5/comparison.txt
else
  echo 'decoded videos differ; model nondeterminism can also cause differences' > aspect-trace-smoke5/comparison.txt
fi
export TASK=fl2va
export PROMPT='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, keeping their broad soft painted shapes, the light on the landscape stays soft and nearly constant, without strong moving cloud shadows, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road move very slowly, taking small natural steps without sudden movements. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'
export EXTRA_JSON='{"long_video":true,"long_video_mode":"continuation","preencode_mp4":true}'
run_case aspect-trace-duration16 16:960:544:20:1101 d16_960x544_s20_seed1101_fl2va 7500 "$results/koninck.png"
PY="$metrics_python" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics" \
  "$results/aspect-trace-duration16/d16_960x544_s20_seed1101_fl2va.mp4" 277 22 > "$results/aspect-trace-duration16/metrics.txt"
timeout --signal=TERM --kill-after=30s 3600 "$metrics_python" "$flicker" \
  "$results/aspect-trace-duration16/d16_960x544_s20_seed1101_fl2va.mp4" --minimum 2 > "$results/aspect-trace-duration16/luma-flicker.txt"
phase=restoring
restore_attempted=1
restore_original
needs_restore=0
phase=done
mark 'H3-ASPECT-TRACE-DONE vae-trace'
