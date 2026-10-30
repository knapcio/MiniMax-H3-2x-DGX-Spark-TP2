#!/usr/bin/env bash
# If the 8 h 300 s attempt is skipped ONLY for time, try once at 960x576 after
# verified streaming controls and the prompt patch. Extend timeouts to 16 h via
# a hash-checked transaction. Never reboot a host.
set -Eeuo pipefail

results=${H3_LONG300_RESULTS:-"$HOME/h3-tp2-results"}
bench=${H3_LONG300_BENCH:-"$HOME/h3-tp2/scripts/bench-matrix.sh"}
metrics=${H3_LONG300_METRICS:-"$HOME/h3-tp2/scripts/clip-metrics.sh"}
py=${H3_LONG300_PYTHON:-"$HOME/h3-long/.venv/bin/python"}
flicker=${H3_LONG300_FLICKER:-"$results/luma-flicker-20261030.py"}
txn=${H3_LONG300_TXN:-"$results/timeout-env-txn-20261030.py"}
env_file=${H3_LONG300_ENV_FILE:-"$HOME/.h3-tp2.env"}
meta=${H3_LONG300_META:-"$results/timeout16h-transaction-20261030.json"}
proc_root=${H3_LONG300_PROC_ROOT:-/proc}
meminfo=${H3_LONG300_MEMINFO:-/proc/meminfo}
worker=${H3_LONG300_WORKER:-10.100.96.5}
wait_attempts=${H3_LONG300_WAIT_ATTEMPTS:-5760}  # 96 h
idle_attempts=${H3_LONG300_IDLE_ATTEMPTS:-720}   # 2 h
ready_attempts=${H3_LONG300_READY_ATTEMPTS:-100}
wait_s=${H3_LONG300_WAIT_S:-60}
idle_s=${H3_LONG300_IDLE_S:-10}
ready_s=${H3_LONG300_READY_S:-15}
head=h3-tp2-head.service
phase=waiting
needs_restore=0
restore_attempted=0
cd "$results"
exec 9>"$results/long300-20261030.lock"
flock -n 9 || { echo 'long300 queue already running'; exit 2; }

mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a gen.log; }
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
check_head_env() {
  local pid
  pid=$(systemctl --user show "$head" -p MainPID --value) || return 1
  python3 - "$proc_root" "$pid" "$1" <<'PY'
from pathlib import Path
import sys
env = dict(x.split(b'=', 1) for x in (Path(sys.argv[1]) / sys.argv[2] / 'environ').read_bytes().split(b'\0') if b'=' in x)
values = (b'28800', b'28800', b'29400') if sys.argv[3] == 'old' else (b'57600', b'57600', b'58200')
for key, value in zip((b'H3_TP2_REQUEST_TIMEOUT_S', b'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT', b'VLLM_OMNI_VIDEO_SYNC_TIMEOUT'), values):
    if env.get(key) != value:
        raise SystemExit('head has unexpected timeout')
if env.get(b'H3_TP2_VAE_BLEND_TRACE', b'0') == b'1':
    raise SystemExit('diagnostic trace unexpectedly enabled')
PY
}
txn_op() {
  local checksum
  checksum=$(sha256sum "$txn" | awk '{print $1}') || return 1
  [[ "$checksum" = ecec61db53f32f5bba18a3804cd6749f75b48890300f243ea72031b10966c009 ]] || return 1
  python3 "$txn" "$1" --env "$env_file" --meta "$meta"
}
restore_timeouts() {
  local state config
  mark 'H3-AUTO-LONG300-RESTORING timeouts'
  config=$(txn_op state) || return 1
  [[ "$config" = old || "$config" = new ]] || { echo 'restore: environment changed'; return 1; }
  state=$(systemctl --user show "$head" -p ActiveState --value) || return 1
  case "$state" in
    active)
      if [ "$config" = old ]; then check_head_env old; return $?; fi
      wait_idle && no_jobs || { echo 'restore: active head not verifiably idle or async jobs exist'; return 1; }
      systemctl --user stop "$head" || return 1
      ;;
    inactive|failed) systemctl --user stop "$head" || return 1;;
    *) echo "restore: head state $state is not safely stopped"; return 1;;
  esac
  ensure_stopped && post_stop_idle && ensure_stopped || return 1
  txn_op restore || return 1
  systemctl --user start "$head" && wait_ready && check_head_env old || return 1
  mark 'H3-AUTO-LONG300-RESTORED timeouts'
}
on_exit() {
  local rc=$? restore_rc=0
  trap - EXIT
  if ((needs_restore && !restore_attempted)); then
    restore_attempted=1
    restore_timeouts || restore_rc=$?
  fi
  if ((rc || restore_rc)); then
    mark "H3-AUTO-HOLD long300 phase=$phase rc=$rc restore_rc=$restore_rc" || true
  fi
  if ((rc)); then exit "$rc"; fi
  if ((restore_rc)); then exit "$restore_rc"; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -qE '^H3-AUTO-HOLD (quality-20261030|prompt-patch-fc14580) ' gen.log 2>/dev/null; then
    echo 'predecessor held; no long300 attempt'; exit 1
  fi
  if grep -q '^H3-AUTO-PATCH-DONE fc14580 ' gen.log 2>/dev/null && \
     valid_case patch-duration16 d16_1344x768_s20_seed1101_fl2va && \
     flock -n "$results/prompt-patch-fc14580.lock" true; then gate=1; break; fi
  sleep "$wait_s"
done
((gate)) || { echo 'patch did not finish'; exit 1; }

if grep -q '^H3-AUTO-QUALITY-DONE ' gen.log && \
   valid_case quality300-960-stream d300_960x576_s20_seed1101_fl2va; then
  mark 'H3-AUTO-LONG300-SKIPPED earlier 300 s render already verified'; exit 0
fi
if ! grep -q '^H3-AUTO-SKIP-300 timeout or speed margin insufficient ' gen.log; then
  mark 'H3-AUTO-LONG300-SKIPPED earlier quality or memory guard did not pass'; exit 0
fi

case960=quality16-960-stream
tag960=d16_960x576_s20_seed1101_fl2va
case60=quality60-960-stream
tag60=d60_960x576_s20_seed1101_fl2va
tag300=d300_960x576_s20_seed1101_fl2va
valid_case "$case960" "$tag960"
valid_case "$case60" "$tag60"
valid_case quality5-stream d5_1344x768_s20_seed1101_fl2va
valid_case quality16-stream d16_1344x768_s20_seed1101_fl2va
grep -Fqx 'cuts (scene score > 0.1): none' "$case960/metrics.txt"
grep -Fqx 'cuts (scene score > 0.1): none' "$case60/metrics.txt"
grep -q '^periodic luma drops: none .* for 2 successive 17-frame periods ' "$case960/luma-flicker.txt"
for f in "$case960/$tag960.mem-head.csv" "$case960/$tag960.mem-worker.csv" \
         "$case60/$tag60.mem-head.csv" "$case60/$tag60.mem-worker.csv"; do
  awk -F, 'NF >= 2 && $2 ~ /^[0-9]+$/ {n++; if ($2 < 8192) low=1} END {exit !(n >= 2 && !low)}' "$f"
done
slowest_phase=$(grep -oE 'phase diffuse [0-9.]+ s' "$case60/phases.txt" | awk 'BEGIN {max=0} {if ($3>max) max=$3} END {if (max>0) print max}' || true)
[[ "$slowest_phase" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo '60 s window phase unavailable'; exit 1; }
if ! python3 - "$slowest_phase" <<'PY'
import sys
phase=float(sys.argv[1])
projected=29*phase*1.35+1800
print(f'300s projected guard with 16h timeout: {projected:.0f}s')
if phase <= 0 or projected >= 57600*0.95:
    raise SystemExit(1)
PY
then
  mark 'H3-AUTO-LONG300-SKIPPED projection exceeds 16 h safety margin'; exit 0
fi

original='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, their shadows glide over the plain, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road walk on. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'
slow=${original/'the tiny figures on the road walk on'/'the tiny figures on the road move very slowly, taking small natural steps without sudden movements'}
combined=${slow/'their shadows glide over the plain'/'the light on the landscape stays soft and nearly constant, without strong moving cloud shadows'}
python3 - "$case960/$tag960.mp4.timing.json" "$case60/$tag60.mp4.timing.json" "$combined" "$results/koninck.png" <<'PY'
import json,sys
from pathlib import Path
for path, duration in ((sys.argv[1],16), (sys.argv[2],60)):
    x=json.loads(Path(path).read_text())
    if x['http'] not in (200, '200') or x['curl_rc'] != 0:
        raise SystemExit('control request did not complete successfully')
    if (x['duration'], x['size'], x['steps'], x['seed'], x['task']) != (duration, '960x576', 20, 1101, 'fl2va'):
        raise SystemExit('control shape or task changed')
    if x['prompt'] != sys.argv[3] or x['image'] != sys.argv[4]:
        raise SystemExit('control prompt or source image changed')
    extra=x['extra_params']
    if extra.get('long_video') is not True or extra.get('long_video_mode') != 'continuation' or extra.get('preencode_mp4') is not True:
        raise SystemExit('control did not use the expected streaming settings')
PY
[[ ! -e "$results/long300-960-stream" && ! -e "$meta" && -s "$results/koninck.png" ]] || { echo 'output or transaction exists, or image missing'; exit 1; }
[[ "$(sha256sum "$txn" | awk '{print $1}')" = ecec61db53f32f5bba18a3804cd6749f75b48890300f243ea72031b10966c009 ]] || { echo 'timeout transaction helper changed'; exit 1; }
[[ "$(sha256sum "$flicker" | awk '{print $1}')" = 15f6267530a8e6051a2f5d27400798adc99d0fe8fe1f115cfd9319d2934c5a63 ]] || { echo 'flicker detector changed'; exit 1; }
image_sha=$(sha256sum "$results/koninck.png" | awk '{print $1}')
wait_idle
no_jobs
check_head_env old
txn_op prepare
mark 'H3-AUTO-LONG300-START timeout16h'
phase=stopping
needs_restore=1
systemctl --user stop "$head"
phase=stopped
ensure_stopped || { echo 'head did not stop'; exit 1; }
post_stop_idle || { echo 'GPUs did not become idle'; exit 1; }
ensure_stopped || { echo 'head changed state after stop'; exit 1; }
txn_op apply
phase=starting
systemctl --user start "$head"
wait_ready
check_head_env new
phase=healthy
mark 'H3-AUTO-LONG300-HEAD-READY timeout16h'
wait_idle
no_jobs
[[ "$(sha256sum "$results/koninck.png" | awk '{print $1}')" = "$image_sha" ]] || { echo 'painting changed'; exit 1; }
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL || true
export TASK=fl2va PROMPT="$combined"
export EXTRA_JSON='{"long_video":true,"long_video_mode":"continuation","preencode_mp4":true}'
mark 'H3-AUTO-START long300-960-stream'
OUT="$results/long300-960-stream" timeout --signal=TERM --kill-after=30s 59000 bash "$bench" \
  '300:960:576:20:1101' "$results/koninck.png" > "$results/long300-960-stream.log" 2>&1
valid_case long300-960-stream "$tag300"
mark 'H3-AUTO-OK long300-960-stream'
PY="$py" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics" \
  "$results/long300-960-stream/$tag300.mp4" 277 22 > "$results/long300-960-stream/metrics.txt"
timeout --signal=TERM --kill-after=30s 3600 "$py" "$flicker" \
  "$results/long300-960-stream/$tag300.mp4" --minimum 2 > "$results/long300-960-stream/luma-flicker.txt"
needs_restore=0  # successful 300 s: retain the 16 h limits for future long requests
phase=done
mark 'H3-AUTO-LONG300-DONE timeout16h'

