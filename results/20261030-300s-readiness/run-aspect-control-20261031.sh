#!/usr/bin/env bash
# Run the 5 s or 16 s aspect-corrected control. The 16 s control waits for
# the verified 5 s result. No service restart or configuration change.
set -Eeuo pipefail
duration=${1:-}
[[ "$duration" = 5 || "$duration" = 16 ]] || { echo 'duration must be 5 or 16'; exit 2; }
results=${H3_SINGLE_RESULTS:-"$HOME/h3-tp2-results"}
bench=${H3_SINGLE_BENCH:-"$HOME/h3-tp2/scripts/bench-matrix.sh"}
metrics=${H3_SINGLE_METRICS:-"$HOME/h3-tp2/scripts/clip-metrics.sh"}
py=${H3_SINGLE_PYTHON:-"$HOME/h3-long/.venv/bin/python"}
flicker=${H3_SINGLE_FLICKER:-"$results/luma-flicker-20261030.py"}
worker=${H3_SINGLE_WORKER:-10.100.96.5}
meminfo=${H3_SINGLE_MEMINFO:-/proc/meminfo}
wait_attempts=${H3_SINGLE_WAIT_ATTEMPTS:-180}
idle_attempts=${H3_SINGLE_IDLE_ATTEMPTS:-720}
gate_attempts=${H3_SINGLE_GATE_ATTEMPTS:-720}
wait_s=${H3_SINGLE_WAIT_S:-10}
idle_s=${H3_SINGLE_IDLE_S:-10}
name="quality${duration}-960x544-20261031"
tag="d${duration}_960x544_s20_seed1101_fl2va"
paint="$results/koninck.png"
phase=waiting
cd "$results"
exec 9>"$results/single-aspect-${duration}-20261031.lock"
flock -n 9 || { echo 'control already running'; exit 2; }
mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a aspect-events-20261031.log; }
on_exit() {
  local rc=$?
  if ((rc)); then mark "H3-ASPECT-HOLD $name phase=$phase rc=$rc" || true; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
health() { curl -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18093/health || true; }
no_jobs() {
  curl -fsS -m 5 'http://127.0.0.1:18093/v1/videos?limit=1' | \
    python3 -c 'import json,sys; x=json.load(sys.stdin); sys.exit(0 if x.get("data") == [] and x.get("has_more") is False else 1)'
}
idle_sample() {
  local m gpu0 gpu1 mem0 mem1
  [ "$(health)" = 200 ] || return 1
  systemctl --user is-active --quiet h3-tp2-head.service || return 1
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
[[ ! -e "$results/$name" && -s "$paint" ]] || { echo 'output exists or painting missing'; exit 1; }
[[ "$(sha256sum "$flicker" | awk '{print $1}')" = 15f6267530a8e6051a2f5d27400798adc99d0fe8fe1f115cfd9319d2934c5a63 ]] || { echo 'flicker detector changed'; exit 1; }
image_sha=$(sha256sum "$paint" | awk '{print $1}')
if [ "$duration" = 16 ]; then
  gate=0
  for ((i=0; i<gate_attempts; i++)); do
    if grep -q '^H3-ASPECT-HOLD quality5-960x544-20261031 ' aspect-events-20261031.log 2>/dev/null; then
      echo '5 s aspect control held'; exit 1
    fi
    if grep -q '^H3-ASPECT-OK quality5-960x544-20261031 ' aspect-events-20261031.log 2>/dev/null && \
       [ -s quality5-960x544-20261031/d5_960x544_s20_seed1101_fl2va.mp4 ] && \
       grep -Fxq 'd5_960x544_s20_seed1101_fl2va ok' quality5-960x544-20261031/STATUS && \
       [ "$(cat quality5-960x544-20261031/source.sha256)" = "$image_sha" ]; then gate=1; break; fi
    sleep "$wait_s"
  done
  ((gate)) || { echo '5 s aspect control did not finish with matching source'; exit 1; }
fi
ready=0
for ((i=0; i<wait_attempts; i++)); do
  if [ "$(health)" = 200 ]; then ready=1; break; fi
  sleep "$wait_s"
done
((ready)) || { echo 'head did not become healthy'; exit 1; }
consecutive=0
for ((i=0; i<idle_attempts; i++)); do
  if idle_sample; then consecutive=$((consecutive+1)); else consecutive=0; fi
  ((consecutive >= 3)) && break
  sleep "$idle_s"
done
((consecutive >= 3)) || { echo 'pair did not become healthy and idle'; exit 1; }
no_jobs || { echo 'async jobs exist or cannot be listed'; exit 1; }
[[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]] || { echo 'painting changed'; exit 1; }
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL || true
export TASK=fl2va
export PROMPT='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, keeping their broad soft painted shapes, the light on the landscape stays soft and nearly constant, without strong moving cloud shadows, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road move very slowly, taking small natural steps without sudden movements. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'
export EXTRA_JSON='{"long_video":true,"long_video_mode":"continuation","preencode_mp4":true}'
start=$(date '+%F %T')
phase=rendering
mark "H3-ASPECT-START $name"
limit=7500
if [ "$duration" = 5 ]; then limit=3600; fi
OUT="$results/$name" timeout --signal=TERM --kill-after=30s "$limit" bash "$bench" \
  "${duration}:960:544:20:1101" "$paint" > "$results/$name.log" 2>&1
if [ ! -s "$results/$name/$tag.mp4" ] || ! grep -Fxq "$tag ok" "$results/$name/STATUS"; then
  echo 'render did not produce a verified successful output'
  exit 1
fi
python3 - "$results/$name/$tag.mp4.timing.json" "$duration" "$PROMPT" "$paint" <<'PY'
import json, sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
if x['http'] not in (200,'200') or x['curl_rc'] != 0:
    raise SystemExit('control request failed')
if (x['duration'],x['size'],x['steps'],x['seed'],x['task']) != (int(sys.argv[2]),'960x544',20,1101,'fl2va'):
    raise SystemExit('control shape or task differs')
if x['prompt'] != sys.argv[3] or x['image'] != sys.argv[4]:
    raise SystemExit('control prompt or source differs')
e=x['extra_params']
if e.get('long_video') is not True or e.get('long_video_mode') != 'continuation' or e.get('preencode_mp4') is not True:
    raise SystemExit('control streaming settings differ')
PY
[[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]] || { echo 'painting changed during render'; exit 1; }
printf '%s\n' "$image_sha" > "$results/$name/source.sha256"
phase=metrics
PY="$py" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics" \
  "$results/$name/$tag.mp4" 277 22 > "$results/$name/metrics.txt"
timeout --signal=TERM --kill-after=30s 3600 "$py" "$flicker" \
  "$results/$name/$tag.mp4" --minimum 2 > "$results/$name/luma-flicker.txt"
journalctl --user -u h3-tp2-head.service -a --since "$start" --no-pager -o short | \
  grep -oE 'phase diffuse [0-9.]+ s|seam check window [^"]+' > "$results/$name/phases.txt" || true
printf '%s\n' "$start" > "$results/$name/started.txt"
phase=done
mark "H3-ASPECT-OK $name"

