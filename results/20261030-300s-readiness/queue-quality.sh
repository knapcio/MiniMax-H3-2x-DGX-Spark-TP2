#!/usr/bin/env bash
# Independent, guarded follow-up: no host reboot, service restart, or model/code changes.
# Waits for the pre-existing 60 s + calibration chain, then runs matched quality/speed
# controls. A 300 s 960x576 streaming experiment is allowed only by the checks below.
set -Eeuo pipefail

results=${H3_QUALITY_RESULTS:-"$HOME/h3-tp2-results"}
bench=${H3_QUALITY_BENCH:-"$HOME/h3-tp2/scripts/bench-matrix.sh"}
metrics_script=${H3_QUALITY_METRICS:-"$HOME/h3-tp2/scripts/clip-metrics.sh"}
metrics_python=${H3_QUALITY_PYTHON:-"$HOME/h3-long/.venv/bin/python"}
flicker_script=${H3_QUALITY_FLICKER:-"$results/luma-flicker-20261030.py"}
flicker_sha=15f6267530a8e6051a2f5d27400798adc99d0fe8fe1f115cfd9319d2934c5a63
worker=${H3_QUALITY_WORKER:-10.100.96.5}
sleep_s=${H3_QUALITY_SLEEP_S:-60}
idle_sleep_s=${H3_QUALITY_IDLE_SLEEP_S:-10}
wait_attempts=${H3_QUALITY_WAIT_ATTEMPTS:-2880}  # 48 h
idle_attempts=${H3_QUALITY_IDLE_ATTEMPTS:-720}   # 2 h
min_available_kb=${H3_QUALITY_MIN_AVAILABLE_KB:-12582912}  # 12 GiB, each host
meminfo=${H3_QUALITY_MEMINFO:-/proc/meminfo}
proc_root=${H3_QUALITY_PROC_ROOT:-/proc}
tag60=d60_1344x768_s20_seed1101_fl2va
head=h3-tp2-head.service
cd "$results"
exec 9>"$results/quality-20261030.lock"
flock -n 9 || { echo 'quality queue already running'; exit 2; }

mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a gen.log; }
on_exit() {
  rc=$?
  if ((rc)); then mark "H3-AUTO-HOLD quality-20261030 rc=$rc" || true; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

verified() { [ -s "$1/$2.mp4" ] && grep -Fxq "$2 ok" "$1/STATUS"; }
gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -q '^CALIB41-DONE ' gen.log 2>/dev/null && verified koninck60-fl2va "$tag60"; then
    gate=1; break
  fi
  if grep -q '^H3-RECOVERY-RETRY60-OK ' gen.log 2>/dev/null && verified koninck60-fl2va-retry "$tag60"; then
    gate=1; break
  fi
  if grep -qE '^(CALIB41-HOLD|H3-RECOVERY-HOLD) ' gen.log 2>/dev/null; then
    echo 'previous queue held; refusing to start the follow-up'; exit 1
  fi
  sleep "$sleep_s"
done
((gate)) || { echo 'predecessor did not finish with a verified 60 s video'; exit 1; }

health() { curl -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18093/health || true; }
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
  ((gpu0 < 10 && gpu1 < 10 && mem0 >= min_available_kb && mem1 >= min_available_kb))
}
wait_idle() {
  local consecutive=0
  for ((j=0; j<idle_attempts; j++)); do
    if idle_sample; then consecutive=$((consecutive+1)); else consecutive=0; fi
    ((consecutive >= 3)) && return 0
    sleep "$idle_sleep_s"
  done
  echo 'pair did not become healthy, guarded, idle and sufficiently free'
  return 1
}

original='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, their shadows glide over the plain, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road walk on. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'
old_figures='the tiny figures on the road walk on'
new_figures='the tiny figures on the road move very slowly, taking small natural steps without sudden movements'
old_shadows='their shadows glide over the plain'
new_shadows='the light on the landscape stays soft and nearly constant, without strong moving cloud shadows'
[[ "$original" == *"$old_figures"* && "$original" == *"$old_shadows"* ]]
slow_figures=${original/"$old_figures"/"$new_figures"}
combined=${slow_figures/"$old_shadows"/"$new_shadows"}
paint="$results/koninck.png"
[ -s "$paint" ]
image_sha=$(sha256sum "$paint" | awk '{print $1}')
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL || true
export TASK=fl2va

run_case() {
  local name=$1 duration=$2 width=$3 height=$4 window=$5 prompt=$6 extra=$7 limit=$8
  local out="$results/$name" tag="d${duration}_${width}x${height}_s20_seed1101_fl2va" start
  [ ! -e "$out" ] || { echo "refusing to overwrite $out"; return 1; }
  [ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ] || { echo 'source image changed'; return 1; }
  wait_idle || return 1
  start=$(date '+%F %T')
  echo "starting $name at $start"
  mark "H3-AUTO-START $name"
  export PROMPT="$prompt" EXTRA_JSON="$extra"
  OUT="$out" timeout --signal=TERM --kill-after=30s "$limit" bash "$bench" \
    "$duration:$width:$height:20:1101" "$paint" > "$results/$name.log" 2>&1 || return 1
  verified "$out" "$tag" || { echo "$name has no verified result"; return 1; }
  printf '%s\n' "$start" > "$out/started.txt"
  PY="$metrics_python" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics_script" \
    "$out/$tag.mp4" "$window" 22 > "$out/metrics.txt" 2>&1 || { echo "$name metrics failed"; return 1; }
  journalctl --user -u "$head" -a --since "$start" --no-pager -o short | \
    grep -oE 'phase diffuse [0-9.]+ s|seam check window [^\"]+' > "$out/phases.txt" || true
  mark "H3-AUTO-OK $name"
}

base='{"long_video":true,"long_video_mode":"continuation"}'
smaller='{"long_video":true,"long_video_mode":"continuation","continuation_window_frames":175}'
stream='{"long_video":true,"long_video_mode":"continuation","preencode_mp4":true}'
run_case quality16-base 16 1344 768 277 "$original" "$base" 7500
run_case quality16-win175 16 1344 768 175 "$original" "$smaller" 7500
run_case quality16-slowfig 16 1344 768 277 "$slow_figures" "$base" 7500
run_case quality5-stream 5 1344 768 277 "$original" "$stream" 3600
run_case quality16-stream 16 1344 768 277 "$original" "$stream" 7500
run_case quality16-960-stream 16 960 576 277 "$combined" "$stream" 7500

# A 300 s exploratory run has no per-window checkpoint. Refuse it unless the
# 960p control passed media checks, detects no hard cut/repeated global flicker,
# has memory samples with at least 8 GiB free on each host, and the first full window projects safely
# under the live 8 h executor/wave limits with a 35% margin + 30 min decode.
case960=quality16-960-stream
tag960=d16_960x576_s20_seed1101_fl2va
# The live 0ed86ba clip-metrics.sh predates the luma detector. Run a pinned
# standalone copy explicitly; never infer a clean result from an absent line.
if [[ ! -f "$flicker_script" ]] || \
   [[ "$(sha256sum "$flicker_script" | awk '{print $1}')" != "$flicker_sha" ]] || \
   ! timeout --signal=TERM --kill-after=30s 3600 "$metrics_python" "$flicker_script" \
       "$case960/$tag960.mp4" --minimum 2 > "$case960/luma-flicker.txt" 2>&1; then
  mark 'H3-AUTO-SKIP-300 standalone flicker detector unavailable or failed'; exit 0
fi
if ! grep -Fqx 'cuts (scene score > 0.1): none' "$case960/metrics.txt" || \
   ! grep -q '^periodic luma drops: none .* for 2 successive 17-frame periods ' "$case960/luma-flicker.txt" || \
   [ ! -s "$case960/$tag960.mem-head.csv" ] || [ ! -s "$case960/$tag960.mem-worker.csv" ]; then
  mark 'H3-AUTO-SKIP-300 960 control quality or memory samples did not pass'; exit 0
fi
for f in "$case960/$tag960.mem-head.csv" "$case960/$tag960.mem-worker.csv"; do
  if ! awk -F, 'NF >= 2 && $2 ~ /^[0-9]+$/ {n++; if ($2 < 8192) low=1} END {exit !(n >= 2 && !low)}' "$f"; then
    mark 'H3-AUTO-SKIP-300 960 control had insufficient memory margin'; exit 0
  fi
done
first_phase=$(grep -m1 -oE 'phase diffuse [0-9.]+ s' "$case960/phases.txt" | awk '{print $3}' || true)
if [[ ! "$first_phase" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  mark 'H3-AUTO-SKIP-300 960 first-window time unavailable'; exit 0
fi
pid=$(systemctl --user show "$head" -p MainPID --value)
if ! python3 - "$proc_root" "$pid" "$first_phase" <<'PY'
from pathlib import Path
import sys
root, pid, phase = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
env = dict(item.split(b'=', 1) for item in (root / str(pid) / 'environ').read_bytes().split(b'\0') if b'=' in item)
names = (b'H3_TP2_REQUEST_TIMEOUT_S', b'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT')
limits = [float(env.get(name, b'0')) for name in names]
api = float(env.get(b'VLLM_OMNI_VIDEO_SYNC_TIMEOUT', b'0'))
# 7213 frames = 28 full 277-frame windows and one shorter window. The
# first-window time substitutes for that shorter window conservatively.
projected = 29 * phase * 1.35 + 1800
print(f'300s projected guard: {projected:.0f}s, executor/wave {limits}, API {api:.0f}s')
if min(limits) < 28800 or api <= min(limits) or projected >= min(limits) * 0.95:
    raise SystemExit(1)
PY
then
  mark 'H3-AUTO-SKIP-300 timeout or speed margin insufficient'; exit 0
fi
run_case quality300-960-stream 300 960 576 277 "$combined" "$stream" 30100
mark 'H3-AUTO-QUALITY-DONE'
