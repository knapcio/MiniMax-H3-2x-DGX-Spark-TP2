#!/usr/bin/env bash
# Continue the 2026-10-31 control: wait for its verified 16 s result, then run
# a matched 60 s control and, only if all guards pass, one experimental 300 s.
# Uses a new event log and output names; never replays the canceled 2026-10-30 queue.
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
head=h3-tp2-head.service
cd "$results"
exec 9>"$results/quality-resume-20261031.lock"
flock -n 9 || { echo 'quality queue already running'; exit 2; }

mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a resume-events-20261031.log; }
on_exit() {
  rc=$?
  if ((rc)); then mark "H3-RESUME-HOLD quality-20261031 rc=$rc" || true; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

paint="$results/koninck.png"
[ -s "$paint" ]
image_sha=$(sha256sum "$paint" | awk '{print $1}')
verified() { [ -s "$1/$2.mp4" ] && grep -Fxq "$2 ok" "$1/STATUS"; }
gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -q '^H3-RESUME-HOLD quality16-960-resume-20261031 ' resume-events-20261031.log 2>/dev/null; then
    echo '16 s control held; refusing to start the follow-up'; exit 1
  fi
  if grep -q '^H3-RESUME-OK quality16-960-resume-20261031 ' resume-events-20261031.log 2>/dev/null && \
     verified quality16-960-resume-20261031 d16_960x576_s20_seed1101_fl2va; then
    gate=1; break
  fi
  sleep "$sleep_s"
done
((gate)) || { echo '16 s control did not finish with a verified video'; exit 1; }

health() { curl -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18093/health || true; }
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
old_clouds='The clouds drift slowly across the sky'
new_clouds='The clouds drift slowly across the sky, keeping their broad soft painted shapes'
if [[ "$original" != *"$old_figures"* || "$original" != *"$old_shadows"* || "$original" != *"$old_clouds"* ]]; then
  echo 'expected prompt clauses are missing'; exit 1
fi
slow_figures=${original/"$old_figures"/"$new_figures"}
soft_light=${slow_figures/"$old_shadows"/"$new_shadows"}
combined=${soft_light/"$old_clouds"/"$new_clouds"}
[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ] || { echo 'source image changed while waiting'; exit 1; }
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL || true
export TASK=fl2va

verify_timing() {
  python3 - "$1" "$2" "$combined" "$paint" <<'PY'
import json, sys
from pathlib import Path
x = json.loads(Path(sys.argv[1]).read_text())
if x['http'] not in (200, '200') or x['curl_rc'] != 0:
    raise SystemExit('request did not complete successfully')
if (x['duration'], x['size'], x['steps'], x['seed'], x['task']) != (int(sys.argv[2]), '960x576', 20, 1101, 'fl2va'):
    raise SystemExit('control shape or task differs')
if x['prompt'] != sys.argv[3] or x['image'] != sys.argv[4]:
    raise SystemExit('control prompt or source image differs')
extra = x['extra_params']
if extra.get('long_video') is not True or extra.get('long_video_mode') != 'continuation' or extra.get('preencode_mp4') is not True:
    raise SystemExit('control did not use the expected streaming settings')
PY
}

run_case() {
  local name=$1 duration=$2 width=$3 height=$4 window=$5 prompt=$6 extra=$7 limit=$8
  local steps=${9:-20}
  local out="$results/$name" tag="d${duration}_${width}x${height}_s${steps}_seed1101_fl2va" start
  [ ! -e "$out" ] || { echo "refusing to overwrite $out"; return 1; }
  [ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ] || { echo 'source image changed'; return 1; }
  wait_idle || return 1
  no_jobs || { echo 'async jobs exist or cannot be listed'; return 1; }
  start=$(date '+%F %T')
  echo "starting $name at $start"
  mark "H3-RESUME-START $name"
  export PROMPT="$prompt" EXTRA_JSON="$extra"
  OUT="$out" timeout --signal=TERM --kill-after=30s "$limit" bash "$bench" \
    "$duration:$width:$height:$steps:1101" "$paint" > "$results/$name.log" 2>&1 || return 1
  verified "$out" "$tag" || { echo "$name has no verified result"; return 1; }
  verify_timing "$out/$tag.mp4.timing.json" "$duration" || return 1
  [ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ] || { echo 'source image changed during render'; return 1; }
  printf '%s\n' "$start" > "$out/started.txt"
  PY="$metrics_python" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics_script" \
    "$out/$tag.mp4" "$window" 22 > "$out/metrics.txt" 2>&1 || { echo "$name metrics failed"; return 1; }
  journalctl --user -u "$head" -a --since "$start" --no-pager -o short | \
    grep -oE 'phase diffuse [0-9.]+ s|seam check window [^\"]+' > "$out/phases.txt" || true
  mark "H3-RESUME-OK $name"
}

stream='{"long_video":true,"long_video_mode":"continuation","preencode_mp4":true}'
verify_timing "$results/quality16-960-resume-20261031/d16_960x576_s20_seed1101_fl2va.mp4.timing.json" 16
run_case quality60-960-resume-20261031 60 960 576 277 "$combined" "$stream" 15000

# A 300 s exploratory run has no per-window checkpoint. Refuse it unless the
# 960 controls passed media checks, have no hard cuts, the 16 s control detects
# no repeated global flicker, memory stays >=8 GiB on each host, and the slowest
# measured 60 s window projects under 8 h with a 35% margin + 30 min decode.
case960=quality16-960-resume-20261031
tag960=d16_960x576_s20_seed1101_fl2va
case60=quality60-960-resume-20261031
tag60_960=d60_960x576_s20_seed1101_fl2va
# The live 0ed86ba clip-metrics.sh predates the luma detector. Run a pinned
# standalone copy explicitly; never infer a clean result from an absent line.
if [[ ! -f "$flicker_script" ]] || \
   [[ "$(sha256sum "$flicker_script" | awk '{print $1}')" != "$flicker_sha" ]]; then
  mark 'H3-RESUME-SKIP-300 standalone flicker detector unavailable or failed'; exit 0
fi
for clip in "$case960/$tag960.mp4" "$case60/$tag60_960.mp4"; do
  if ! timeout --signal=TERM --kill-after=30s 3600 "$metrics_python" "$flicker_script" \
       "$clip" --minimum 2 > "$(dirname "$clip")/luma-flicker.txt" 2>&1; then
    mark 'H3-RESUME-SKIP-300 standalone flicker detector unavailable or failed'; exit 0
  fi
done
if ! grep -Fqx 'cuts (scene score > 0.1): none' "$case960/metrics.txt" || \
   ! grep -Fqx 'cuts (scene score > 0.1): none' "$case60/metrics.txt" || \
   ! grep -q '^periodic luma drops: none .* for 2 successive 17-frame periods ' "$case960/luma-flicker.txt" || \
   [ ! -s "$case960/$tag960.mem-head.csv" ] || [ ! -s "$case960/$tag960.mem-worker.csv" ] || \
   [ ! -s "$case60/$tag60_960.mem-head.csv" ] || [ ! -s "$case60/$tag60_960.mem-worker.csv" ]; then
  mark 'H3-RESUME-SKIP-300 960 control quality or memory samples did not pass'; exit 0
fi
for f in "$case960/$tag960.mem-head.csv" "$case960/$tag960.mem-worker.csv" \
         "$case60/$tag60_960.mem-head.csv" "$case60/$tag60_960.mem-worker.csv"; do
  if ! awk -F, 'NF >= 2 && $2 ~ /^[0-9]+$/ {n++; if ($2 < 8192) low=1} END {exit !(n >= 2 && !low)}' "$f"; then
    mark 'H3-RESUME-SKIP-300 960 control had insufficient memory margin'; exit 0
  fi
done
slowest_phase=$(grep -oE 'phase diffuse [0-9.]+ s' "$case60/phases.txt" | awk 'BEGIN {max=0} {if ($3>max) max=$3} END {if (max>0) print max}' || true)
if [[ ! "$slowest_phase" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  mark 'H3-RESUME-SKIP-300 960 long-window time unavailable'; exit 0
fi
pid=$(systemctl --user show "$head" -p MainPID --value)
client_timeout=${H3_QUALITY_CLIENT_TIMEOUT_S:-$(bash -c '. "$HOME/h3-tp2/scripts/env.sh"; printf "%s" "${H3_RENDER_TIMEOUT_S:-14400}"')}
if ! python3 - "$proc_root" "$pid" "$slowest_phase" "$client_timeout" <<'PY'
from pathlib import Path
import sys
root, pid, phase, client = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
env = dict(item.split(b'=', 1) for item in (root / str(pid) / 'environ').read_bytes().split(b'\0') if b'=' in item)
names = (b'H3_TP2_REQUEST_TIMEOUT_S', b'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT')
limits = [float(env.get(name, b'0')) for name in names]
api = float(env.get(b'VLLM_OMNI_VIDEO_SYNC_TIMEOUT', b'0'))
# 7213 frames = 28 full 277-frame windows and one shorter window. The
# slowest 60 s window time substitutes for every window conservatively.
projected = 29 * phase * 1.35 + 1800
print(f'300s projected guard: {projected:.0f}s, executor/wave {limits}, API {api:.0f}s, client {client:.0f}s')
if min(limits) < 28800 or api <= min(limits) or client <= api or projected >= min(limits) * 0.95:
    raise SystemExit(1)
PY
then
    mark 'H3-RESUME-SKIP-300 timeout or speed margin insufficient'; exit 0
fi
run_case quality300-960-resume-20261031 300 960 576 277 "$combined" "$stream" 30100
mark 'H3-RESUME-QUALITY-DONE'
