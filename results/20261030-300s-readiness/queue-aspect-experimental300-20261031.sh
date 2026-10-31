#!/usr/bin/env bash
# A structural 300 s experiment after the trace and direct 10/15 s controls.
# The 16 s control has detected flicker, so this is NOT a quality-gate pass or
# a production setting. It asks whether the requested duration can complete;
# it preserves the raw output and measures its defects for follow-up.
set -Eeuo pipefail
results=${H3_EXP_RESULTS:-"$HOME/h3-tp2-results"}
live=${H3_EXP_LIVE:-"$HOME/h3-tp2"}
stage=${H3_EXP_STAGE:-"$HOME/h3-tp2-next-fc14580"}
backup=${H3_EXP_BACKUP:-"$HOME/h3-tp2-backup-fc14580"}
txn=${H3_EXP_TXN:-"$results/file-txn-fc14580.py"}
bench=${H3_EXP_BENCH:-"$live/scripts/bench-matrix.sh"}
metrics=${H3_EXP_METRICS:-"$live/scripts/clip-metrics.sh"}
py=${H3_EXP_PYTHON:-"$HOME/h3-long/.venv/bin/python"}
flicker=${H3_EXP_FLICKER:-"$results/luma-flicker-20261030.py"}
worker=${H3_EXP_WORKER:-10.100.96.5}
meminfo=${H3_EXP_MEMINFO:-/proc/meminfo}
proc_root=${H3_EXP_PROC_ROOT:-/proc}
wait_attempts=${H3_EXP_WAIT_ATTEMPTS:-5760}
idle_attempts=${H3_EXP_IDLE_ATTEMPTS:-720}
wait_s=${H3_EXP_WAIT_S:-60}
idle_s=${H3_EXP_IDLE_S:-10}
min_disk_kb=${H3_EXP_MIN_DISK_KB:-16777216}
case16=quality16-960x544-20261031
case60=quality60-960x544-20261031
name=experiment300-960x544-20261031
tag16=d16_960x544_s20_seed1101_fl2va
tag60=d60_960x544_s20_seed1101_fl2va
tag300=d300_960x544_s20_seed1101_fl2va
cd "$results"
exec 9>"$results/experimental300-aspect-20261031.lock"
flock -n 9 || { echo 'experimental 300 s queue already running'; exit 2; }
phase=waiting
mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a aspect-events-20261031.log; }
on_exit() {
  local rc=$?
  if ((rc)); then mark "H3-ASPECT-HOLD experiment300-20261031 phase=$phase rc=$rc" || true; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
valid() { [ -s "$1/$2.mp4" ] && grep -Fxq "$2 ok" "$1/STATUS"; }
gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -qE '^H3-ASPECT-HOLD (quality-20261031|long300-20261031|prompt-patch-fc14580|vae-trace|short-20261031) ' aspect-events-20261031.log 2>/dev/null; then
    echo 'a predecessor held; no 300 s experiment'; exit 1
  fi
  if grep -q '^H3-ASPECT-SHORT-DONE ' aspect-events-20261031.log 2>/dev/null && \
     valid quality10-direct-960x544-20261031 d10_960x544_s20_seed1101_fl2va && \
     valid quality15-direct-960x544-20261031 d15_960x544_s20_seed1101_fl2va && \
     flock -n "$results/short-aspect-20261031.lock" true; then gate=1; break; fi
  sleep "$wait_s"
done
((gate)) || { echo 'direct controls did not finish and release their lock'; exit 1; }
phase=preflight
for path in "$txn" "$live" "$stage" "$backup"; do
  [[ "$path" =~ ^/[a-zA-Z0-9._/-]+$ ]] || { echo 'unsupported path characters'; exit 1; }
done
[[ "$(sha256sum "$txn" | awk '{print $1}')" = b2904bd67bfa3985d2eedd8fa4974a97e4d0a41f27cb56dbe32fe726792dd2c1 ]]
[[ "$(sha256sum "$flicker" | awk '{print $1}')" = 15f6267530a8e6051a2f5d27400798adc99d0fe8fe1f115cfd9319d2934c5a63 ]]
[[ "$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "sha256sum '$txn'" | awk '{print $1}')" = b2904bd67bfa3985d2eedd8fa4974a97e4d0a41f27cb56dbe32fe726792dd2c1 ]]
python3 "$txn" verify-new --live "$live" --stage "$stage" --backup "$backup"
ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "python3 '$txn' verify-new --live '$live' --stage '$stage' --backup '$backup'"
valid "$case16" "$tag16"
valid "$case60" "$tag60"
grep -q '^H3-ASPECT-OK quality16-960x544-20261031 ' aspect-events-20261031.log
grep -q '^H3-ASPECT-OK quality60-960x544-20261031 ' aspect-events-20261031.log
[[ ! -e "$results/$name" && ! -L "$results/$name" && -s "$results/koninck.png" ]] || { echo 'output exists or painting missing'; exit 1; }
for folder in "$case16" "$case60"; do
  grep -Fqx 'cuts (scene score > 0.1): none' "$folder/metrics.txt"
  [[ -s "$folder/luma-flicker.txt" ]] || { echo 'control flicker report unavailable'; exit 1; }
done
for f in "$case16/$tag16.mem-head.csv" "$case16/$tag16.mem-worker.csv" \
         "$case60/$tag60.mem-head.csv" "$case60/$tag60.mem-worker.csv"; do
  awk -F, 'NF >= 2 && $2 ~ /^[0-9]+$/ {n++; if ($2 < 8192) low=1} END {exit !(n >= 2 && !low)}' "$f"
done
paint="$results/koninck.png"
image_sha=$(sha256sum "$paint" | awk '{print $1}')
grep -Fxq "$image_sha" "$case16/source.sha256"
grep -Fxq "$image_sha" "$case60/source.sha256"
export PROMPT='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, keeping their broad soft painted shapes, the light on the landscape stays soft and nearly constant, without strong moving cloud shadows, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road move very slowly, taking small natural steps without sudden movements. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'
python3 - "$case16/$tag16.mp4.timing.json" "$case60/$tag60.mp4.timing.json" "$PROMPT" "$paint" <<'PY'
import json,sys
from pathlib import Path
for path,duration in ((sys.argv[1],16),(sys.argv[2],60)):
    x=json.loads(Path(path).read_text())
    if x['http'] not in (200,'200') or x['curl_rc'] != 0: raise SystemExit('control request failed')
    if (x['duration'],x['size'],x['steps'],x['seed'],x['task']) != (duration,'960x544',20,1101,'fl2va'):
        raise SystemExit('control settings differ')
    if x['prompt'] != sys.argv[3] or x['image'] != sys.argv[4]: raise SystemExit('source or prompt differ')
    e=x['extra_params']
    if e.get('long_video') is not True or e.get('long_video_mode') != 'continuation' or e.get('preencode_mp4') is not True:
        raise SystemExit('control was not streamed continuation')
PY
slowest=$(grep -oE 'phase diffuse [0-9.]+ s' "$case60/phases.txt" | awk 'BEGIN {max=0} {n++; if ($3>max) max=$3} END {if (n>=6 && max>0) print max}' || true)
[[ "$slowest" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'six measured 60 s window timings required'; exit 1; }
client_timeout=${H3_EXP_CLIENT_TIMEOUT_S:-$(bash -c '. "$HOME/h3-tp2/scripts/env.sh"; printf "%s" "${H3_RENDER_TIMEOUT_S:-14400}"')}
check_budget() {
  local pid
  pid=$(systemctl --user show h3-tp2-head.service -p MainPID --value)
  python3 - "$proc_root" "$pid" "$slowest" "$client_timeout" <<'PY'
from pathlib import Path
import sys
env=dict(x.split(b'=',1) for x in (Path(sys.argv[1])/sys.argv[2]/'environ').read_bytes().split(b'\0') if b'=' in x)
limits=[float(env.get(x,b'0')) for x in (b'H3_TP2_REQUEST_TIMEOUT_S',b'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT')]
api=float(env.get(b'VLLM_OMNI_VIDEO_SYNC_TIMEOUT',b'0')); client=float(sys.argv[4]); phase=float(sys.argv[3])
projected=29*phase*1.35+1800
print(f'300 s conservative projection {projected:.0f}s; executor/wave {limits}, API {api:.0f}s, client {client:.0f}s')
if (env.get(b'H3_TP2_VAE_BLEND_TRACE',b'0') == b'1' or min(limits)<28800 or api<=min(limits)
        or client<=api or projected>=min(limits)*.95): raise SystemExit(1)
PY
}
if ! check_budget; then mark 'H3-ASPECT-EXPERIMENT300-SKIPPED timeout speed or tracing preflight'; exit 0; fi
disk=$(df -Pk "$results" | awk 'NR==2 {print $4}')
[[ "$disk" =~ ^[0-9]+$ ]] && ((disk >= min_disk_kb)) || { echo 'insufficient verified disk space'; exit 1; }
health() { curl -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18093/health || true; }
no_jobs() { curl -fsS -m 5 'http://127.0.0.1:18093/v1/videos?limit=1' | python3 -c 'import json,sys; x=json.load(sys.stdin); sys.exit(0 if x.get("data")==[] and x.get("has_more") is False else 1)'; }
idle_sample() {
  local m gpu0 gpu1 mem0 mem1
  [ "$(health)" = 200 ] || return 1
  systemctl --user is-active --quiet h3-tp2-head.service || return 1
  systemctl --user is-active --quiet h3-tp2-memguard-head.service || return 1
  ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" 'systemctl --user is-active --quiet h3-tp2-agent.service && systemctl --user is-active --quiet h3-tp2-memguard-worker.service' || return 1
  m=$(curl -fsS -m 5 http://127.0.0.1:18093/metrics) || return 1
  printf '%s\n' "$m" | awk '/^vllm_omni:num_requests_running\{/ {r++; if ($NF != 0) bad=1} /^vllm_omni:num_requests_waiting\{/ {w++; if ($NF != 0) bad=1} END {exit !(r>0 && w>0 && !bad)}' || return 1
  gpu0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
  gpu1=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
  mem0=$(awk '/^MemAvailable:/ {print $2}' "$meminfo") || return 1
  mem1=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "awk '/^MemAvailable:/ {print \$2}' /proc/meminfo") || return 1
  [[ "$gpu0" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ && "$gpu1" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ && "$mem0" =~ ^[0-9]+$ && "$mem1" =~ ^[0-9]+$ ]] || return 1
  ((gpu0 < 10 && gpu1 < 10 && mem0 >= 12582912 && mem1 >= 12582912))
}
consecutive=0
for ((j=0; j<idle_attempts; j++)); do
  if idle_sample; then consecutive=$((consecutive+1)); else consecutive=0; fi
  ((consecutive >= 3)) && break
  sleep "$idle_s"
done
((consecutive >= 3)) && no_jobs || { echo 'pair is busy or not fully guarded'; exit 1; }
check_budget || { echo 'head timeout or tracing state changed while waiting'; exit 1; }
python3 "$txn" verify-new --live "$live" --stage "$stage" --backup "$backup"
ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "python3 '$txn' verify-new --live '$live' --stage '$stage' --backup '$backup'"
[[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]] || { echo 'source image changed'; exit 1; }
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL || true
export TASK=fl2va EXTRA_JSON='{"long_video":true,"long_video_mode":"continuation","preencode_mp4":true}'
start=$(date '+%F %T'); phase=rendering
mark 'H3-ASPECT-EXPERIMENT300-START known-control-flicker; structural-test-only'
mark "H3-ASPECT-START $name"
OUT="$results/$name" timeout --signal=TERM --kill-after=30s 30100 bash "$bench" \
  '300:960:544:20:1101' "$paint" > "$results/$name.log" 2>&1
valid "$name" "$tag300"
for f in "$name/$tag300.mem-head.csv" "$name/$tag300.mem-worker.csv"; do
  awk -F, 'NF >= 2 && $2 ~ /^[0-9]+$/ {n++} END {exit !(n >= 2)}' "$f"
done
python3 - "$name/$tag300.mp4.timing.json" "$PROMPT" "$paint" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text()); e=x['extra_params']
if x['http'] not in (200,'200') or x['curl_rc'] != 0: raise SystemExit('request failed')
if (x['duration'],x['size'],x['steps'],x['seed'],x['task']) != (300,'960x544',20,1101,'fl2va'): raise SystemExit('settings differ')
if x['prompt'] != sys.argv[2] or x['image'] != sys.argv[3]: raise SystemExit('source or prompt differ')
if e.get('long_video') is not True or e.get('long_video_mode') != 'continuation' or e.get('preencode_mp4') is not True: raise SystemExit('streaming differs')
PY
[[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]]
printf '%s\n' "$image_sha" > "$name/source.sha256"
printf '%s\n' "$start" > "$name/started.txt"
phase=metrics
PY="$py" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics" "$name/$tag300.mp4" 277 22 > "$name/metrics.txt" 2>&1
timeout --signal=TERM --kill-after=30s 3600 "$py" "$flicker" "$name/$tag300.mp4" --minimum 2 > "$name/luma-flicker.txt" 2>&1
journalctl --user -u h3-tp2-head.service -a --since "$start" --no-pager -o short | grep -oE 'phase diffuse [0-9.]+ s|seam check window [^\"]+' > "$name/phases.txt" || true
mark "H3-ASPECT-OK $name"
phase=done
mark 'H3-ASPECT-EXPERIMENT300-DONE structural-media-verification-only'
