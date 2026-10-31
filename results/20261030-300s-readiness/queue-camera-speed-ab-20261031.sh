#!/usr/bin/env bash
# Matched 30 s tests after the structural 300 s stage: default continuation
# wording vs explicit locked-camera wording, then 20 vs 10 steps with the same
# locked wording. Each is experimental; no service defaults are changed.
set -Eeuo pipefail
results=${H3_AB_RESULTS:-"$HOME/h3-tp2-results"}
live=${H3_AB_LIVE:-"$HOME/h3-tp2"}
stage=${H3_AB_STAGE:-"$HOME/h3-tp2-next-fc14580"}
backup=${H3_AB_BACKUP:-"$HOME/h3-tp2-backup-fc14580"}
txn=${H3_AB_TXN:-"$results/file-txn-fc14580.py"}
bench=${H3_AB_BENCH:-"$live/scripts/bench-matrix.sh"}
metrics=${H3_AB_METRICS:-"$live/scripts/clip-metrics.sh"}
py=${H3_AB_PYTHON:-"$HOME/h3-long/.venv/bin/python"}
flicker=${H3_AB_FLICKER:-"$results/luma-flicker-20261030.py"}
worker=${H3_AB_WORKER:-10.100.96.5}
proc_root=${H3_AB_PROC_ROOT:-/proc}
meminfo=${H3_AB_MEMINFO:-/proc/meminfo}
wait_attempts=${H3_AB_WAIT_ATTEMPTS:-5760}
idle_attempts=${H3_AB_IDLE_ATTEMPTS:-720}
wait_s=${H3_AB_WAIT_S:-60}
idle_s=${H3_AB_IDLE_S:-10}
min_disk_kb=${H3_AB_MIN_DISK_KB:-8388608}
default=quality30-camera-default-960x544-20261031
locked=quality30-camera-locked-960x544-20261031
fast=quality30-camera-locked-s10-960x544-20261031
cd "$results"
exec 9>"$results/camera-speed-ab-20261031.lock"
flock -n 9 || { echo 'camera/speed comparison already running'; exit 2; }
phase=waiting
mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a aspect-events-20261031.log; }
on_exit() { local rc=$?; if ((rc)); then mark "H3-ASPECT-HOLD camera-speed-ab-20261031 phase=$phase rc=$rc" || true; fi; }
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
valid() { [ -s "$1/$2.mp4" ] && grep -Fxq "$2 ok" "$1/STATUS"; }
gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -qE '^H3-ASPECT-HOLD (quality-20261031|long300-20261031|prompt-patch-fc14580|vae-trace|short-20261031|experiment300-20261031) ' aspect-events-20261031.log 2>/dev/null; then
    echo 'a predecessor held; no camera/speed comparison'; exit 1
  fi
  if grep -q '^H3-ASPECT-EXPERIMENT300-DONE ' aspect-events-20261031.log 2>/dev/null; then
    valid experiment300-960x544-20261031 d300_960x544_s20_seed1101_fl2va || { echo '300 s result lacks verification'; exit 1; }
    terminal=1
  elif grep -q '^H3-ASPECT-EXPERIMENT300-SKIPPED ' aspect-events-20261031.log 2>/dev/null; then terminal=1
  else terminal=0; fi
  if ((terminal)) && grep -q '^H3-ASPECT-SHORT-DONE ' aspect-events-20261031.log 2>/dev/null && \
     valid quality10-direct-960x544-20261031 d10_960x544_s20_seed1101_fl2va && \
     valid quality15-direct-960x544-20261031 d15_960x544_s20_seed1101_fl2va && \
     flock -n "$results/experimental300-aspect-20261031.lock" true; then gate=1; break; fi
  sleep "$wait_s"
done
((gate)) || { echo 'short or structural 300 s stage did not finish'; exit 1; }
phase=preflight
for path in "$txn" "$live" "$stage" "$backup"; do
  [[ "$path" =~ ^/[a-zA-Z0-9._/-]+$ ]] || { echo 'unsupported path characters'; exit 1; }
done
[[ "$(sha256sum "$txn" | awk '{print $1}')" = b2904bd67bfa3985d2eedd8fa4974a97e4d0a41f27cb56dbe32fe726792dd2c1 ]]
[[ "$(sha256sum "$flicker" | awk '{print $1}')" = 15f6267530a8e6051a2f5d27400798adc99d0fe8fe1f115cfd9319d2934c5a63 ]]
[[ "$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "sha256sum '$txn'" | awk '{print $1}')" = b2904bd67bfa3985d2eedd8fa4974a97e4d0a41f27cb56dbe32fe726792dd2c1 ]]
python3 "$txn" verify-new --live "$live" --stage "$stage" --backup "$backup"
ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "python3 '$txn' verify-new --live '$live' --stage '$stage' --backup '$backup'"
paint="$results/koninck.png"
[[ -s "$paint" ]] || { echo 'painting missing'; exit 1; }
image_sha=$(sha256sum "$paint" | awk '{print $1}')
grep -Fxq "$image_sha" "$results/quality16-960x544-20261031/source.sha256"
grep -Fxq "$image_sha" "$results/quality60-960x544-20261031/source.sha256"
for out in "$default" "$locked" "$fast"; do
  [[ ! -e "$results/$out" && ! -L "$results/$out" ]] || { echo 'comparison output exists'; exit 1; }
done
head_pid=$(systemctl --user show h3-tp2-head.service -p MainPID --value)
agent_pid=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" systemctl --user show h3-tp2-agent.service -p MainPID --value)
[[ "$head_pid" =~ ^[1-9][0-9]*$ && "$agent_pid" =~ ^[1-9][0-9]*$ ]] || { echo 'service PID unavailable'; exit 1; }
stable() {
  [[ "$(systemctl --user show h3-tp2-head.service -p MainPID --value)" = "$head_pid" ]] || return 1
  [[ "$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" systemctl --user show h3-tp2-agent.service -p MainPID --value)" = "$agent_pid" ]] || return 1
  python3 - "$proc_root" "$head_pid" <<'PY'
from pathlib import Path
import sys
env=dict(x.split(b'=',1) for x in (Path(sys.argv[1])/sys.argv[2]/'environ').read_bytes().split(b'\0') if b'=' in x)
if env.get(b'H3_TP2_VAE_BLEND_TRACE',b'0') == b'1': raise SystemExit('trace still enabled')
if env.get(b'H3_TP2_CONT_PROMPTS',b'auto').strip().lower() != b'auto': raise SystemExit('prompt mode differs')
if env.get(b'H3_TP2_CONT_PROMPTS_EXTRA',b'').strip() or env.get(b'H3_TP2_CONT_PROMPTS_REST',b'').strip():
    raise SystemExit('custom server prompt wording would confound comparison')
PY
}
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
wait_idle() {
  local consecutive=0
  for ((j=0; j<idle_attempts; j++)); do
    if idle_sample; then consecutive=$((consecutive+1)); else consecutive=0; fi
    ((consecutive >= 3)) && return 0
    sleep "$idle_s"
  done
  return 1
}
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL || true
export TASK=fl2va
export PROMPT='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, keeping their broad soft painted shapes, the light on the landscape stays soft and nearly constant, without strong moving cloud shadows, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road move very slowly, taking small natural steps without sudden movements. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'
stream='{"long_video":true,"long_video_mode":"continuation","preencode_mp4":true}'
locked_stream=$(python3 - "$PROMPT" <<'PY'
import json,sys
base=sys.argv[1].strip()
first=base.rstrip('. ')+'. One single continuous take from a steady camera, no cuts, no transitions.'
rest='The same continuous shot goes on without any cut or transition, the camera remains locked on a tripod: '+base
print(json.dumps({'long_video':True,'long_video_mode':'continuation','preencode_mp4':True,
                  'continuation_prompts':[first,rest,rest]}))
PY
)
run_case() {
  local name=$1 steps=$2 kind=$3 extra=$4 tag="d30_960x544_s${2}_seed1101_fl2va" start disk
  [[ ! -e "$results/$name" && ! -L "$results/$name" ]] || return 1
  [[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]] || return 1
  disk=$(df -Pk "$results" | awk 'NR==2 {print $4}') || return 1
  [[ "$disk" =~ ^[0-9]+$ ]] && ((disk >= min_disk_kb)) || return 1
  wait_idle && no_jobs && stable || return 1
  start=$(date '+%F %T'); phase="rendering-$kind-$steps"; mark "H3-ASPECT-START $name"
  export EXTRA_JSON="$extra"
  OUT="$results/$name" timeout --signal=TERM --kill-after=30s 12000 bash "$bench" \
    "30:960:544:${steps}:1101" "$paint" > "$results/$name.log" 2>&1 || return 1
  valid "$name" "$tag" && stable || return 1
  python3 - "$name/$tag.mp4.timing.json" "$steps" "$PROMPT" "$paint" "$extra" <<'PY' || return 1
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text()); expected=json.loads(sys.argv[5]); e=x['extra_params']
if x['http'] not in (200,'200') or x['curl_rc'] != 0: raise SystemExit('request failed')
if (x['duration'],x['size'],x['steps'],x['seed'],x['task']) != (30,'960x544',int(sys.argv[2]),1101,'fl2va'):
    raise SystemExit('settings differ')
if x['prompt'] != sys.argv[3] or x['image'] != sys.argv[4]: raise SystemExit('source or prompt differ')
if any(e.get(k)!=v for k,v in expected.items()): raise SystemExit('continuation parameters differ')
if 'continuation_prompts' not in expected and 'continuation_prompts' in e: raise SystemExit('default prompt was replaced')
PY
  [[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]] || return 1
  for f in "$name/$tag.mem-head.csv" "$name/$tag.mem-worker.csv"; do
    awk -F, 'NF >= 2 && $2 ~ /^[0-9]+$/ {n++} END {exit !(n >= 2)}' "$f" || return 1
  done
  printf '%s\n' "$image_sha" > "$name/source.sha256"
  printf '%s %s\n' "$head_pid" "$agent_pid" > "$name/service-pids.txt"
  printf '%s\n' "$start" > "$name/started.txt"
  PY="$py" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics" "$name/$tag.mp4" 277 22 > "$name/metrics.txt" 2>&1 || return 1
  timeout --signal=TERM --kill-after=30s 3600 "$py" "$flicker" "$name/$tag.mp4" --minimum 2 > "$name/luma-flicker.txt" 2>&1 || return 1
  journalctl --user -u h3-tp2-head.service -a --since "$start" --no-pager -o short | \
    grep -oE 'phase diffuse [0-9.]+ s|seam check window [^\"]+|step0 sha256 [^\"]+' > "$name/phases.txt" || true
  mark "H3-ASPECT-OK $name"
}
stable
mark 'H3-ASPECT-CAMERAAB-START prompt-and-step-comparisons'
run_case "$default" 20 default "$stream"
run_case "$locked" 20 locked "$locked_stream"
run_case "$fast" 10 locked "$locked_stream"
phase=done
mark 'H3-ASPECT-CAMERAAB-DONE three-30s-controls'
