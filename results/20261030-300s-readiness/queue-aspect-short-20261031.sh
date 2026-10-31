#!/usr/bin/env bash
# Direct 10/15 s FL2VA controls after the scoped VAE trace has restored the head.
# The first-frame image, seed and prompt match the aspect-corrected 16 s control.
set -Eeuo pipefail
results=${H3_SHORT_RESULTS:-"$HOME/h3-tp2-results"}
live=${H3_SHORT_LIVE:-"$HOME/h3-tp2"}
stage=${H3_SHORT_STAGE:-"$HOME/h3-tp2-next-fc14580"}
backup=${H3_SHORT_BACKUP:-"$HOME/h3-tp2-backup-fc14580"}
txn=${H3_SHORT_TXN:-"$results/file-txn-fc14580.py"}
bench=${H3_SHORT_BENCH:-"$live/scripts/bench-matrix.sh"}
metrics=${H3_SHORT_METRICS:-"$live/scripts/clip-metrics.sh"}
py=${H3_SHORT_PYTHON:-"$HOME/h3-long/.venv/bin/python"}
flicker=${H3_SHORT_FLICKER:-"$results/luma-flicker-20261030.py"}
worker=${H3_SHORT_WORKER:-10.100.96.5}
meminfo=${H3_SHORT_MEMINFO:-/proc/meminfo}
proc_root=${H3_SHORT_PROC_ROOT:-/proc}
wait_attempts=${H3_SHORT_WAIT_ATTEMPTS:-5760}
idle_attempts=${H3_SHORT_IDLE_ATTEMPTS:-720}
wait_s=${H3_SHORT_WAIT_S:-60}
idle_s=${H3_SHORT_IDLE_S:-10}
cd "$results"
exec 9>"$results/short-aspect-20261031.lock"
flock -n 9 || { echo 'short queue already running'; exit 2; }
phase=waiting
mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a aspect-events-20261031.log; }
on_exit() {
  local rc=$?
  if ((rc)); then mark "H3-ASPECT-HOLD short-20261031 phase=$phase rc=$rc" || true; fi
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
valid() { [ -s "$1/$2.mp4" ] && grep -Fxq "$2 ok" "$1/STATUS"; }
gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -qE '^H3-ASPECT-HOLD (quality-20261031|long300-20261031|prompt-patch-fc14580|vae-trace) ' aspect-events-20261031.log 2>/dev/null; then
    echo 'a predecessor held; no short tests'; exit 1
  fi
  if grep -q '^H3-ASPECT-TRACE-DONE ' aspect-events-20261031.log 2>/dev/null && \
     valid aspect-trace-smoke5 d5_960x576_s10_seed0 && \
     valid aspect-trace-duration16 d16_960x544_s20_seed1101_fl2va && \
     flock -n "$results/vae-trace-aspect-20261031.lock" true; then gate=1; break; fi
  sleep "$wait_s"
done
((gate)) || { echo 'VAE trace did not finish and restore'; exit 1; }

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
  gpu1=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
  mem0=$(awk '/^MemAvailable:/ {print $2}' "$meminfo") || return 1
  mem1=$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" "awk '/^MemAvailable:/ {print \$2}' /proc/meminfo") || return 1
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
pid=$(systemctl --user show h3-tp2-head.service -p MainPID --value)
python3 - "$proc_root" "$pid" <<'PY'
from pathlib import Path
import sys
env=dict(x.split(b'=',1) for x in (Path(sys.argv[1])/sys.argv[2]/'environ').read_bytes().split(b'\0') if b'=' in x)
if env.get(b'H3_TP2_VAE_BLEND_TRACE',b'0') == b'1': raise SystemExit('trace is still enabled')
PY
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL EXTRA_JSON || true
export TASK=fl2va
export PROMPT='The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, keeping their broad soft painted shapes, the light on the landscape stays soft and nearly constant, without strong moving cloud shadows, the river and the pond in the distance glint and ripple, leaves on the trees tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road move very slowly, taking small natural steps without sudden movements. The camera is locked on a tripod and does not move at all; one continuous take; soft natural ambience.'

run_case() {
  local duration=$1 name="quality${1}-direct-960x544-20261031" tag="d${1}_960x544_s20_seed1101_fl2va" start
  [[ ! -e "$results/$name" ]] || { echo 'refusing to overwrite output'; return 1; }
  [[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]] || { echo 'source image changed'; return 1; }
  wait_idle && no_jobs || return 1
  start=$(date '+%F %T'); phase="rendering-$duration"; mark "H3-ASPECT-START $name"
  OUT="$results/$name" timeout --signal=TERM --kill-after=30s 7500 bash "$bench" \
    "${duration}:960:544:20:1101" "$paint" > "$results/$name.log" 2>&1 || return 1
  valid "$results/$name" "$tag" || return 1
  python3 - "$results/$name/$tag.mp4.timing.json" "$duration" "$PROMPT" "$paint" <<'PY' || return 1
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
if x['http'] not in (200,'200') or x['curl_rc'] != 0: raise SystemExit('request failed')
if (x['duration'],x['size'],x['steps'],x['seed'],x['task']) != (int(sys.argv[2]),'960x544',20,1101,'fl2va'):
    raise SystemExit('unexpected direct control settings')
if x['prompt'] != sys.argv[3] or x['image'] != sys.argv[4] or x['extra_params'].get('long_video',False):
    raise SystemExit('prompt, source or direct mode differs')
PY
  [[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]] || return 1
  printf '%s\n' "$image_sha" > "$results/$name/source.sha256"
  PY="$py" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics" "$results/$name/$tag.mp4" 10000 0 > "$results/$name/metrics.txt" || return 1
  timeout --signal=TERM --kill-after=30s 3600 "$py" "$flicker" "$results/$name/$tag.mp4" --minimum 2 > "$results/$name/luma-flicker.txt" || return 1
  journalctl --user -u h3-tp2-head.service -a --since "$start" --no-pager -o short | \
    grep -oE 'phase diffuse [0-9.]+ s' > "$results/$name/phases.txt" || true
  printf '%s\n' "$start" > "$results/$name/started.txt"
  mark "H3-ASPECT-OK $name"
}
run_case 10
run_case 15
phase=done
mark 'H3-ASPECT-SHORT-DONE direct-10-15'
