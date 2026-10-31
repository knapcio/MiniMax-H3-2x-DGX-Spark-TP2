#!/usr/bin/env bash
# Matched 30 s cloud-light prompt experiment after the isolated retry follow-ups.
# It changes only one lighting clause relative to the locked-camera baseline.
# It does not establish physical cloud-shadow correspondence or promote a default.
set -Eeuo pipefail
results=${H3_CLOUD_RESULTS:-"$HOME/h3-tp2-results"}
live=${H3_CLOUD_LIVE:-"$HOME/h3-tp2"}
stage=${H3_CLOUD_STAGE:-"$HOME/h3-tp2-next-fc14580"}
backup=${H3_CLOUD_BACKUP:-"$HOME/h3-tp2-backup-fc14580"}
txn=${H3_CLOUD_TXN:-"$results/file-txn-fc14580.py"}
bench=${H3_CLOUD_BENCH:-"$live/scripts/bench-matrix.sh"}
metrics=${H3_CLOUD_METRICS:-"$live/scripts/clip-metrics.sh"}
py=${H3_CLOUD_PYTHON:-"$HOME/h3-long/.venv/bin/python"}
flicker=${H3_CLOUD_FLICKER:-"$results/luma-flicker-20261030.py"}
worker=${H3_CLOUD_WORKER:-10.100.96.5}
proc_root=${H3_CLOUD_PROC_ROOT:-/proc}
meminfo=${H3_CLOUD_MEMINFO:-/proc/meminfo}
wait_attempts=${H3_CLOUD_WAIT_ATTEMPTS:-5760}
idle_attempts=${H3_CLOUD_IDLE_ATTEMPTS:-720}
wait_s=${H3_CLOUD_WAIT_S:-60}
idle_s=${H3_CLOUD_IDLE_S:-10}
min_disk_kb=${H3_CLOUD_MIN_DISK_KB:-8388608}
baseline=quality30-camera-locked-960x544-20261031
name=quality30-cloud-light-960x544-20261031
tag=d30_960x544_s20_seed1101_fl2va
cd "$results"
exec 9>"$results/cloud-light-aspect-20261031.lock"
flock -n 9 || { echo 'cloud-light queue already running'; exit 2; }
phase=waiting
mark() { printf '%s %s\n' "$1" "$(date '+%F %T %Z')" | tee -a aspect-events-20261031.log; }
on_exit() { local rc=$?; if ((rc)); then mark "H3-ASPECT-HOLD cloud-light-20261031 phase=$phase rc=$rc" || true; fi; }
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
valid() { [ -s "$1/$2.mp4" ] && grep -Fxq "$2 ok" "$1/STATUS"; }
gate=0
for ((i=0; i<wait_attempts; i++)); do
  if grep -qE '^H3-ASPECT-HOLD |^H3-RETRY-FOLLOWUPS-HOLD ' aspect-events-20261031.log 2>/dev/null; then
    echo 'a predecessor held; no cloud-light test'; exit 1
  fi
  if grep -q '^H3-RETRY-FOLLOWUPS-DONE ' aspect-events-20261031.log 2>/dev/null && \
     grep -q '^H3-ASPECT-HUMANPACE-DONE ' aspect-events-20261031.log 2>/dev/null && \
     valid "$baseline" "$tag" && valid quality30-human-pace-960x544-20261031 "$tag" && \
     flock -n "$results/followups.lock" true; then gate=1; break; fi
  sleep "$wait_s"
done
((gate)) || { echo 'retry follow-ups did not finish'; exit 1; }
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
[[ -s "$paint" && ! -e "$results/$name" && ! -L "$results/$name" ]] || { echo 'painting missing or output exists'; exit 1; }
image_sha=$(sha256sum "$paint" | awk '{print $1}')
grep -Fxq "$image_sha" "$baseline/source.sha256"
grep -Fxq "$image_sha" "$results/quality30-human-pace-960x544-20261031/source.sha256"
grep -Fxq "$image_sha" "$results/quality16-960x544-20261031/source.sha256"
read -r head_pid agent_pid < "$baseline/service-pids.txt"
[[ "$head_pid" =~ ^[1-9][0-9]*$ && "$agent_pid" =~ ^[1-9][0-9]*$ ]] || { echo 'invalid baseline service PIDs'; exit 1; }
PROMPT=$(python3 - "$baseline/$tag.mp4.timing.json" "$paint" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text()); e=x['extra_params']; p=x['prompt'].strip()
if x['http'] not in (200,'200') or x['curl_rc'] != 0: raise SystemExit('baseline failed')
if (x['duration'],x['size'],x['steps'],x['seed'],x['task'],x['image']) != (30,'960x544',20,1101,'fl2va',sys.argv[2]): raise SystemExit('baseline settings differ')
if not p or e.get('long_video') is not True or e.get('long_video_mode') != 'continuation' or e.get('preencode_mp4') is not True: raise SystemExit('baseline mode differs')
q=e.get('continuation_prompts'); first=p.rstrip('. ')+'. One single continuous take from a steady camera, no cuts, no transitions.'
rest='The same continuous shot goes on without any cut or transition, the camera remains locked on a tripod: '+p
if q != [first,rest,rest] or e.get('continuation_window_frames',277) != 277: raise SystemExit('baseline camera prompts differ')
old='the light on the landscape stays soft and nearly constant, without strong moving cloud shadows'
new='the ground remains evenly lit by soft daylight, with only faint diffuse shadows moving slowly in step with the clouds'
if p.count(old)!=1: raise SystemExit('baseline cloud-light clause differs')
print(p.replace(old,new))
PY
)
EXTRA_JSON=$(python3 - "$baseline/$tag.mp4.timing.json" "$PROMPT" <<'PY'
import json,sys
from pathlib import Path
e=json.loads(Path(sys.argv[1]).read_text())['extra_params']; p=sys.argv[2]
first=p.rstrip('. ')+'. One single continuous take from a steady camera, no cuts, no transitions.'
rest='The same continuous shot goes on without any cut or transition, the camera remains locked on a tripod: '+p
e['continuation_prompts']=[first,rest,rest]
print(json.dumps(e))
PY
)
export PROMPT EXTRA_JSON TASK=fl2va
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL || true
stable() {
  [[ "$(systemctl --user show h3-tp2-head.service -p MainPID --value)" = "$head_pid" ]] || return 1
  [[ "$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" systemctl --user show h3-tp2-agent.service -p MainPID --value)" = "$agent_pid" ]] || return 1
  python3 - "$proc_root" "$head_pid" <<'PY'
from pathlib import Path
import sys
e=dict(x.split(b'=',1) for x in (Path(sys.argv[1])/sys.argv[2]/'environ').read_bytes().split(b'\0') if b'=' in x)
if e.get(b'H3_TP2_VAE_BLEND_TRACE',b'0') == b'1': raise SystemExit('trace still enabled')
PY
}
health() { curl -sS -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:18093/health || true; }
no_jobs() { curl -fsS -m 5 'http://127.0.0.1:18093/v1/videos?limit=1' | python3 -c 'import json,sys; x=json.load(sys.stdin); sys.exit(0 if x.get("data")==[] and x.get("has_more") is False else 1)'; }
no_new_xid() {
  local head_log worker_log
  head_log=$(mktemp); worker_log=$(mktemp)
  if ! journalctl -k --since '2026-10-31 21:42:00' --no-pager > "$head_log" ||
     ! ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$worker" \
       "journalctl -k --since '2026-10-31 21:42:00' --no-pager" > "$worker_log"; then
    rm -f "$head_log" "$worker_log"; echo 'kernel journal query failed'; return 1
  fi
  if grep 'NVRM: Xid' "$head_log" "$worker_log"; then
    rm -f "$head_log" "$worker_log"; echo 'new Xid requires diagnosis'; return 1
  fi
  rm -f "$head_log" "$worker_log"
}
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
  ((consecutive>=3)) && break
  sleep "$idle_s"
done
((consecutive>=3)) && no_jobs && stable && no_new_xid || { echo 'pair busy, unguarded, changed or has a new Xid'; exit 1; }
disk=$(df -Pk "$results" | awk 'NR==2 {print $4}')
[[ "$disk" =~ ^[0-9]+$ ]] && ((disk>=min_disk_kb)) || { echo 'disk margin insufficient'; exit 1; }
[[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]]
start=$(date '+%F %T'); phase=rendering
mark 'H3-ASPECT-CLOUDLIGHT-START matched-30s'
mark "H3-ASPECT-START $name"
OUT="$results/$name" timeout --signal=TERM --kill-after=30s 12000 bash "$bench" \
  '30:960:544:20:1101' "$paint" > "$results/$name.log" 2>&1
valid "$name" "$tag" && stable && no_new_xid
python3 - "$name/$tag.mp4.timing.json" "$PROMPT" "$paint" "$EXTRA_JSON" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
if x['http'] not in (200,'200') or x['curl_rc']!=0: raise SystemExit('request failed')
if (x['duration'],x['size'],x['steps'],x['seed'],x['task']) != (30,'960x544',20,1101,'fl2va'): raise SystemExit('settings differ')
if x['prompt']!=sys.argv[2] or x['image']!=sys.argv[3] or x['extra_params']!=json.loads(sys.argv[4]): raise SystemExit('source or parameters differ')
PY
[[ "$(sha256sum "$paint" | awk '{print $1}')" = "$image_sha" ]]
for f in "$name/$tag.mem-head.csv" "$name/$tag.mem-worker.csv"; do
  awk -F, 'NF>=2 && $2~/^[0-9]+$/ {n++; if ($2<8192) low=1} END {exit !(n>=2 && !low)}' "$f"
done
printf '%s\n' "$image_sha" > "$name/source.sha256"
printf '%s %s\n' "$head_pid" "$agent_pid" > "$name/service-pids.txt"
printf '%s\n' "$start" > "$name/started.txt"
phase=metrics
PY="$py" timeout --signal=TERM --kill-after=30s 3600 bash "$metrics" "$name/$tag.mp4" 277 22 > "$name/metrics.txt" 2>&1
timeout --signal=TERM --kill-after=30s 3600 "$py" "$flicker" "$name/$tag.mp4" --minimum 2 > "$name/luma-flicker.txt" 2>&1
journalctl --user -u h3-tp2-head.service -a --since "$start" --no-pager -o short | grep -oE 'phase diffuse [0-9.]+ s|seam check window [^\"]+' > "$name/phases.txt" || true
mark "H3-ASPECT-OK $name"
phase=done
mark 'H3-ASPECT-CLOUDLIGHT-DONE matched-30s'
