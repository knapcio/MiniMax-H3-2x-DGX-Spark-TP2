#!/bin/bash
# Shared helpers for the render queues on the head (source it: . ~/h3-tp2/scripts/queue-lib.sh). Replaces the header
# that every queueNN.sh used to copy. Env lines written to ~/.h3-tp2.env must be quoted when the value has spaces.
cd ~/h3-tp2-results
IMG=~/h3-forest-20260920/source-4k.jpg
PAINT=~/h3-tp2-results/koninck.png
health() { curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:18093/health; }
STRIP='^export H3_TP2_(WINDOW_SEED_STEP|REF_SHORT_EDGE|WINDOW_TEMPORAL_OFFSET|PROFILE_STEP|ANCHOR_FRAMES|ANCHOR_SOURCE|FP8_GEMM|VAE_STACK_TILES|VAE_TILE_BATCH|FP8_FUSE_SWIGLU|CONT_PROMPTS_EXTRA|CONT_PROMPTS_REST|CONT_MATCH|CONT_XFADE|CONT_EDIT|CONT_RETRY|CONT_SEAM_RATIO|CONT_LOCK|CONT_LOCK_STRENGTH|CONT_LOCK_RAMP|CONT_LOCK_AUDIO|CONT_LOCK_STD|CONT_LOCK_REF|CONT_LOCK_CLASSES|SAVE_LATENTS|VAE_BLEND_TRACE|VAE_LEFT_CONTEXT|EMPTY_CACHE_BEFORE_DECODE|A2A_STREAM_PRIORITY|ULYSSES_OVERLAP|SEAM_POSTCHECK|ULYSSES_NOCOPY)=|^export NCCL_MAX_CTAS='
BASEENV='export H3_TP2_FP8_FUSE_SWIGLU=1
export H3_TP2_REF_SHORT_EDGE=1024'
# restart_head "export VAR=value[newline export ...]": rewrite the experiment vars, restart, wait for health (20 min max)
restart_head() {
  # skip when the head is healthy and the env file already carries exactly the requested experiment lines (a queue
  # that ends with a restoring restart followed by one that starts with the same restart cost a 13-minute double boot)
  if [ "$(health)" = 200 ]; then
    grep -v -E "$STRIP" ~/.h3-tp2.env > /tmp/h3env.want; printf "%s\n" "$BASEENV" >> /tmp/h3env.want; [ -n "$1" ] && printf "%s\n" "$1" >> /tmp/h3env.want
    if cmp -s /tmp/h3env.want ~/.h3-tp2.env; then echo "=== head already healthy with this env, no restart $(date +%H:%M:%S)"; return 0; fi
  fi
  grep Normal /proc/buddyinfo | awk '{printf "buddyinfo before restart: "; for(i=13;i<=NF;i++) printf "%s ", $i; print ""}'
  grep -v -E "$STRIP" ~/.h3-tp2.env > ~/.h3-tp2.env.new; printf "%s\n" "$BASEENV" >> ~/.h3-tp2.env.new; [ -n "$1" ] && printf "%s\n" "$1" >> ~/.h3-tp2.env.new; mv -f ~/.h3-tp2.env.new ~/.h3-tp2.env
  systemctl --user restart h3-tp2-head.service; sleep 20
  for i in $(seq 1 80); do [ "$(health)" = 200 ] && break; systemctl --user is-active --quiet h3-tp2-head.service || { echo "head died during restart"; break; }; sleep 15; done
  echo "=== head restarted ($1) health $(health) $(date +%H:%M:%S)"
  journalctl --user -u h3-tp2-head.service --since "-8min" --no-pager | grep -E "h3_tp2:.*(edit prefix|latent match)" | tail -1 | cut -c80-240
}
# restart_head_gated "env": restart, then the boot probe; a slow boot is restarted again (up to 3 boots)
restart_head_gated() {
  for attempt in 1 2 3; do
    restart_head "$1"
    bash ~/h3-tp2/scripts/boot-probe.sh ${PROBE_REF:-8.65} ${PROBE_TOL:-0.10}; rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 2 ] && { echo "=== slow boot (attempt $attempt), restarting"; continue; }
    echo "=== probe failed (rc $rc), keeping this boot"; return 1
  done
  echo "=== three slow boots in a row, keeping the last"; return 2
}
# run <name> <duration_s> "<dur:w:h:steps:seed>" [image]: render via bench-matrix, wait, print timing/seam/lock lines
run() { name=$1; dur=$2; shift 2
  [ "$(health)" = 200 ] || { echo "health $(health) before $name, skipping"; return 1; }
  T0=$(date "+%Y-%m-%d %H:%M:%S")
  OUT=~/h3-tp2-results/$name setsid nohup bash ~/h3-tp2/scripts/bench-matrix.sh "$@" > ~/h3-tp2-results/$name.log 2>&1 < /dev/null &
  sleep 15; echo "=== $name launched $(date +%H:%M:%S)"
  for i in $(seq 1 1500); do grep -q "^results in" ~/h3-tp2-results/$name.log 2>/dev/null && break; sleep 25; done
  echo "=== $name done $(date +%H:%M:%S)"
  grep -v -E "^\[" ~/h3-tp2-results/$name.log | grep -E "elapsed|peak|framemd5|verify|results in|failed|curl" | cut -c1-200
  journalctl --user -u h3-tp2-head.service --since "$T0" --no-pager | grep -o -E "h3_tp2 phase (diffuse|decode|decode_to_mp4) [0-9.]+ s|seam check[^\"]{0,160}|look-lock[^\"]{0,200}|Traceback|Error[^\"]{0,160}" | tail -14 | sed "s/^/$name: /"
  F=$(ls ~/h3-tp2-results/$name/d${dur}*.mp4 2>/dev/null | head -1)
  [ -n "$F" ] && { PY=~/h3-long/.venv/bin/python bash ~/h3-tp2/scripts/clip-metrics.sh "$F" ${WIN:-277} ${OV:-22} 2>/dev/null | grep -E "cuts|seam|light|colour|motion|detail" | cut -c1-260; echo "##### flicker (>=1 YAVG) + boundary steps"; python3 ~/h3-tp2/scripts/luma_flicker.py --threshold 1.0 "$F" | tail -1; ffmpeg -loglevel error -i "$F" -vf "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-" -f null - 2>/dev/null | grep -o "YAVG=[0-9.]*" | cut -d= -f2 | python3 ~/h3-tp2/scripts/boundary_steps.py; }
}
