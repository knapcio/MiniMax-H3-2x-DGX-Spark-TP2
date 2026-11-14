#!/bin/bash
# Shared helpers for the render queues on the head (source it: . ~/h3-tp2/scripts/queue-lib.sh). Replaces the header
# that every queueNN.sh used to copy. Env lines written to ~/.h3-tp2.env must be quoted when the value has spaces.
cd ~/h3-tp2-results
IMG=~/h3-forest-20260920/source-4k.jpg
PAINT=~/h3-tp2-results/koninck.png
health() { curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:18093/health; }
# rate "<since>": the step rates the head logged since that time; digs "<since>": the 12-hex latents digests (probe, then
# one per window). Both were private to the gate queues until 2026-11-05 (queue83 printed "FAIL" for want of them).
rate() { journalctl --user -u h3-tp2-head.service -a --since "$1" --no-pager -o short | tr "\r" "\n" | grep -o -E "20/20 \[[0-9:]+<00:00, +[0-9.]+s/it\]" | tr "\n" " "; }
digs() { journalctl --user -u h3-tp2-head.service --since "$1" --no-pager | grep -o -E "latents sha256 [0-9a-f]{12}" | awk '"'"'{print $3}'"'"' | tr "\n" " "; }
# lever_lines "<since>": how often the experiment levers that run inside the denoising loop logged, per rank. The head
# journal misses the per-attempt lines of the loop wrappers (queue145 saw none for the hand-over noise while rank 1 had
# one per attempt), so the rank 1 worker logs newer than <since> are read too (W must be set to the worker's address).
LEVER_PAT='hand-over noise: [0-9]+ of [0-9]+ pinned video rows shown at noise level [0-9.]+|voice anchor: the first [0-9.]+ s of window [0-9]+.{0,60}|audio pin released at step [0-9]+ of [0-9]+'
lever_lines() { local T_=$1
  echo "rank0: $(journalctl --user -u h3-tp2-head.service --since "$T_" --no-pager | grep -o -E "$LEVER_PAT" | sort | uniq -c | tr -s " " | tr "\n" ";" | cut -c1-400) | rank1: $(ssh -o BatchMode=yes -o ConnectTimeout=10 ${W:-10.100.96.5} "find ~/h3-tp2-logs -name 'worker-rank1-*.log' -newermt '$T_' -print0 | xargs -0 grep -h -o -E '$LEVER_PAT' 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ';'" 2>/dev/null | cut -c1-400)"
}
STRIP='^export H3_TP2_(FP8_CUTLASS_VERIFY|ID_LOCK[A-Z_]*|ID_END_EXTEND|A2A_OVERLAP|A2A_STREAM_PRIORITY|WINDOW_SEED_STEP|REF_SHORT_EDGE|REF_NO_UPSCALE|IDENTITY_[A-Z_]*|SPEECH_[A-Z_]*|WINDOW_TEMPORAL_OFFSET|PROFILE_STEP|ANCHOR_FRAMES|ANCHOR_SOURCE|FP8_GEMM|VAE_STACK_TILES|VAE_TILE_BATCH|FP8_FUSE_SWIGLU|CONT_PROMPTS_EXTRA|CONT_PROMPTS_REST|CONT_MATCH|CONT_XFADE|CONT_EDIT[A-Z_]*|CONT_RETRY[A-Z_]*|CONT_SEAM_RATIO|CONT_LOCK[A-Z_]*|CONT_CUT_RATIO|SAVE_LATENTS|REPLAY_LATENTS|REPLAY_PARTIAL|REF_KEY_SCALE|VAE_BLEND_TRACE|VAE_LEFT_CONTEXT|EMPTY_CACHE_BEFORE_DECODE|KEEP_POOL|LADDER_TAIL|REF_FOLLOW_WINDOW|AUTO_CARDS|GUARD_SINGLE|VOICE_ANCHOR|A2A_STREAM_PRIORITY|ULYSSES_OVERLAP|SEAM_POSTCHECK|ULYSSES_NOCOPY|ULYSSES_EXACT|FP8_CONSUMER_FUSE|STEP_CACHE|FP8_QUANT_FUSE|FP8_QUANT_FUSE_VERIFY|FP8_QUANT_FUSE_MODE)=|^export NCCL_MAX_CTAS='
BASEENV='export H3_TP2_FP8_FUSE_SWIGLU=1
export H3_TP2_REF_SHORT_EDGE=1024'
# restart_head "export VAR=value[newline export ...]": rewrite the experiment vars, restart, wait for health (20 min max)
restart_head() {
  local i   # the wait loops below must not clobber a caller's loop variable (2026-11-14: a queue lost two renders to i=24)
  # skip when the head is healthy and the env file already carries exactly the requested experiment lines (a queue
  # that ends with a restoring restart followed by one that starts with the same restart cost a 13-minute double boot)
  # (H3_FORCE_RESTART=1 skips the shortcut: restart_head_gated sets it after a slow probe, when "the same env on a
  # healthy head" is exactly the boot it wants to replace)
  if [ -z "${H3_FORCE_RESTART:-}" ] && [ "$(health)" = 200 ]; then
    grep -v -E "$STRIP" ~/.h3-tp2.env > /tmp/h3env.want; printf "%s\n" "$BASEENV" >> /tmp/h3env.want; [ -n "$1" ] && printf "%s\n" "$1" >> /tmp/h3env.want
    # ... and only when the running head loaded the currently synced code (queue52 skipped a restart that was meant to
    # bring a new default into serving, 2026-11-02 21:15)
    if cmp -s /tmp/h3env.want ~/.h3-tp2.env && [ "$(cat ~/h3-tp2-results/.head-rev 2>/dev/null)" = "$(cat ~/h3-tp2/.synced-rev 2>/dev/null)" ]; then echo "=== head already healthy with this env and code, no restart $(date +%H:%M:%S)"; return 0; fi
  fi
  grep Normal /proc/buddyinfo | awk '{printf "buddyinfo before restart: "; for(i=13;i<=NF;i++) printf "%s ", $i; print ""}'
  grep -v -E "$STRIP" ~/.h3-tp2.env > ~/.h3-tp2.env.new; printf "%s\n" "$BASEENV" >> ~/.h3-tp2.env.new; [ -n "$1" ] && printf "%s\n" "$1" >> ~/.h3-tp2.env.new; mv -f ~/.h3-tp2.env.new ~/.h3-tp2.env
  # stop, compact both hosts (root script /usr/local/sbin/h3-compact via sudoers.d/h3-compact, 2026-11-02), start:
  # boots into a fragmented host ran 7-40 % slower (results/20261102-speed-plan)
  systemctl --user stop h3-tp2-head.service; sleep 15
  sudo -n /usr/local/sbin/h3-compact 2>/dev/null | awk '{printf "compacted head, orders 9+: "; for(i=13;i<=NF;i++) printf "%s ", $i; print ""}'
  ssh -o BatchMode=yes -o ConnectTimeout=10 10.100.96.5 "sudo -n /usr/local/sbin/h3-compact 2>/dev/null" | awk '{printf "compacted agent host, orders 9+: "; for(i=13;i<=NF;i++) printf "%s ", $i; print ""}'
  cp -f ~/h3-tp2/.synced-rev ~/h3-tp2-results/.head-rev 2>/dev/null
  systemctl --user start h3-tp2-head.service; sleep 20
  for i in $(seq 1 80); do [ "$(health)" = 200 ] && break; systemctl --user is-active --quiet h3-tp2-head.service || { echo "head died during restart"; break; }; sleep 15; done
  echo "=== head restarted ($1) health $(health) $(date +%H:%M:%S)"
  journalctl --user -u h3-tp2-head.service --since "-8min" --no-pager | grep -E "h3_tp2:.*(edit prefix|latent match)" | tail -1 | cut -c80-240
}
# restart_head_gated "env": restart, then the boot probe; a slow boot is restarted again (up to 3 boots)
restart_head_gated() {
  local i   # the wait loops below must not clobber a caller's loop variable (2026-11-14: a queue lost two renders to i=24)
  for attempt in 1 2 3; do
    restart_head "$1"; unset H3_FORCE_RESTART
    bash ~/h3-tp2/scripts/boot-probe.sh ${PROBE_REF:-8.65} ${PROBE_TOL:-0.08}; rc=$?
    [ $rc -eq 0 ] && return 0
    [ $rc -eq 2 ] && { echo "=== slow boot (attempt $attempt), restarting"; H3_FORCE_RESTART=1; continue; }
    echo "=== probe failed (rc $rc), keeping this boot"; return 1
  done
  echo "=== three slow boots in a row, keeping the last"; return 2
}
# run <name> <duration_s> "<dur:w:h:steps:seed>" [image]: render via bench-matrix, wait, print timing/seam/lock lines
run() { name=$1; dur=$2; shift 2
  local i   # see restart_head
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
