#!/bin/bash
# Render matrix with memory sampling on both nodes. Run on the head. Results land in $OUT (default ~/h3-tp2-results/<stamp>).
# Usage: bench-matrix.sh "<dur:w:h:steps:seed> ..." [image.jpg]
# Every case gets a status line (ok / render-failed / verify-failed) in $OUT/STATUS and the script exits non-zero if any failed.
set -uo pipefail
source "$(dirname "$0")/env.sh"
S="$(cd "$(dirname "$0")" && pwd)"
WORKER_SSH=${WORKER_SSH:-knapcio@$WORKER_IP}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
OUT=${OUT:-$HOME/h3-tp2-results/$(date +%Y%m%d-%H%M%S)}; mkdir -p "$OUT"
PROMPT=${PROMPT:-"Slow drifting clouds over rolling green hills at golden hour, soft wind in the grass, gentle light moving across the valley, calm, photoreal, steady camera."}
img=${2:-}
WDIR="h3-tp2-results/memwatch"   # on the worker, relative to $HOME
cur_tag=""; rpid=""
cleanup() {
  [ -n "$cur_tag" ] || return 0
  bash "$S/memwatch.sh" stop "$OUT/$cur_tag.mem-head.csv" >/dev/null 2>&1
  ssh "${SSH_OPTS[@]}" "$WORKER_SSH" "bash ~/h3-tp2/scripts/memwatch.sh stop ~/$WDIR/$cur_tag.mem-worker.csv" >/dev/null 2>&1
}
on_signal() { sig=$1; echo "interrupted ($sig)"; [ -n "$rpid" ] && { kill -TERM "$rpid" 2>/dev/null; wait "$rpid" 2>/dev/null; }; cleanup; trap - EXIT; [ "$sig" = INT ] && exit 130 || exit 143; }
trap cleanup EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
fail=0
for spec in $1; do
  IFS=: read -r dur w h steps seed <<< "$spec"
  tag="d${dur}_${w}x${h}_s${steps}_seed${seed}${img:+_fl2va}"
  n=0; while [ -e "$OUT/$tag.mp4" ]; do n=$((n+1)); tag="d${dur}_${w}x${h}_s${steps}_seed${seed}${img:+_fl2va}_r$n"; done   # never overwrite
  cur_tag=$tag
  echo "=== $tag  $(date +%H:%M:%S)"
  bash "$S/memwatch.sh" start "$OUT/$tag.mem-head.csv"
  ssh "${SSH_OPTS[@]}" "$WORKER_SSH" "mkdir -p ~/$WDIR && bash ~/h3-tp2/scripts/memwatch.sh start ~/$WDIR/$tag.mem-worker.csv" || echo "worker memwatch start failed (continuing without worker samples)"
  status=ok
  bash "$S/render.sh" "$OUT/$tag.mp4" "$dur" "$w" "$h" "$steps" "$seed" "$PROMPT" ${img:+"$img"} ${img:+first} > "$OUT/$tag.render.log" 2>&1 &
  rpid=$!; wait "$rpid" || status=render-failed; rpid=""
  tail -3 "$OUT/$tag.render.log"
  bash "$S/memwatch.sh" stop "$OUT/$tag.mem-head.csv" | sed 's/^/head: /'
  if ssh "${SSH_OPTS[@]}" "$WORKER_SSH" "bash ~/h3-tp2/scripts/memwatch.sh stop ~/$WDIR/$tag.mem-worker.csv; cat ~/$WDIR/$tag.mem-worker.csv" > "$OUT/$tag.mem-worker.csv.tmp" 2>/dev/null; then
    head -1 "$OUT/$tag.mem-worker.csv.tmp" | sed 's/^/worker: /'; tail -n +2 "$OUT/$tag.mem-worker.csv.tmp" > "$OUT/$tag.mem-worker.csv"
  else echo "worker: samples unavailable"; fi
  rm -f "$OUT/$tag.mem-worker.csv.tmp"
  if [ "$status" = ok ]; then
    bash "$S/verify-output.sh" "$OUT/$tag.mp4" "$dur" "$w" "$h" | tee "$OUT/$tag.verify.txt" || status=verify-failed
  fi
  echo "$tag $status" | tee -a "$OUT/STATUS"
  [ "$status" = ok ] || fail=1
  cur_tag=""
done
echo "results in $OUT (fail=$fail)"
exit $fail
