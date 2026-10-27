#!/bin/bash
# One synchronous render against the head API. Usage:
#   render.sh <out.mp4> <duration_s> <width> <height> <steps> <seed> "<prompt>" [image.jpg] [first|last|both]
# Writes <out>.headers (incl. X-Stage-Durations / X-Peak-Memory-MB when the profiler is on) and <out>.timing.json.
set -uo pipefail
source "$(dirname "$0")/env.sh"
out=$1; dur=$2; w=$3; h=$4; steps=$5; seed=$6; prompt=$7; img=${8:-}; where=${9:-first}
# TASK=ref2va switches to the Ref2VA partition's request shape (reference image, no frame_indices); EXTRA_JSON merges
# more extra_params, e.g. '{"long_video": true, "long_video_mode": "continuation"}' for the continuation mode.
TASK=${TASK:-}
if [ "$TASK" = ref2va ]; then
  extra="{\"task\": \"ref2va\", \"duration\": $dur, \"audio_flow_shift\": 3.0}"
elif [ -n "$img" ]; then
  case "$where" in first) fi='[0]';; last) fi='[-1]';; both) fi='[0, -1]';; esac
  extra="{\"task\": \"fl2va\", \"duration\": $dur, \"audio_flow_shift\": 3.0, \"frame_indices\": $fi}"
else
  extra="{\"task\": \"t2va\", \"duration\": $dur, \"audio_flow_shift\": 3.0}"
fi
if [ -n "${EXTRA_JSON:-}" ]; then
  extra=$(python3 -c 'import json,sys; a=json.loads(sys.argv[1]); b=json.loads(sys.argv[2]); assert isinstance(b, dict), "EXTRA_JSON must be a JSON object"; a.update(b); print(json.dumps(a))' "$extra" "$EXTRA_JSON") \
    || { echo "render.sh: invalid EXTRA_JSON, nothing submitted" >&2; exit 2; }
fi
args=(-F "prompt=$prompt" -F "width=$w" -F "height=$h" -F "aspect_ratio=16:9" -F "fps=24" -F "num_inference_steps=$steps" -F "flow_shift=12" -F "seed=$seed" -F "extra_params=$extra")
[ -n "$img" ] && args+=(-F "input_reference=@$img")
# AUDIO_URL: a driving soundtrack for Ref2VA (audio_mode=lock_source in EXTRA_JSON); http(s) or data: URL per upstream docs
[ -n "${AUDIO_URL:-}" ] && args+=(-F "audio_reference={\"audio_url\": \"$AUDIO_URL\"}")
t0=$(date +%s.%N)
code=$(curl -sS --max-time 14400 -X POST "http://$H3_API_HOST:$H3_API_PORT/v1/videos/sync" "${args[@]}" -D "$out.headers" -o "$out" -w "%{http_code}"); crc=$?
t1=$(date +%s.%N)
el=$(python3 -c "print(round($t1-$t0,1))")
printf '{"out":"%s","http":"%s","curl_rc":%s,"elapsed_s":%s,"duration":%s,"size":"%sx%s","steps":%s,"seed":%s,"image":"%s","frame_indices":"%s"}\n' "$out" "$code" "$crc" "$el" "$dur" "$w" "$h" "$steps" "$seed" "$img" "$where" | tee "$out.timing.json"
grep -i -E "x-stage-durations|x-peak-memory" "$out.headers" || true
[ "$crc" = 0 ] && [ "$code" = "200" ]   # a 200 with a broken body transfer is a failure too
