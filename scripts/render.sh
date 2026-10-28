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
# CONT_PROMPTS=auto: for a continuation request, build one prompt per window from "$prompt" (window plan as upstream:
# frames aligned to 17n+5, windows of continuation_window_frames (277) with continuation_overlap_frames (22) overlap),
# restating that the same shot continues.  Runs after the EXTRA_JSON merge and uses the effective duration (upstream
# precedence: duration_seconds, target.duration_seconds, duration); supplied continuation_prompts are kept as they are.
# CONT_PROMPTS_EXTRA="<sentence>" is appended to every generated window prompt (experiments on wording).
if [ "${CONT_PROMPTS:-}" = auto ]; then
  extra=$(python3 - "$extra" "$prompt" <<'PY'
import json, math, os, sys
e, base = json.loads(sys.argv[1]), sys.argv[2]
if e.get("long_video") and e.get("long_video_mode", "continuation") == "continuation" and not e.get("continuation_prompts"):
    dur = e.get("duration_seconds", (e.get("target") or {}).get("duration_seconds", e.get("duration")))
    frames = int(round(float(dur) * 24)); frames = ((frames - 5 + 16) // 17) * 17 + 5
    win, ov = int(e.get("continuation_window_frames", 277)), int(e.get("continuation_overlap_frames", 22))
    n = 1 if frames <= win else 1 + math.ceil((frames - win) / (win - ov))
    extra = os.environ.get("CONT_PROMPTS_EXTRA", "").strip()   # e.g. "The light and the sky stay exactly as they are."
    first = base.rstrip(". ") + ". One single continuous take from a steady camera, no cuts, no transitions." + (" " + extra if extra else "")
    rest = "The same continuous shot goes on without any cut or transition, the camera keeps moving the same way: " + base + (" " + extra if extra else "")
    e["continuation_prompts"] = [first] + [rest] * (n - 1)
print(json.dumps(e))
PY
) || { echo "render.sh: CONT_PROMPTS=auto failed" >&2; exit 2; }
fi
args=(-F "prompt=$prompt" -F "width=$w" -F "height=$h" -F "aspect_ratio=16:9" -F "fps=24" -F "num_inference_steps=$steps" -F "flow_shift=12" -F "seed=$seed" -F "extra_params=$extra")
[ -n "$img" ] && args+=(-F "input_reference=@$img")
# AUDIO_URL: a driving soundtrack for Ref2VA (audio_mode=lock_source in EXTRA_JSON); http(s) or data: URL per upstream docs
[ -n "${AUDIO_URL:-}" ] && args+=(--form-string "audio_reference=$(python3 -c 'import json,sys; print(json.dumps({"audio_url": sys.argv[1]}))' "$AUDIO_URL")")
t0=$(date +%s.%N)
code=$(curl -sS --max-time "${H3_RENDER_TIMEOUT_S:-14400}" -X POST "http://$H3_API_HOST:$H3_API_PORT/v1/videos/sync" "${args[@]}" -D "$out.headers" -o "$out" -w "%{http_code}"); crc=$?
t1=$(date +%s.%N)
el=$(python3 -c "print(round($t1-$t0,1))")
python3 - "$out" "$code" "$crc" "$el" "$dur" "$w" "$h" "$steps" "$seed" "$img" "$where" "$extra" "$prompt" "${AUDIO_URL:-}" <<'PY' > "$out.timing.json" || { echo "render.sh: timing.json failed" >&2; exit 3; }
import json, sys
o, code, crc, el, dur, w, h, steps, seed, img, where, extra, prompt, audio = sys.argv[1:]
e = json.loads(extra)
eff = e.get("duration_seconds", (e.get("target") or {}).get("duration_seconds", e.get("duration", dur)))
print(json.dumps({"out": o, "http": code, "curl_rc": int(crc), "elapsed_s": float(el), "duration": float(dur),
                  "effective_duration": float(eff), "size": f"{w}x{h}", "steps": int(steps), "seed": int(seed),
                  "task": e.get("task"), "image": img, "frame_indices": where if (img and e.get("task") == "fl2va") else None,
                  "extra_params": e, "audio_reference": audio or None, "prompt": prompt}))
PY
cat "$out.timing.json"
grep -i -E "x-stage-durations|x-peak-memory" "$out.headers" || true
[ "$crc" = 0 ] && [ "$code" = "200" ]   # a 200 with a broken body transfer is a failure too
