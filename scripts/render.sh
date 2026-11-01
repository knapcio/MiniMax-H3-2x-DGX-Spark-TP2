#!/bin/bash
# One synchronous render against the head API. Usage:
#   render.sh <out.mp4> <duration_s> <width> <height> <steps> <seed> "<prompt>" [image.jpg] [first|last|both]
# Writes <out>.headers (incl. X-Stage-Durations / X-Peak-Memory-MB when the profiler is on) and <out>.timing.json.
set -uo pipefail
source "$(dirname "$0")/env.sh"
out=$1; dur=$2; w=$3; h=$4; steps=$5; seed=$6; prompt=$7; img=${8:-}; where=${9:-first}
# Optional request-level sampler/adapter settings. Defaults preserve existing
# requests. An adapter requires an object accepted by /v1/videos: name, path,
# optional scale and ID. Its path is not written into the timing record.
video_shift=${H3_VIDEO_FLOW_SHIFT:-12}
audio_shift=${H3_AUDIO_FLOW_SHIFT:-3.0}
lora_json=${H3_LORA_JSON:-}
python3 - "$video_shift" "$audio_shift" "$lora_json" <<'PY' || { echo "render.sh: invalid flow shift or LoRA object, nothing submitted" >&2; exit 2; }
import json, math, sys
for raw in sys.argv[1:3]:
    value = json.loads(raw)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("flow shifts must be finite positive JSON numbers")
if sys.argv[3]:
    lora = json.loads(sys.argv[3], parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(lora, dict):
        raise ValueError("LoRA must be an object")
    name = lora.get("name") or lora.get("lora_name") or lora.get("adapter")
    path = lora.get("local_path") or lora.get("path") or lora.get("lora_path") or lora.get("lora_local_path")
    if not isinstance(name, str) or not name.strip() or not isinstance(path, str) or not path.strip():
        raise ValueError("LoRA name and path are required")
    scale = lora.get("scale", lora.get("lora_scale"))
    if scale is not None and (isinstance(scale, bool) or not math.isfinite(float(scale))):
        raise ValueError("LoRA scale must be finite")
    adapter_id = lora.get("int_id", lora.get("lora_int_id"))
    if adapter_id is not None and (type(adapter_id) not in (int, str) or not str(adapter_id).isdigit()
                                   or not 1 <= int(adapter_id) < 2**63):
        raise ValueError("LoRA int_id must be a positive 63-bit integer")
PY
# TASK=ref2va switches to the Ref2VA partition's request shape (reference image, no frame_indices); EXTRA_JSON merges
# more extra_params, e.g. '{"long_video": true, "long_video_mode": "continuation"}' for the continuation mode.
TASK=${TASK:-}
if [ "$TASK" = ref2va ]; then
  extra="{\"task\": \"ref2va\", \"duration\": $dur, \"audio_flow_shift\": $audio_shift}"
elif [ -n "$img" ]; then
  case "$where" in first) fi='[0]';; last) fi='[-1]';; both) fi='[0, -1]';; esac
  extra="{\"task\": \"fl2va\", \"duration\": $dur, \"audio_flow_shift\": $audio_shift, \"frame_indices\": $fi}"
else
  extra="{\"task\": \"t2va\", \"duration\": $dur, \"audio_flow_shift\": $audio_shift}"
fi
if [ -n "${EXTRA_JSON:-}" ]; then
  extra=$(python3 -c 'import json,sys; a=json.loads(sys.argv[1]); b=json.loads(sys.argv[2]); assert isinstance(b, dict), "EXTRA_JSON must be a JSON object"; a.update(b); print(json.dumps(a))' "$extra" "$EXTRA_JSON") \
    || { echo "render.sh: invalid EXTRA_JSON, nothing submitted" >&2; exit 2; }
fi
# CONT_PROMPTS=auto: for a continuation request, build one prompt per window from "$prompt" (window plan as upstream:
# frames aligned to 17n+5, windows of continuation_window_frames (277) with continuation_overlap_frames (22) overlap),
# restating that the same shot continues.  Runs after the EXTRA_JSON merge and uses the effective duration (upstream
# precedence: target.duration_seconds, duration_seconds, duration); supplied continuation_prompts are kept as they are.
# CONT_PROMPTS_EXTRA="<sentence>" is appended to every generated window prompt (experiments on wording).
if [ "${CONT_PROMPTS:-}" = auto ]; then
  extra=$(python3 - "$extra" "$prompt" <<'PY'
import json, math, os, sys
e, base = json.loads(sys.argv[1]), sys.argv[2]
if e.get("long_video") and e.get("long_video_mode", "continuation") == "continuation" and not e.get("continuation_prompts"):
    target = e.get("target")
    if target is not None and not isinstance(target, dict):
        raise ValueError("target must be an object")
    dur = (target if target is not None else {}).get("duration_seconds", e.get("duration_seconds", e.get("duration")))
    if dur is not None:  # null selects the server's num_frames/default path
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
args=(-F "prompt=$prompt" -F "width=$w" -F "height=$h" -F "aspect_ratio=16:9" -F "fps=24" -F "num_inference_steps=$steps" -F "flow_shift=$video_shift" -F "seed=$seed" -F "extra_params=$extra")
[ -n "$lora_json" ] && args+=(--form-string "lora=$lora_json")
[ -n "$img" ] && args+=(-F "input_reference=@$img")
# AUDIO_URL: a driving soundtrack for Ref2VA (audio_mode=lock_source in EXTRA_JSON); http(s) or data: URL per upstream docs
[ -n "${AUDIO_URL:-}" ] && args+=(--form-string "audio_reference=$(python3 -c 'import json,sys; print(json.dumps({"audio_url": sys.argv[1]}))' "$AUDIO_URL")")
t0=$(date +%s.%N)
code=$(curl -sS --max-time "${H3_RENDER_TIMEOUT_S:-14400}" -X POST "http://$H3_API_HOST:$H3_API_PORT/v1/videos/sync" "${args[@]}" -D "$out.headers" -o "$out" -w "%{http_code}"); crc=$?
t1=$(date +%s.%N)
el=$(python3 -c "print(round($t1-$t0,1))")
python3 - "$out" "$code" "$crc" "$el" "$dur" "$w" "$h" "$steps" "$seed" "$img" "$where" "$extra" "$prompt" "${AUDIO_URL:-}" "$video_shift" "$lora_json" <<'PY' > "$out.timing.json" || { echo "render.sh: timing.json failed" >&2; exit 3; }
import json, sys
o, code, crc, el, dur, w, h, steps, seed, img, where, extra, prompt, audio, video_shift, lora_json = sys.argv[1:]
e = json.loads(extra)
lora = json.loads(lora_json) if lora_json else None
if lora is not None:
    lora = {"name": lora.get("name") or lora.get("lora_name") or lora.get("adapter"),
            "scale": lora.get("scale", lora.get("lora_scale")),
            "int_id": lora.get("int_id", lora.get("lora_int_id")), "path_recorded": False}
target = e.get("target")
if target is not None and not isinstance(target, dict):
    raise ValueError("target must be an object")
eff = (target if target is not None else {}).get("duration_seconds", e.get("duration_seconds", e.get("duration", dur)))
print(json.dumps({"out": o, "http": code, "curl_rc": int(crc), "elapsed_s": float(el), "duration": float(dur),
                  "effective_duration": None if eff is None else float(eff), "size": f"{w}x{h}", "steps": int(steps), "seed": int(seed),
                  "task": e.get("task"), "image": img, "frame_indices": where if (img and e.get("task") == "fl2va") else None,
                  "video_flow_shift_form": float(video_shift),
                  "video_flow_shift_after_extra_merge": e.get("flow_shift", float(video_shift)), "lora": lora,
                  "extra_params": e, "audio_reference": audio or None, "prompt": prompt}))
PY
cat "$out.timing.json"
grep -i -E "x-stage-durations|x-peak-memory" "$out.headers" || true
[ "$crc" = 0 ] && [ "$code" = "200" ]   # a 200 with a broken body transfer is a failure too
