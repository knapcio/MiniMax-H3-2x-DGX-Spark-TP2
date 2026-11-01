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
# CONT_PROMPTS_REST supplies the later-window wording; {base} explicitly inserts the original prompt if wanted.
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
        rest_template = os.environ.get("CONT_PROMPTS_REST", "").strip()
        rest = (rest_template.replace("{base}", base) if rest_template else
                "The same continuous shot goes on without any cut or transition, the camera keeps moving the same way: " + base)
        rest += " " + extra if extra else ""
        e["continuation_prompts"] = [first] + [rest] * (n - 1)
print(json.dumps(e))
PY
) || { echo "render.sh: CONT_PROMPTS=auto failed" >&2; exit 2; }
fi
# All text fields are literal. With curl -F, a semicolon inside a prompt or
# JSON string is parsed as form metadata and silently truncates its value.
args=(--form-string "prompt=$prompt" --form-string "width=$w" --form-string "height=$h" --form-string "aspect_ratio=16:9" --form-string "fps=24" --form-string "num_inference_steps=$steps" --form-string "flow_shift=$video_shift" --form-string "seed=$seed" --form-string "extra_params=$extra")
[ -n "$lora_json" ] && args+=(--form-string "lora=$lora_json")
[ -n "$img" ] && args+=(-F "input_reference=@$img")
# AUDIO_URL or H3_AUDIO_FILE: a driving soundtrack (audio_mode=lock_source in
# EXTRA_JSON). For a file, curl reads the JSON field from a temporary file;
# passing a long data URL as one command-line argument can exceed ARG_MAX.
audio_meta=""
if [ -n "${AUDIO_URL:-}" ] && [ -n "${H3_AUDIO_FILE:-}" ]; then
  echo "render.sh: choose AUDIO_URL or H3_AUDIO_FILE, not both" >&2; exit 2
fi
if [ -n "${H3_AUDIO_FILE:-}" ]; then
  audio_tmp=$(mktemp -d "${TMPDIR:-/tmp}/h3-audio.XXXXXX") || exit 2
  trap 'rm -rf -- "$audio_tmp"' EXIT
  audio_meta="$audio_tmp/meta.json"
  python3 - "$H3_AUDIO_FILE" "$audio_tmp/reference.json" "$audio_meta" "${H3_AUDIO_MAX_FIELD_BYTES:-1048576}" <<'PY' || { echo "render.sh: invalid or oversized audio file, nothing submitted" >&2; exit 2; }
import base64, hashlib, json, sys
from pathlib import Path
source, reference, meta = map(Path, sys.argv[1:4])
limit = int(sys.argv[4])
types = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}
mime = types.get(source.suffix.lower())
if not mime or not source.is_file() or source.stat().st_size == 0 or limit <= 0:
    raise ValueError("audio must be a nonempty WAV, MP3, FLAC, OGG or M4A file")
size_on_disk = source.stat().st_size
prefix = '{"audio_url":"data:' + mime + ';base64,'
field_bytes = len(prefix) + 4 * ((size_on_disk + 2) // 3) + 2
if field_bytes > limit:
    raise ValueError(f"audio reference field needs {field_bytes} bytes, limit is {limit}; use a reachable AUDIO_URL or raise the limit only if the server is configured for it")
digest, size = hashlib.sha256(), 0
with source.open("rb") as inp, reference.open("w", encoding="ascii") as out:
    out.write(prefix)
    while chunk := inp.read(3 * 1024 * 1024):
        digest.update(chunk); size += len(chunk)
        out.write(base64.b64encode(chunk).decode("ascii"))
    out.write('"}')
if size != size_on_disk:
    raise ValueError("audio file size changed while reading")
meta.write_text(json.dumps({"kind": "inline-audio-file", "bytes": size, "field_bytes": field_bytes,
                            "sha256": digest.hexdigest(), "mime": mime}))
PY
  args+=(-F "audio_reference=<$audio_tmp/reference.json")
elif [ -n "${AUDIO_URL:-}" ]; then
  args+=(--form-string "audio_reference=$(python3 -c 'import json,sys; print(json.dumps({"audio_url": sys.argv[1]}))' "$AUDIO_URL")")
fi
t0=$(date +%s.%N)
code=$(curl -sS --max-time "${H3_RENDER_TIMEOUT_S:-14400}" -X POST "http://$H3_API_HOST:$H3_API_PORT/v1/videos/sync" "${args[@]}" -D "$out.headers" -o "$out" -w "%{http_code}"); crc=$?
t1=$(date +%s.%N)
el=$(python3 -c "print(round($t1-$t0,1))")
python3 - "$out" "$code" "$crc" "$el" "$dur" "$w" "$h" "$steps" "$seed" "$img" "$where" "$extra" "$prompt" "${AUDIO_URL:-}" "$video_shift" "$lora_json" "$audio_meta" <<'PY' > "$out.timing.json" || { echo "render.sh: timing.json failed" >&2; exit 3; }
import json, sys
from pathlib import Path
o, code, crc, el, dur, w, h, steps, seed, img, where, extra, prompt, audio, video_shift, lora_json, audio_meta = sys.argv[1:]
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
                  "extra_params": e, "audio_reference": audio or None,
                  "audio_reference_file": json.loads(Path(audio_meta).read_text()) if audio_meta else None, "prompt": prompt}))
PY
cat "$out.timing.json"
grep -i -E "x-stage-durations|x-peak-memory" "$out.headers" || true
[ "$crc" = 0 ] && [ "$code" = "200" ]   # a 200 with a broken body transfer is a failure too
