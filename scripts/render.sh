#!/bin/bash
# One synchronous render against the head API. Usage:
#   render.sh <out.mp4> <duration_s> <width> <height> <steps> <seed> "<prompt>" [image.jpg] [first|last|both]
# Writes <out>.headers (incl. X-Stage-Durations / X-Peak-Memory-MB when the profiler is on) and <out>.timing.json.
set -uo pipefail
source "$(dirname "$0")/env.sh"
out=$1; dur=$2; w=$3; h=$4; steps=$5; seed=$6; prompt=$7; img=${8:-}; where=${9:-first}
end_img=${H3_END_IMAGE_FILE:-}
if [ -n "$end_img" ] && { [ -z "$img" ] || [ "$where" != both ]; }; then
  echo "render.sh: H3_END_IMAGE_FILE requires a first image and frame position both" >&2; exit 2
fi
# Content remains in shell memory or private files, never in child argv/env.
# File inputs use an empty seventh positional argument; ambiguous sources fail.
if [ -n "${H3_PROMPT_FILE:-}" ] && [ -n "$prompt" ]; then
  echo "render.sh: choose H3_PROMPT_FILE or the positional prompt, not both" >&2; exit 2
fi
if [ -n "${H3_EXTRA_JSON_FILE:-}" ] && [ -n "${EXTRA_JSON:-}" ]; then
  echo "render.sh: choose H3_EXTRA_JSON_FILE or EXTRA_JSON, not both" >&2; exit 2
fi
export -n EXTRA_JSON CONT_PROMPTS_EXTRA CONT_PROMPTS_REST
request_tmp=$(mktemp -d "${TMPDIR:-/tmp}/h3-text.XXXXXX") || exit 2
audio_tmp=""
cleanup() {
  rm -rf -- "$request_tmp"
  [ -z "$audio_tmp" ] || rm -rf -- "$audio_tmp"
}
trap cleanup EXIT
# printf is a bash builtin: even legacy inputs are not copied into an exec argument.
printf '%s' "$prompt" > "$request_tmp/prompt.input"
printf '%s' "${EXTRA_JSON:-}" > "$request_tmp/extra.input"
printf '%s' "${CONT_PROMPTS_EXTRA:-}" > "$request_tmp/words.input"
printf '%s' "${CONT_PROMPTS_REST:-}" > "$request_tmp/rest.input"
unset prompt EXTRA_JSON CONT_PROMPTS_EXTRA CONT_PROMPTS_REST
# curl parses <path and @path using its form grammar. Quote the path itself,
# including literal quotes/backslashes, so punctuation in TMPDIR is harmless.
curl_form_path() {
  local path=$1
  path=${path//\\/\\\\}
  path=${path//\"/\\\"}
  printf '"%s"' "$path"
}
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
# Prepare both field snapshots in one process. Automatic window prompts and
# JSON merges stay inside Python memory/files; only paths/scalars cross exec.
text_args=("--directory=$request_tmp" "--prompt=${H3_PROMPT_FILE:-$request_tmp/prompt.input}" "--extra=${H3_EXTRA_JSON_FILE:-$request_tmp/extra.input}"
           "--duration=$dur" "--audio-shift=$audio_shift" "--image=$img" "--where=$where" "--task=${TASK:-}"
           "--limit=${H3_TEXT_MAX_FIELD_BYTES:-1048576}")
[ "${CONT_PROMPTS:-}" != auto ] || text_args+=(--auto)
case "${H3_CONT_PREFIX_BASE:-0}" in
  0) ;;
  1) text_args+=(--prefix-base) ;;
  *) echo "render.sh: H3_CONT_PREFIX_BASE must be 0 or 1" >&2; exit 2 ;;
esac
python3 "$(dirname "$0")/render_text_fields.py" "${text_args[@]}" \
  || { echo "render.sh: invalid or oversized text fields, nothing submitted" >&2; exit 2; }
image_meta=""
if [ -n "$end_img" ]; then
  # Preserve known image suffixes so curl and the API identify both files as images.
  image_suffix() {
    case "$1" in
      *.[Pp][Nn][Gg]) printf '.png';;
      *.[Jj][Pp][Gg]|*.[Jj][Pp][Ee][Gg]) printf '.jpg';;
      *.[Ww][Ee][Bb][Pp]) printf '.webp';;
      *.[Hh][Ee][Ii][Cc]) printf '.heic';;
      *.[Hh][Ee][Ii][Ff]) printf '.heif';;
      *) return 1;;
    esac
  }
  first_suffix=$(image_suffix "$img") && last_suffix=$(image_suffix "$end_img") \
    || { echo "render.sh: keyframe images need PNG, JPEG, WEBP, HEIC or HEIF suffixes" >&2; exit 2; }
  first_snapshot="$request_tmp/first$first_suffix"; last_snapshot="$request_tmp/last$last_suffix"
  image_meta="$request_tmp/keyframes.json"
  python3 "$(dirname "$0")/snapshot_first_last.py" --first="$img" --last="$end_img" \
    --first-output="$first_snapshot" --last-output="$last_snapshot" --extra="$request_tmp/extra.json" \
    --metadata="$image_meta" --seconds="$dur" \
    || { echo "render.sh: invalid two-keyframe request, nothing submitted" >&2; exit 2; }
fi
args=(-F "prompt=<$(curl_form_path "$request_tmp/prompt.txt")" --form-string "width=$w" --form-string "height=$h" --form-string "aspect_ratio=16:9" --form-string "fps=24" --form-string "num_inference_steps=$steps" --form-string "flow_shift=$video_shift" --form-string "seed=$seed" -F "extra_params=<$(curl_form_path "$request_tmp/extra.json")")
[ -n "$lora_json" ] && args+=(--form-string "lora=$lora_json")
if [ -n "$end_img" ]; then
  args+=(-F "input_references=@$(curl_form_path "$first_snapshot")" -F "input_references=@$(curl_form_path "$last_snapshot")")
elif [ -n "${H3_REF_IMAGES:-}" ]; then
  # H3_REF_IMAGES=a.png,b.png,...: several reference images for one Ref2VA request (TASK=ref2va). The API's
  # input_references list reaches the pipeline as an image list and every image becomes its own reference block in the
  # packed sequence (upstream encodes each with the VAE and lists each shape), so a scene frame plus a head-and-shoulders
  # crop per character gives the model every identity in every window, visible or not. Experiment of 2026-11-05.
  [ "${TASK:-}" = ref2va ] || { echo "render.sh: H3_REF_IMAGES needs TASK=ref2va (FL2VA reads several images as keyframes)" >&2; exit 2; }
  IFS=, read -r -a _refs <<< "$H3_REF_IMAGES"
  for _r in "${_refs[@]}"; do [ -f "$_r" ] || { echo "render.sh: H3_REF_IMAGES: $_r is not a file" >&2; exit 2; }; args+=(-F "input_references=@$(curl_form_path "$_r")"); done
elif [ -n "$img" ]; then
  args+=(-F "input_reference=@$(curl_form_path "$img")")
fi
# AUDIO_URL or H3_AUDIO_FILE: a driving soundtrack (audio_mode=lock_source in
# EXTRA_JSON). For a file, curl reads the JSON field from a temporary file;
# passing a long data URL as one command-line argument can exceed ARG_MAX.
audio_meta=""
if [ -n "${AUDIO_URL:-}" ] && [ -n "${H3_AUDIO_FILE:-}" ]; then
  echo "render.sh: choose AUDIO_URL or H3_AUDIO_FILE, not both" >&2; exit 2
fi
if [ -n "${H3_AUDIO_FILE:-}" ]; then
  audio_tmp=$(mktemp -d "${TMPDIR:-/tmp}/h3-audio.XXXXXX") || exit 2
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
  args+=(-F "audio_reference=<$(curl_form_path "$audio_tmp/reference.json")")
elif [ -n "${AUDIO_URL:-}" ]; then
  args+=(--form-string "audio_reference=$(python3 -c 'import json,sys; print(json.dumps({"audio_url": sys.argv[1]}))' "$AUDIO_URL")")
fi
# H3_SOURCE_VIDEO + H3_VIDEO_NOISE_MASK: upstream's latent-mask editing for one request (the server resizes the
# source to the request and clones its last frame to the request's length; the mask is a JSON token grid, 0 = keep the
# source, 1 = generate). scripts/first_frame_pin.py writes both for a pinned first frame. Not for continuation requests.
edit_meta=""
if [ -n "${H3_SOURCE_VIDEO:-}" ] || [ -n "${H3_VIDEO_NOISE_MASK:-}" ]; then
  [ -n "${H3_SOURCE_VIDEO:-}" ] && [ -n "${H3_VIDEO_NOISE_MASK:-}" ] || { echo "render.sh: H3_SOURCE_VIDEO and H3_VIDEO_NOISE_MASK go together" >&2; exit 2; }
  [ -f "$H3_SOURCE_VIDEO" ] && [ -f "$H3_VIDEO_NOISE_MASK" ] || { echo "render.sh: H3_SOURCE_VIDEO or H3_VIDEO_NOISE_MASK is not a file" >&2; exit 2; }
  case "$H3_SOURCE_VIDEO" in *.[Mm][Pp]4|*.[Mm][Oo][Vv]) ;; *) echo "render.sh: H3_SOURCE_VIDEO must be an MP4 or MOV file" >&2; exit 2;; esac
  python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$H3_VIDEO_NOISE_MASK" || { echo "render.sh: H3_VIDEO_NOISE_MASK is not JSON" >&2; exit 2; }
  args+=(-F "source_video=@$(curl_form_path "$H3_SOURCE_VIDEO")" -F "video_noise_mask=@$(curl_form_path "$H3_VIDEO_NOISE_MASK")")
  edit_meta="$H3_SOURCE_VIDEO,$H3_VIDEO_NOISE_MASK"
fi
t0=$(date +%s.%N)
# -q must be curl's first argument. Otherwise a user's .curlrc may add URLs
# or other options and send this multipart request to unintended destinations.
code=$(curl -q -sS --max-time "${H3_RENDER_TIMEOUT_S:-14400}" -X POST "http://$H3_API_HOST:$H3_API_PORT/v1/videos/sync" "${args[@]}" -D "$out.headers" -o "$out" -w "%{http_code}"); crc=$?
t1=$(date +%s.%N)
el=$(python3 -c "print(round($t1-$t0,1))")
python3 - "$out" "$code" "$crc" "$el" "$dur" "$w" "$h" "$steps" "$seed" "$img" "$where" "$request_tmp/extra.json" "$request_tmp/prompt.txt" "${AUDIO_URL:-}" "$video_shift" "$lora_json" "$audio_meta" "$request_tmp/text-meta.json" "$image_meta" <<'PY' > "$out.timing.json" || { echo "render.sh: timing.json failed" >&2; exit 3; }
import json, os, sys
from pathlib import Path
o, code, crc, el, dur, w, h, steps, seed, img, where, extra_file, prompt_file, audio, video_shift, lora_json, audio_meta, text_meta, image_meta = sys.argv[1:]
e = json.loads(Path(extra_file).read_bytes())
prompt = Path(prompt_file).read_bytes().decode("utf-8")
lora = json.loads(lora_json) if lora_json else None
if lora is not None:
    lora = {"name": lora.get("name") or lora.get("lora_name") or lora.get("adapter"),
            "scale": lora.get("scale", lora.get("lora_scale")),
            "int_id": lora.get("int_id", lora.get("lora_int_id")), "path_recorded": False}
target = e.get("target")
if target is not None and not isinstance(target, dict):
    raise ValueError("target must be an object")
eff = (target if target is not None else {}).get("duration_seconds", e.get("duration_seconds", e.get("duration", dur)))
record = {"out": o, "http": code, "curl_rc": int(crc), "elapsed_s": float(el), "duration": float(dur),
                  "effective_duration": None if eff is None else float(eff), "size": f"{w}x{h}", "steps": int(steps), "seed": int(seed),
                  "task": e.get("task"), "image": img, "ref_images": os.environ.get("H3_REF_IMAGES") or None,
                  "frame_indices": where if (img and e.get("task") == "fl2va") else None,
                  "video_flow_shift_form": float(video_shift),
                  "video_flow_shift_after_extra_merge": e.get("flow_shift", float(video_shift)), "lora": lora,
                  "latent_edit": os.environ.get("H3_SOURCE_VIDEO") and {"source_video": os.environ.get("H3_SOURCE_VIDEO"), "video_noise_mask": os.environ.get("H3_VIDEO_NOISE_MASK")} or None,
                  "extra_params": e, "text_fields": json.loads(Path(text_meta).read_bytes()), "audio_reference": audio or None,
                  "audio_reference_file": json.loads(Path(audio_meta).read_text()) if audio_meta else None, "prompt": prompt}
if image_meta:
    record["keyframe_snapshots"] = json.loads(Path(image_meta).read_bytes())
print(json.dumps(record))
PY
cat "$out.timing.json"
grep -i -E "x-stage-durations|x-peak-memory" "$out.headers" || true
[ "$crc" = 0 ] && [ "$code" = "200" ]   # a 200 with a broken body transfer is a failure too
