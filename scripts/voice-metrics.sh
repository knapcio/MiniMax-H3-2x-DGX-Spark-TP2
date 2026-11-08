#!/bin/bash
# The whole speech report of a dialogue clip in one command: transcript, word and character error rate per window
# against the prompt's <d> lines, UTMOS per window (by spoken line and by window), speaker continuity and the audio
# hand-over seams.  It runs the five tools of docs/DIALOGUE_CLIPS.md section 5 in the interpreters they need and prints
# one compact block; the JSON of every step lands in --out (default: next to the clip, <clip>.voice/).
#
# Usage: voice-metrics.sh <clip.mp4> [--language pl] [--extra extra.json | --prompt prompt.txt] [--respell file]
#                        [--window 277] [--overlap 22] [--fps 24] [--model small] [--out dir]
#   --extra     the request's extra JSON with continuation_prompts (one prompt per window, the first included)
#   --prompt    one prompt for every window (the window count comes from the clip's length)
#   --respell   `prompt_spelling real_word` lines for phonetic respellings written into the prompt
# Interpreters (override with the variables): SPEECH_PY = faster-whisper (.speech-venv/bin/python under the repo,
# ~/speech-venv or ~/.speech-venv), MOS_PY = UTMOS/DNSMOS (~/facenv/bin/python), PY = the H3 venv (numpy, torch,
# transformers; the WavLM speaker model and the first UTMOS run download their weights).  A tool whose interpreter is
# missing is skipped with a note; the rest still print.  CPU only; a 60 s clip takes 2-4 minutes.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
F=""; LANGUAGE=""; EXTRA=""; PROMPT=""; RESPELL=""; W=277; O=22; FPS=24; MODEL=small; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --language) LANGUAGE=$2; shift 2;; --extra) EXTRA=$2; shift 2;; --prompt) PROMPT=$2; shift 2;;
    --respell) RESPELL=$2; shift 2;; --window) W=$2; shift 2;; --overlap) O=$2; shift 2;; --fps) FPS=$2; shift 2;;
    --model) MODEL=$2; shift 2;; --out) OUT=$2; shift 2;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    -*) echo "voice-metrics: unknown option $1" >&2; exit 2;;
    *) F=$1; shift;;
  esac
done
[ -n "$F" ] && [ -f "$F" ] || { echo "voice-metrics: clip file required" >&2; exit 2; }
OUT=${OUT:-${F%.mp4}.voice}; mkdir -p "$OUT"
pick() { for c in "$@"; do [ -x "$c" ] && { echo "$c"; return; }; done; }
SPEECH_PY=${SPEECH_PY:-$(pick "$HERE/../.speech-venv/bin/python" "$HOME/speech-venv/bin/python" "$HOME/.speech-venv/bin/python")}
MOS_PY=${MOS_PY:-$(pick "$HOME/facenv/bin/python")}
PY=${PY:-$(pick "$HERE/../../.venv/bin/python" "$HOME/h3-long/.venv/bin/python")}
export CUDA_VISIBLE_DEVICES=   # every tool here runs on the CPU; the GPU stays with the model
NAME=$(basename "${F%.mp4}")
FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$F" 2>/dev/null | head -1)
STEP=$((W - O)); [ -n "$FRAMES" ] && [ "$FRAMES" -gt "$W" ] && WINDOWS=$(( (FRAMES - W + STEP - 1) / STEP + 1 )) || WINDOWS=1
echo "voice-metrics $NAME: $FRAMES frames, $WINDOWS windows of $W (overlap $O) at $FPS fps"

SEG="$OUT/segments.json"
if [ -n "$SPEECH_PY" ]; then
  rm -f "$SEG"
  "$SPEECH_PY" "$HERE/speech_qa.py" "$F" ${LANGUAGE:+--language "$LANGUAGE"} --model "$MODEL" --threads 6 --output "$SEG" 2> "$OUT/speech_qa.err" \
    || { echo "transcript: speech_qa.py failed ($(tail -1 "$OUT/speech_qa.err"))"; SEG=""; }
  [ -s "$SEG" ] && echo "transcript: $(python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['segments']; print(len(s), 'segments;', ' '.join(x['text'].strip() for x in s)[:300])" "$SEG")"
else
  echo "transcript: skipped (no faster-whisper interpreter; set SPEECH_PY)"; SEG=""
fi

if [ -s "$SEG" ] && { [ -n "$EXTRA" ] || [ -n "$PROMPT" ]; }; then
  python3 "$HERE/script_wer.py" --segments "$SEG" ${EXTRA:+--extra "$EXTRA"} ${PROMPT:+--prompt "$PROMPT" --windows "$WINDOWS"} \
    ${RESPELL:+--respell "$RESPELL"} --window-frames "$W" --overlap-frames "$O" --fps "$FPS" --label "script:" 2>&1
elif [ -s "$SEG" ]; then
  echo "script: no --extra or --prompt given, word error rate not scored"
fi

if [ -n "$MOS_PY" ]; then
  if [ -s "$SEG" ]; then
    "$MOS_PY" "$HERE/voice_naturalness.py" "$F" --window "$W" --overlap "$O" --fps "$FPS" --segments "$SEG" --json "$OUT/naturalness_segments.json" > /dev/null 2> "$OUT/naturalness.err" \
      && python3 - "$OUT/naturalness_segments.json" <<'PY' || echo "naturalness by line: failed ($(tail -1 "$OUT/naturalness.err"))"
import collections, json, sys
rows = json.load(open(sys.argv[1]))["segments"]
w = collections.defaultdict(list)
for r in rows: w[r["w"]].append(r)
print("utmos by line, per window:", [round(sum(r["utmos"] for r in v) / len(v), 2) for _, v in sorted(w.items())],
      "| low lines:", " ; ".join(f'w{r["w"]} {r["utmos"]:.2f} "{r["text"].strip()[:30]}"' for r in sorted(rows, key=lambda r: r["utmos"])[:3]))
PY
  fi
  "$MOS_PY" "$HERE/voice_naturalness.py" "$F" --window "$W" --overlap "$O" --fps "$FPS" --json "$OUT/naturalness_windows.json" 2>> "$OUT/naturalness.err" | tail -1 | sed 's/^/utmos by window: /'
else
  echo "naturalness: skipped (no UTMOS interpreter; set MOS_PY)"
fi

if [ -n "$PY" ]; then
  if [ -s "$SEG" ]; then
    nice -n 10 "$PY" "$HERE/speaker_drift.py" "$F" "$SEG" 2> "$OUT/speakers.err" | tail -1 | sed "s/^[^ ]* /speakers (cosine to window 1): /" || echo "speakers: failed ($(tail -1 "$OUT/speakers.err"))"
  fi
  "$PY" "$HERE/audio_seams.py" --step "$(python3 -c "print($STEP / $FPS)")" "$F" 2> "$OUT/seams.err" | tail -1 | sed "s/^[^ ]* /seams: /" || echo "seams: failed ($(tail -1 "$OUT/seams.err"))"
else
  echo "speakers, seams: skipped (no numpy/torch interpreter; set PY)"
fi
echo "files: $OUT/"
