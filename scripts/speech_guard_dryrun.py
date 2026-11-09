#!/usr/bin/env python3
"""What the speech guard would have said about a finished clip: every window's audio is cut out of the clip,
transcribed and compared with the `<d>` lines of that window's prompt exactly as `H3_TP2_SPEECH_GUARD` does during a
render (`h3_tp2/speech_guard.py`: character error rate without word boundaries, free speech around the lines).

Use it to pick a limit before paying for re-samples (which windows of clips you already have would have been sampled
again at 0.005, 0.01, 0.02?) and to check a delivered clip window by window.  The window audio comes from the muxed
clip, not from the window's own latents, so a rate can differ by a character from what the live guard measured.

    ~/speech-venv/bin/python scripts/speech_guard_dryrun.py clip.mp4 --extra extra.json [--respell respell.txt]
    ... --prompt prompt.txt             # one prompt for every window
    ... --limits 0.005,0.01,0.02 --json out.json

Run it in the interpreter that has faster-whisper (the one `scripts/speech_qa.py` uses), or pass `--python`.
Needs ffmpeg.  CPU only, 3-4 s per window with the `small` model.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3_tp2 import speech_guard as sg  # noqa: E402


def window_starts(frames_total: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """(start frame, frame count) of every window of a clip of ``frames_total`` frames."""
    if frames_total <= window:
        return [(0, frames_total)]
    step = window - overlap
    out = []
    start = 0
    while start + overlap < frames_total:
        out.append((start, min(window, frames_total - start)))
        start += step
    return out


def frame_count(path: str) -> int:
    res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return int(res.stdout.strip().split("\n")[0])


def cut_wav(clip: str, start_s: float, seconds: float, out: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start_s:.4f}", "-t", f"{seconds:.4f}", "-i", clip, "-vn",
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip")
    parser.add_argument("--extra", type=Path, help="extra JSON with continuation_prompts (one per window)")
    parser.add_argument("--prompt", type=Path, help="one prompt for every window")
    parser.add_argument("--respell", default="", help="`prompt_spelling real_word` lines")
    parser.add_argument("--window", type=int, default=277)
    parser.add_argument("--overlap", type=int, default=22)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--model", default="small", help="faster-whisper model, or several separated by commas (the lowest rate counts)")
    parser.add_argument("--language", default="", help="override the [Language] tag of the lines")
    parser.add_argument("--python", default=sys.executable, help="interpreter with faster-whisper")
    parser.add_argument("--limits", default="0.005,0.01,0.02")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.extra:
        prompts = [str(p) for p in json.loads(args.extra.read_text(encoding="utf-8")).get("continuation_prompts") or []]
    else:
        prompts = []
    if not prompts:
        if not args.prompt:
            parser.error("--prompt is required when the extra JSON has no continuation_prompts")
        prompts = [args.prompt.read_text(encoding="utf-8")]
    here = Path(__file__).resolve().parent
    cfg = sg.Config(max_cer=1.0, python=args.python, script=str(here / "speech_qa.py"), model=args.model,
                    language=args.language.strip().lower(), respell_file=args.respell)
    guard = sg.Guard(cfg, prompts)
    windows = window_starts(frame_count(args.clip), args.window, args.overlap)
    limits = [float(x) for x in args.limits.split(",") if x.strip()]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, (start, count) in enumerate(windows, 1):
            script, language = guard.script(index)
            if len(script) < cfg.min_words:
                rows.append({"window": index, "judged": False})
                print(f"w{index}: no lines, not judged")
                continue
            wav = os.path.join(tmp, f"w{index}.wav")
            cut_wav(args.clip, start / args.fps, count / args.fps, wav)
            wdir = os.path.join(tmp, f"w{index}")
            os.makedirs(wdir, exist_ok=True)
            # limit 0: every judge is asked, the verdict is the lowest rate (what the live guard compares with its limit)
            verdict, heard, heard_words = sg.judge_wav(cfg, script, language, wav, wdir, max_cer=0.0)
            row = {"window": index, "judged": True, "cer": verdict["cer"], "distance": verdict["distance"],
                   "chars": verdict["chars"], "rates": verdict["rates"], "script_words": len(script),
                   "heard_words": len(heard_words), "heard": " ".join(heard).strip()}
            rows.append(row)
            judges = f" judges {verdict['rates']}" if len(verdict["rates"]) > 1 else ""
            print(f"w{index}: CER {row['cer']:.4f} ({row['distance']} of {row['chars']} chars){judges}, script {row['script_words']} "
                  f"words, heard {row['heard_words']}: {row['heard'][:110]}")
    judged = [r for r in rows if r.get("judged")]
    summary = {f"{limit:g}": [r["window"] for r in judged if r["cer"] > limit] for limit in limits}
    print("re-sampled at limit: " + "; ".join(f"{k}: {('w' + ', w'.join(map(str, v))) if v else 'none'}" for k, v in summary.items()))
    if args.json:
        args.json.write_text(json.dumps({"clip": args.clip, "windows": rows, "resampled_at": summary}, indent=1, ensure_ascii=False),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
