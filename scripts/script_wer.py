#!/usr/bin/env python3
"""Word error rate per continuation window: the Whisper transcript against the dialogue written in each window's prompt.

The spoken lines of an H3 prompt sit in `<d>[Language] ...</d>` tags; a chained clip has one prompt per window
(`continuation_prompts` in the extra JSON, the first window included; without it every window hears the one prompt).  Each Whisper segment (from `speech_qa.py`) is assigned to the
window that holds its midpoint, and the window's transcript is scored against the window's script with the same
Levenshtein word error rate `speech_qa.py` uses for the whole clip.  Slips of the tongue, dropped or fused words raise it;
it says nothing about how natural the voice sounds (that is `voice_naturalness.py`).  Any clip whose prompts carry `<d>`
tags can be scored; the window geometry is the one the clip was rendered with.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_qa import word_error_rate, words  # noqa: E402

DIALOGUE = re.compile(r"<d>\s*(?:\[[^\]]*\])?\s*(.*?)</d>", flags=re.S)


def script_lines(prompt: str) -> list[str]:
    """The spoken lines of one prompt, in order."""
    return [" ".join(line.split()) for line in DIALOGUE.findall(prompt)]


def window_spans(windows: int, window_frames: int, overlap_frames: int, fps: float) -> list[tuple[float, float]]:
    """Start and end in seconds of each window's own frames (the overlap belongs to the earlier window)."""
    if windows < 1 or window_frames <= overlap_frames or overlap_frames < 0 or fps <= 0:
        raise ValueError("windows >= 1, 0 <= overlap < window frames and fps > 0 are required")
    step = window_frames - overlap_frames
    spans = []
    for index in range(windows):
        start = index * step
        end = start + (window_frames if index == windows - 1 else step)
        spans.append((start / fps, end / fps))
    return spans


def assign(segments: list[dict], spans: list[tuple[float, float]]) -> list[list[str]]:
    """Each segment goes to the window that holds its midpoint; a segment past the last window goes to the last one."""
    texts: list[list[str]] = [[] for _ in spans]
    for segment in segments:
        middle = (float(segment["start_s"]) + float(segment["end_s"])) / 2
        index = next((i for i, (start, end) in enumerate(spans) if start <= middle < end), len(spans) - 1)
        texts[index].append(segment["text"])
    return texts


def score(prompts: list[str], segments: list[dict], *, window_frames: int, overlap_frames: int, fps: float) -> dict:
    """Per-window word error rate plus the words that differ, for every window whose prompt has spoken lines."""
    spans = window_spans(len(prompts), window_frames, overlap_frames, fps)
    heard = assign(segments, spans)
    rows = []
    for index, prompt in enumerate(prompts):
        reference = " ".join(script_lines(prompt))
        if not words(reference):
            continue
        transcript = " ".join(heard[index])
        expected, actual = words(reference), words(transcript)
        rows.append({"window": index + 1, "start_s": round(spans[index][0], 2), "end_s": round(spans[index][1], 2),
                     "script_words": len(expected), "heard_words": len(actual),
                     "word_error_rate": round(word_error_rate(reference, transcript), 4),
                     "missing": sorted(set(expected) - set(actual)), "extra": sorted(set(actual) - set(expected))})
    mean = sum(row["word_error_rate"] for row in rows) / len(rows) if rows else None
    return {"windows": rows, "mean_word_error_rate": None if mean is None else round(mean, 4),
            "limitations": "Whisper mishears too; compare arms rendered from the same script, not absolute values."}


def load_prompts(prompt_file: Path | None, extra_json: Path | None, windows: int) -> list[str]:
    """`continuation_prompts` (one per window, the first included) when present, else the prompt repeated per window."""
    if extra_json is not None:
        extra = json.loads(extra_json.read_text(encoding="utf-8"))
        more = extra.get("continuation_prompts")
        if more:
            if not isinstance(more, list) or not all(isinstance(item, str) for item in more):
                raise ValueError("continuation_prompts must be a list of strings")
            return list(more)
    if prompt_file is None:
        raise ValueError("--prompt is required when the extra JSON has no continuation_prompts")
    return [prompt_file.read_text(encoding="utf-8")] * windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", required=True, type=Path, help="speech_qa.py JSON output for the clip")
    parser.add_argument("--prompt", type=Path, help="the prompt text file (used when the extra JSON has no per-window prompts)")
    parser.add_argument("--extra", type=Path, help="extra JSON with continuation_prompts (one per window)")
    parser.add_argument("--windows", type=int, default=1, help="window count when only --prompt is given")
    parser.add_argument("--window-frames", type=int, default=277)
    parser.add_argument("--overlap-frames", type=int, default=22)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    data = json.loads(args.segments.read_text(encoding="utf-8"))
    result = score(load_prompts(args.prompt, args.extra, args.windows), data["segments"], window_frames=args.window_frames,
                   overlap_frames=args.overlap_frames, fps=args.fps)
    rates = [row["word_error_rate"] for row in result["windows"]]
    print(f"{args.label} per-window WER {rates} mean {result['mean_word_error_rate']}")
    for row in result["windows"]:
        if row["missing"] or row["extra"]:
            print(f"  w{row['window']} missing {row['missing']} heard {row['extra']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
