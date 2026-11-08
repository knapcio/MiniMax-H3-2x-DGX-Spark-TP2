#!/usr/bin/env python3
"""Word error rate per continuation window: the Whisper transcript against the dialogue written in each window's prompt.

The spoken lines of an H3 prompt sit in `<d>[Language] ...</d>` tags; a chained clip has one prompt per window
(`continuation_prompts` in the extra JSON, the first window included; without it every window hears the one prompt).
The whole transcript (the `speech_qa.py` segments in order) is aligned word by word to the whole script with the same
Levenshtein distance `speech_qa.py` uses for the clip, and every error is charged to the window whose script word it
touches (an inserted word to the window of the script word heard before it).  Segment timings play no part, so a Whisper
segment that fuses the last line of one window with the first of the next costs nothing by itself; the per-window errors
add up to the clip's own word error rate.  Slips of the tongue, dropped or fused words raise it; it says nothing about how
natural the voice sounds (that is `voice_naturalness.py`).  Any clip whose prompts carry `<d>` tags can be scored; the
window geometry (for the start and end columns) is the one the clip was rendered with.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_qa import words  # noqa: E402

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


def align(expected: list[str], actual: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein alignment of two word lists: ("ok" | "sub", i, j), ("del", i, None) or ("ins", None, j) in order.

    The distance is the one `speech_qa.word_error_rate` counts; on ties the backtrace prefers the diagonal, then a deletion.
    """
    rows, cols = len(expected), len(actual)
    cost = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0] = i
    for j in range(1, cols + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1, cost[i - 1][j - 1] + (expected[i - 1] != actual[j - 1]))
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (expected[i - 1] != actual[j - 1]):
            ops.append(("ok" if expected[i - 1] == actual[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def score(prompts: list[str], segments: list[dict], *, window_frames: int, overlap_frames: int, fps: float) -> dict:
    """Per-window word error rate plus the words that differ, for every window whose prompt has spoken lines."""
    spans = window_spans(len(prompts), window_frames, overlap_frames, fps)
    expected: list[str] = []
    owner: list[int] = []
    scripted = []
    for index, prompt in enumerate(prompts):
        script = words(" ".join(script_lines(prompt)))
        if script:
            scripted.append(index)
            expected.extend(script)
            owner.extend([index] * len(script))
    if not expected:
        return {"windows": [], "mean_word_error_rate": None, "clip_word_error_rate": None,
                "limitations": "no <d> lines in the prompts"}
    actual = words(" ".join(str(segment["text"]) for segment in segments))
    errors = {index: 0 for index in scripted}
    heard = {index: 0 for index in scripted}
    missing = {index: set() for index in scripted}
    extra = {index: set() for index in scripted}
    current = scripted[0]
    for op, i, j in align(expected, actual):
        if i is not None:
            current = owner[i]
        if op == "ok":
            heard[current] += 1
        elif op == "sub":
            errors[current] += 1
            heard[current] += 1
            missing[current].add(expected[i])
            extra[current].add(actual[j])
        elif op == "del":
            errors[current] += 1
            missing[current].add(expected[i])
        else:
            errors[current] += 1
            heard[current] += 1
            extra[current].add(actual[j])
    rows = []
    for index in scripted:
        count = owner.count(index)
        rows.append({"window": index + 1, "start_s": round(spans[index][0], 2), "end_s": round(spans[index][1], 2),
                     "script_words": count, "heard_words": heard[index],
                     "word_error_rate": round(errors[index] / count, 4),
                     "missing": sorted(missing[index]), "extra": sorted(extra[index])})
    mean = sum(row["word_error_rate"] for row in rows) / len(rows)
    return {"windows": rows, "mean_word_error_rate": round(mean, 4),
            "clip_word_error_rate": round(sum(errors.values()) / len(expected), 4),
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
    print(f"{args.label} per-window WER {rates} mean {result['mean_word_error_rate']} clip {result['clip_word_error_rate']}")
    for row in result["windows"]:
        if row["missing"] or row["extra"]:
            print(f"  w{row['window']} missing {row['missing']} heard {row['extra']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
