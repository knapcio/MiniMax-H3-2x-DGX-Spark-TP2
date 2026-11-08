#!/usr/bin/env python3
"""Lint the dialogue of a continuation request before rendering it: words per window and per second of new time,
windows without lines, language tags that differ between lines, a `<d>` tag left open, a line repeated in another
window.

Every window of a chained clip has (window - overlap) / fps seconds of its own (10.6 s with the 277 / 22 / 24 defaults)
and the model has to fit that window's lines into them; the delivered clips carried 18-28 words in three lines per
window (1.7-2.6 words per second) and every line was present and in order.  More words than the budget and the model
speaks faster, drops a pause or a word.  The lint reports, it does not change anything.

    python3 scripts/dialogue_lint.py --extra extra.json [--budget 30] [--window 277 --overlap 22 --fps 24]
    python3 scripts/dialogue_lint.py --prompt prompt.txt --windows 1       # a single window

Exit status 1 when a window is over the budget or a tag is malformed, 0 otherwise (a CI-style gate for a batch).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DIALOGUE = re.compile(r"<d>\s*(?:\[([^\]]*)\])?\s*(.*?)</d>", flags=re.S)
OPEN_TAG = re.compile(r"<d>")
CLOSE_TAG = re.compile(r"</d>")


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text, flags=re.UNICODE)


def lint_prompt(prompt: str) -> dict:
    """Lines, words, language tags and tag problems of one window's prompt."""
    lines = [(tag.strip() if tag else "", " ".join(text.split())) for tag, text in DIALOGUE.findall(prompt)]
    problems = []
    if len(OPEN_TAG.findall(prompt)) != len(CLOSE_TAG.findall(prompt)):
        problems.append("unbalanced <d> tags")
    tags = {tag for tag, _ in lines if tag}
    if len(tags) > 1:
        problems.append(f"mixed language tags {sorted(tags)}")
    if any(not tag for tag, _ in lines) and tags:
        problems.append("a line without a language tag next to tagged lines")
    empty = sum(1 for _, text in lines if not words(text))
    if empty:
        problems.append(f"{empty} empty line(s)")
    return {"lines": len(lines), "words": sum(len(words(text)) for _, text in lines), "tags": sorted(tags),
            "texts": [text for _, text in lines], "problems": problems}


def lint(prompts: list[str], *, window_frames: int, overlap_frames: int, fps: float, budget: int) -> dict:
    step_s = (window_frames - overlap_frames) / fps
    rows = []
    seen: dict[str, int] = {}
    for index, prompt in enumerate(prompts):
        r = lint_prompt(prompt)
        own_s = window_frames / fps if index == 0 else step_s
        r.update({"window": index + 1, "seconds": round(own_s, 2), "words_per_second": round(r["words"] / own_s, 2),
                  "over_budget": r["words"] > budget})
        for text in r["texts"]:
            key = " ".join(words(text)).casefold()
            if key in seen and key:
                r["problems"].append(f"line also in window {seen[key]}: {text[:50]!r}")
            seen.setdefault(key, index + 1)
        rows.append(r)
    spoken = [r for r in rows if r["lines"]]
    return {"windows": rows, "budget": budget, "step_seconds": round(step_s, 3),
            "total_words": sum(r["words"] for r in rows), "silent_windows": [r["window"] for r in rows if not r["lines"]],
            "ok": not any(r["over_budget"] or r["problems"] for r in rows), "spoken_windows": len(spoken)}


def load_prompts(prompt_file: Path | None, extra_json: Path | None, windows: int) -> list[str]:
    if extra_json is not None:
        extra = json.loads(extra_json.read_text(encoding="utf-8"))
        more = extra.get("continuation_prompts")
        if more:
            return [str(p) for p in more]
    if prompt_file is None:
        raise ValueError("--prompt is required when the extra JSON has no continuation_prompts")
    return [prompt_file.read_text(encoding="utf-8")] * windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extra", type=Path, help="extra JSON with continuation_prompts (one per window)")
    parser.add_argument("--prompt", type=Path, help="one prompt for every window")
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--window", type=int, default=277, help="window frames")
    parser.add_argument("--overlap", type=int, default=22, help="hand-over overlap frames")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--budget", type=int, default=30, help="words per window above which the lint warns")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = lint(load_prompts(args.prompt, args.extra, args.windows), window_frames=args.window, overlap_frames=args.overlap,
                  fps=args.fps, budget=args.budget)
    for r in result["windows"]:
        flag = " OVER BUDGET" if r["over_budget"] else ""
        tags = ",".join(r["tags"]) or "-"
        print(f"w{r['window']}: {r['lines']} line(s), {r['words']} words in {r['seconds']} s ({r['words_per_second']} w/s) [{tags}]{flag}"
              + (" | " + "; ".join(r["problems"]) if r["problems"] else ""))
    print(f"{len(result['windows'])} windows, {result['total_words']} words, budget {result['budget']} per window, "
          f"silent {result['silent_windows'] or 'none'}: {'ok' if result['ok'] else 'problems'}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
