#!/usr/bin/env python3
"""The same spoken lines of two renders side by side: UTMOS per line, paired by what was said.

Two renders of one script (another seed, another hand-over setting) speak the same lines, but a per-window mean mixes
whoever speaks in that window with however many lines fell into it, and a short "Proszę." scores a point below a long
sentence whatever the arm.  This pairs every transcribed line of render A with the line of render B that says the same
thing (letters only, difflib ratio >= 0.8, each line used once) and prints the per-line and per-window UTMOS difference,
so an arm that keeps the words but loses the voice shows as a sign that holds across lines, not as a shifted mean.

    python3 scripts/voice_pairs.py a.voice/naturalness_segments.json b.voice/naturalness_segments.json [--lines]

The inputs are the `--segments --json` output of `scripts/voice_naturalness.py` (also written by `voice-metrics.sh`).
Pure python.
"""
from __future__ import annotations

import argparse
import difflib
import json
import statistics
import sys
from pathlib import Path


def letters(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalpha())


def pair_lines(a: list[dict], b: list[dict], threshold: float = 0.8) -> list[tuple[dict, dict, float]]:
    """(line of a, best unused line of b, similarity) for every line of a with a match at or above the threshold."""
    used: set[int] = set()
    pairs = []
    for x in a:
        key = letters(str(x.get("text", "")))
        best, best_score = None, 0.0
        for j, y in enumerate(b):
            if j in used:
                continue
            score = difflib.SequenceMatcher(None, key, letters(str(y.get("text", "")))).ratio()
            if score > best_score:
                best, best_score = j, score
        if best is not None and best_score >= threshold:
            used.add(best)
            pairs.append((x, b[best], round(best_score, 3)))
    return pairs


def compare(a: list[dict], b: list[dict], metric: str = "utmos", threshold: float = 0.8) -> dict:
    pairs = pair_lines(a, b, threshold)
    diffs = [float(y[metric]) - float(x[metric]) for x, y, _ in pairs]
    per_window: dict[int, list[float]] = {}
    for (x, _, _), d in zip(pairs, diffs):
        per_window.setdefault(int(x.get("w", 0)), []).append(d)
    return {
        "metric": metric, "lines_a": len(a), "lines_b": len(b), "paired": len(pairs),
        "mean_diff": round(statistics.mean(diffs), 3) if diffs else None,
        "median_diff": round(statistics.median(diffs), 3) if diffs else None,
        "b_better": sum(1 for d in diffs if d > 0), "a_better": sum(1 for d in diffs if d < 0),
        "windows": [{"window": w, "paired": len(v), "mean_diff": round(statistics.mean(v), 3)} for w, v in sorted(per_window.items())],
        "lines": [{"window": int(x.get("w", 0)), "a": float(x[metric]), "b": float(y[metric]), "similarity": s,
                   "text": str(x.get("text", "")).strip()} for (x, y, s) in pairs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="naturalness segments JSON of render A (the baseline)")
    parser.add_argument("b", type=Path, help="naturalness segments JSON of render B")
    parser.add_argument("--metric", default="utmos", help="segment field to compare (utmos, ovrl, sig, bak, f0_std_st)")
    parser.add_argument("--threshold", type=float, default=0.8, help="minimum text similarity for a pair")
    parser.add_argument("--lines", action="store_true", help="print every paired line")
    parser.add_argument("--json", type=Path, help="write the comparison here")
    args = parser.parse_args()
    a = json.loads(args.a.read_text(encoding="utf-8"))["segments"]
    b = json.loads(args.b.read_text(encoding="utf-8"))["segments"]
    result = compare(a, b, args.metric, args.threshold)
    print(f"{result['paired']} paired lines of {result['lines_a']} / {result['lines_b']}: {args.metric} B - A mean "
          f"{result['mean_diff']} median {result['median_diff']}, B better in {result['b_better']}, A better in {result['a_better']}")
    print("per window: " + ", ".join(f"w{w['window']} {w['mean_diff']:+.2f} (n={w['paired']})" for w in result["windows"]))
    if args.lines:
        for line in result["lines"]:
            print(f"  w{line['window']} {line['a']:.2f} -> {line['b']:.2f} {line['text'][:60]}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
