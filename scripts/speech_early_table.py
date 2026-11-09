#!/usr/bin/env python3
"""At which denoising step is a slip of the tongue already audible?  Reads the head journal of renders made with
`H3_TP2_SPEECH_GUARD` and `H3_TP2_SPEECH_EARLY=<steps>` and prints, for every window sampling, the character error
rate the early checks heard at each step next to the rate of the finished window, then for every step and candidate
limit how many of the samplings that failed in the end the check would have stopped, and how many of the ones that
passed it would have stopped by mistake.  Use it to choose `H3_TP2_SPEECH_EARLY_STOP=<step>:<limit>`.

    journalctl --user -u h3-tp2-head.service --since "-3 h" --no-pager | python3 scripts/speech_early_table.py
    python3 scripts/speech_early_table.py journal.txt --final-limit 0.01 --limits 0.01,0.02,0.03,0.05 --json out.json

Pure Python, no GPU.  A request ends at its `speech guard summary` line.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

EARLY = re.compile(r"speech early check window (\d+) attempt (\d+) step (\d+) of (\d+): CER ([0-9.]+) \((\d+) of (\d+) chars\)")
FINAL = re.compile(r"speech check window (\d+) attempt (\d+) \(seed (\d+)\): CER ([0-9.]+)")
SUMMARY = re.compile(r"speech guard summary:")


def parse(lines) -> list[dict]:  # noqa: ANN001
    """One record per window sampling: {"request", "window", "attempt", "seed", "early": {step: cer}, "final": cer}."""
    records: dict[tuple[int, int, int], dict] = {}
    request = 1

    def record(window: int, attempt: int) -> dict:
        key = (request, window, attempt)
        if key not in records:
            records[key] = {"request": request, "window": window, "attempt": attempt, "seed": None, "early": {}, "final": None}
        return records[key]

    for line in lines:
        m = EARLY.search(line)
        if m:
            record(int(m.group(1)), int(m.group(2)))["early"][int(m.group(3))] = float(m.group(5))
            continue
        m = FINAL.search(line)
        if m:
            r = record(int(m.group(1)), int(m.group(2)))
            r["seed"], r["final"] = int(m.group(3)), float(m.group(4))
            continue
        if SUMMARY.search(line):
            request += 1
    return [records[k] for k in sorted(records)]


def verdicts(records: list[dict], final_limit: float, limits: list[float]) -> list[dict]:
    """For every step and limit: of the samplings with a final rate, how many failing ones (final > final_limit) and
    how many passing ones the check at that step would have stopped (early rate > limit)."""
    judged = [r for r in records if r["final"] is not None]
    steps = sorted({s for r in judged for s in r["early"]})
    out = []
    for step in steps:
        heard = [r for r in judged if step in r["early"]]
        failing = [r for r in heard if r["final"] > final_limit]
        passing = [r for r in heard if r["final"] <= final_limit]
        for limit in limits:
            out.append({"step": step, "limit": limit,
                        "failing": len(failing), "stopped_failing": sum(r["early"][step] > limit for r in failing),
                        "passing": len(passing), "stopped_passing": sum(r["early"][step] > limit for r in passing)})
    return out


def table(records: list[dict]) -> str:
    steps = sorted({s for r in records for s in r["early"]})
    head = "request window attempt " + " ".join(f"{'s' + str(s):>6}" for s in steps) + "  final"
    rows = [head]
    for r in records:
        cells = " ".join(f"{r['early'][s]:6.3f}" if s in r["early"] else "     -" for s in steps)
        final = f"{r['final']:6.3f}" if r["final"] is not None else "     -"
        rows.append(f"{r['request']:>7} {r['window']:>6} {r['attempt']:>7} {cells} {final}")
    return "\n".join(rows)


def main(argv=None) -> int:  # noqa: ANN001
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("journal", nargs="?", help="journal text (default: standard input)")
    ap.add_argument("--final-limit", type=float, default=0.01, help="the guard's limit on the finished window (default 0.01)")
    ap.add_argument("--limits", default="0.01,0.02,0.03,0.05,0.1", help="candidate limits of the early check")
    ap.add_argument("--json", help="write records and verdicts here")
    args = ap.parse_args(argv)
    if args.journal:
        with open(args.journal, encoding="utf-8", errors="replace") as f:
            records = parse(f)
    else:
        records = parse(sys.stdin)
    if not records:
        print("no speech check lines found (render with H3_TP2_SPEECH_GUARD and H3_TP2_SPEECH_EARLY=<steps>)")
        return 1
    limits = [float(x) for x in args.limits.split(",") if x.strip()]
    result = verdicts(records, args.final_limit, limits)
    print(table(records))
    print()
    print(f"samplings the finished window fails (> {args.final_limit:g}) / passes, stopped by a check at step S above limit L:")
    for v in result:
        print(f"  step {v['step']:>2} limit {v['limit']:<5g}: stops {v['stopped_failing']} of {v['failing']} failing, "
              f"{v['stopped_passing']} of {v['passing']} passing")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"final_limit": args.final_limit, "records": records, "verdicts": result}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
