#!/usr/bin/env python3
"""Per-window naturalness of the speech in a clip: UTMOS22 (a MOS predictor trained on synthetic-speech ratings,
torch.hub tarepan/SpeechMOS), DNSMOS (OVRL / SIG / BAK, the pip package speechmos) and the pitch spread (pyin F0 in
semitones) for every continuation window of a stitched clip.  DNSMOS rates signal quality and barely moves when a voice
turns flat or synthetic; UTMOS does, which is why both are printed.

Windows of `--window` frames with an `--overlap` frame hand-over start every (window - overlap) / fps seconds, the
default 277 / 22 / 24 being the H3 long-video defaults.  Runs on the CPU in a few seconds per window.

    ~/facenv/bin/python scripts/voice_naturalness.py clip.mp4            # one JSON line per window + a summary line
    ~/facenv/bin/python scripts/voice_naturalness.py clip.mp4 --window 345
    ~/facenv/bin/python scripts/voice_naturalness.py clip.mp4 --segments speech.json   # one line per spoken segment

`--segments` takes the JSON of `scripts/speech_qa.py --output` and scores every transcribed segment on its own (with
the window it starts in and its median pitch), which separates "this window sounds worse" from "this speaker or this
line sounds worse": a window's score is the score of whoever speaks in it.

Needs ffmpeg, torch + torchaudio, librosa, speechmos (pip) in the interpreter; the first run downloads the UTMOS weights
through torch.hub.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def decode(path: str, sr: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.clip(np.frombuffer(raw, dtype=np.float32), -1.0, 1.0)


def load_models():
    import torch
    torch.set_num_threads(4)
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    # the hub repo's package is also called speechmos: drop it so the pip speechmos (DNSMOS) imports afterwards
    for k in [k for k in sys.modules if k == "speechmos" or k.startswith("speechmos.")]:
        del sys.modules[k]
    from speechmos import dnsmos
    return torch, utmos, dnsmos


def score(torch, utmos, dnsmos, a16: np.ndarray, a22: np.ndarray) -> dict:
    import librosa
    with torch.no_grad():
        u = float(utmos(torch.from_numpy(a16)[None], 16000).item())
    d = dnsmos.run(a16, sr=16000)
    f0, _, _ = librosa.pyin(a22, fmin=65, fmax=500, sr=22050, frame_length=2048)
    f0v = f0[~np.isnan(f0)]
    st = 12 * np.log2(f0v / np.median(f0v)) if len(f0v) > 10 else np.array([0.0])
    return {"utmos": round(u, 2), "ovrl": round(float(d["ovrl_mos"]), 2), "sig": round(float(d["sig_mos"]), 2),
            "bak": round(float(d["bak_mos"]), 2), "voiced": round(float(np.mean(~np.isnan(f0))), 2),
            "f0_med": round(float(np.median(f0v)), 0) if len(f0v) else None, "f0_std_st": round(float(np.std(st)), 2)}


def window_scores(path: str, win: int, ov: int, fps: float) -> list[dict]:
    torch, utmos, dnsmos = load_models()
    y16 = decode(path, 16000)
    y22 = decode(path, 22050)
    total = len(y16) / 16000
    step = (win - ov) / fps
    rows = []
    k = 0
    while k * step < total - 1.0:
        t0 = k * step
        t1 = min(total, t0 + win / fps)
        rows.append({"w": k + 1, "t0": round(t0, 1),
                     **score(torch, utmos, dnsmos, y16[int(t0 * 16000):int(t1 * 16000)],
                             y22[int(t0 * 22050):int(t1 * 22050)])})
        k += 1
    return rows


def segment_scores(path: str, segments: list[dict], win: int, ov: int, fps: float, min_s: float = 1.0) -> list[dict]:
    """Score every transcribed segment of at least `min_s` seconds; `w` is the window the segment starts in."""
    torch, utmos, dnsmos = load_models()
    y16 = decode(path, 16000)
    y22 = decode(path, 22050)
    step = (win - ov) / fps
    rows = []
    for seg in segments:
        t0, t1 = float(seg["start_s"]), float(seg["end_s"])
        if t1 - t0 < min_s:
            continue
        rows.append({"w": int(t0 // step) + 1, "t0": round(t0, 1), "t1": round(t1, 1),
                     **score(torch, utmos, dnsmos, y16[int(t0 * 16000):int(t1 * 16000)],
                             y22[int(t0 * 22050):int(t1 * 22050)]),
                     "text": seg.get("text", "")})
    return rows


def summary(path: str, rows: list[dict]) -> dict:
    u = [r["utmos"] for r in rows]
    s = [r["sig"] for r in rows]
    return {"clip": path, "windows": len(rows), "utmos_w1": u[0], "utmos_min": min(u),
            "utmos_rest_mean": round(float(np.mean(u[1:])), 2) if len(u) > 1 else None,
            "sig_w1": s[0], "sig_rest_mean": round(float(np.mean(s[1:])), 2) if len(s) > 1 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip")
    ap.add_argument("--window", type=int, default=277, help="window length in frames (default 277)")
    ap.add_argument("--overlap", type=int, default=22, help="hand-over overlap in frames (default 22)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--segments", help="speech_qa.py JSON: score every transcribed segment instead of every window")
    ap.add_argument("--json", help="write the rows and the summary to this file")
    a = ap.parse_args()
    if a.segments:
        with open(a.segments, encoding="utf-8") as f:
            rows = segment_scores(a.clip, json.load(f)["segments"], a.window, a.overlap, a.fps)
        for r in rows:
            print(json.dumps(r, ensure_ascii=False), flush=True)
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump({"segments": rows}, f, indent=1, ensure_ascii=False)
        return 0
    rows = window_scores(a.clip, a.window, a.overlap, a.fps)
    for r in rows:
        print(json.dumps(r), flush=True)
    summ = summary(a.clip, rows)
    print(json.dumps(summ))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"windows": rows, "summary": summ}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
