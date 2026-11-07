#!/usr/bin/env python3
"""Speaker continuity across continuation windows: a WavLM x-vector (microsoft/wavlm-base-plus-sv, transformers)
per transcribed segment of at least 1 s, speakers split by median pitch (autocorrelation F0, below 150 Hz = low
voice), and the mean cosine of every window's segments to the window-1 segments of the same speaker.

    CUDA_VISIBLE_DEVICES= ~/h3-long/.venv/bin/python scripts/speaker_drift.py clip.mp4 speech.json [clip2 json2 ...]

speech.json is the output of scripts/speech_qa.py --output.  Needs torch + transformers (the first run downloads the
model through the Hugging Face hub); runs on the CPU in about a minute per 60 s clip."""
import json, subprocess, sys, numpy as np, torch
from transformers import AutoFeatureExtractor, WavLMForXVector
torch.set_num_threads(4)
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
def decode(p, sr=16000):
    raw = subprocess.run(["ffmpeg","-v","error","-i",p,"-vn","-ac","1","-ar",str(sr),"-f","f32le","-"],capture_output=True,check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
def pitch(y, sr=16000, n=640, hop=160):
    """median F0 of voiced 40 ms frames by normalised autocorrelation (lags for 65-500 Hz)"""
    fr = np.lib.stride_tricks.sliding_window_view(y, n)[::hop]; fr = fr - fr.mean(1, keepdims=True)
    f0s = []
    for x in fr:
        if np.sqrt(np.mean(x**2)) < 0.01: continue
        ac = np.correlate(x, x, "full")[n-1:]; ac = ac / (ac[0] + 1e-9)
        lo, hi = sr // 500, sr // 65
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.6: f0s.append(sr / k)
    return float(np.median(f0s)) if len(f0s) >= 10 else None
def emb(y):
    with torch.no_grad():
        inp = fe(y, sampling_rate=16000, return_tensors="pt")
        e = model(**inp).embeddings[0]
    return torch.nn.functional.normalize(e, dim=-1).numpy()
def run(clip, segjson, step=255/24):
    y = decode(clip); rows = []
    for s in json.load(open(segjson))["segments"]:
        t0, t1 = float(s["start_s"]), float(s["end_s"])
        if t1 - t0 < 1.0: continue
        a = y[int(t0*16000):int(t1*16000)]
        f0 = pitch(a)
        if f0 is None: continue
        rows.append({"w": int(t0 // step) + 1, "spk": "low" if f0 < 150 else "high", "e": emb(a)})
    out = {}
    for spk in ("low", "high"):
        ref = [r["e"] for r in rows if r["spk"] == spk and r["w"] == 1]
        if not ref: continue
        ref = np.mean(ref, 0); ref /= np.linalg.norm(ref)
        per_w = {}
        for r in rows:
            if r["spk"] == spk: per_w.setdefault(r["w"], []).append(float(r["e"] @ ref))
        out[spk] = {w: round(float(np.mean(v)), 3) for w, v in sorted(per_w.items())}
    print(clip.split("/")[-2], json.dumps(out))
if __name__ == "__main__":
    for clip, seg in zip(sys.argv[1::2], sys.argv[2::2]):
        run(clip, seg)
