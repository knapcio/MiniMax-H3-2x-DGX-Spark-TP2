#!/usr/bin/env python3
"""Speaker continuity across continuation windows: a WavLM x-vector (microsoft/wavlm-base-plus-sv, transformers)
per transcribed segment of at least 1 s, the two speakers' references built from window 1 by median pitch
(autocorrelation F0, below 150 Hz = low voice), every later segment assigned to the nearer reference by its x-vector
(the pitch tracker doubles a man's F0 on some lines and would hand his line to the woman), and the mean cosine of
every window's segments to the window-1 reference of the speaker they were assigned to.

    CUDA_VISIBLE_DEVICES= ~/h3-long/.venv/bin/python scripts/speaker_drift.py clip.mp4 speech.json [clip2 json2 ...]

speech.json is the output of scripts/speech_qa.py --output.  Needs torch + transformers (the first run downloads the
model through the Hugging Face hub); runs on the CPU in about a minute per 60 s clip.

Two lines per clip. The last line is the summary that the queue scripts keep (``{speaker: {window: mean cosine}}``).
The line before it is the detail that says how much a window's number is worth: per window the segment count and the
lowest segment, the segments whose pitch label disagreed with their voice (``relabelled``), and for window 1 the
leave-one-out cosine of each of its segments to the rest of the window (the same voice in the same window: a later
window below that floor has drifted, one inside it has not).
"""
import json, subprocess, sys, numpy as np
fe = model = None
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
    global fe, model
    import torch
    if model is None:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        torch.set_num_threads(4)
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
    with torch.no_grad():
        inp = fe(y, sampling_rate=16000, return_tensors="pt")
        e = model(**inp).embeddings[0]
    return torch.nn.functional.normalize(e, dim=-1).numpy()
def _unit(v):
    v = np.asarray(v, dtype=float); return v / (np.linalg.norm(v) + 1e-12)
def summarise(rows):
    """rows: [{"w": window, "spk": pitch label "low"|"high", "e": unit vector}] -> (summary, detail).
    The references are the window-1 rows of each pitch label; when both exist every row goes to the nearer reference
    by cosine (``spk`` is then the voice label), otherwise the pitch label stands.
    summary = {spk: {w: mean cosine to the reference}}; detail = {spk: {"w1_loo": [...], w: {"n", "min", "relabelled"}}}."""
    refs = {}
    for spk in ("low", "high"):
        ref = [np.asarray(r["e"], dtype=float) for r in rows if r["spk"] == spk and r["w"] == 1]
        if ref: refs[spk] = ref
    if len(refs) == 2:
        means = {spk: _unit(np.mean(v, 0)) for spk, v in refs.items()}
        rows = [{**r, "spk": max(means, key=lambda k: float(_unit(r["e"]) @ means[k])), "pitch_spk": r["spk"]} for r in rows]
    out, detail = {}, {}
    for spk, ref in refs.items():
        mean_ref = _unit(np.mean(ref, 0))
        loo = [round(float(_unit(e) @ _unit(np.mean([f for j, f in enumerate(ref) if j != i], 0))), 3)
               for i, e in enumerate(ref)] if len(ref) > 1 else []
        per_w, moved = {}, {}
        for r in rows:
            if r["spk"] == spk:
                per_w.setdefault(int(r["w"]), []).append(float(_unit(r["e"]) @ mean_ref))
                moved[int(r["w"])] = moved.get(int(r["w"]), 0) + (r.get("pitch_spk", spk) != spk)
        if not per_w: continue
        out[spk] = {w: round(float(np.mean(v)), 3) for w, v in sorted(per_w.items())}
        detail[spk] = {"w1_loo": loo, **{w: {"n": len(v), "min": round(float(min(v)), 3), "relabelled": moved[w]} for w, v in sorted(per_w.items())}}
    return out, detail
def run(clip, segjson, step=255/24):
    y = decode(clip); rows = []
    for s in json.load(open(segjson))["segments"]:
        t0, t1 = float(s["start_s"]), float(s["end_s"])
        if t1 - t0 < 1.0: continue
        a = y[int(t0*16000):int(t1*16000)]
        f0 = pitch(a)
        if f0 is None: continue
        rows.append({"w": int(t0 // step) + 1, "spk": "low" if f0 < 150 else "high", "e": emb(a)})
    out, detail = summarise(rows)
    name = clip.split("/")[-2] if "/" in clip else clip
    print(name, "detail", json.dumps(detail))
    print(name, json.dumps(out))
if __name__ == "__main__":
    for clip, seg in zip(sys.argv[1::2], sys.argv[2::2]):
        run(clip, seg)
