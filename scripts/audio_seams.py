#!/usr/bin/env python3
"""Audio hand-over seams of a stitched clip: the log-mel spectral distance between the 250 ms before and after every
window boundary, and the level jump across it, next to the same distance at every other 250 ms grid point of the clip
(median, 95th percentile, max).  A seam that sits inside the clip's own grid spread is not a cut the ear picks out.

    ~/h3-long/.venv/bin/python scripts/audio_seams.py clip.mp4 [more clips]       # numpy + ffmpeg only
    ~/h3-long/.venv/bin/python scripts/audio_seams.py --step 13.4583 clip.mp4     # window 345, overlap 22

Boundaries start every (window - overlap) / fps seconds (default 277 / 22 / 24 = 10.625 s)."""
import subprocess, sys, numpy as np
def decode(p, sr=16000):
    raw = subprocess.run(["ffmpeg","-v","error","-i",p,"-vn","-ac","1","-ar",str(sr),"-f","f32le","-"],capture_output=True,check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)
def logmel(y, sr=16000, n=400, hop=160, nmel=40):
    frames = np.lib.stride_tricks.sliding_window_view(y, n)[::hop] * np.hanning(n)
    spec = np.abs(np.fft.rfft(frames, axis=1))**2
    f = np.fft.rfftfreq(n, 1/sr); m = 2595*np.log10(1+f/700)
    edges = np.linspace(m.min(), m.max(), nmel+2); fb = np.zeros((nmel, len(f)))
    for i in range(nmel):
        lo, c, hi = edges[i:i+3]; fb[i] = np.clip(np.minimum((m-lo)/(c-lo), (hi-m)/(hi-c)), 0, None)
    return np.log(spec @ fb.T + 1e-8)
def main(p, step_s, win=0.25):
    y = decode(p); M = logmel(y); fps = 100
    def seg(t0, t1): return M[int(t0*fps):int(t1*fps)].mean(0)
    def dist(t): return float(np.linalg.norm(seg(t-win, t) - seg(t, t+win)))
    total = len(y)/16000; bounds = [k*step_s for k in range(1, 6) if k*step_s < total-1]
    grid = [t for t in np.arange(1.0, total-1, 0.25) if min(abs(t-b) for b in bounds) > 0.6]
    d_b = [dist(b) for b in bounds]; d_g = np.array([dist(t) for t in grid])
    rms = lambda t0,t1: 20*np.log10(np.sqrt(np.mean(y[int(t0*16000):int(t1*16000)]**2))+1e-9)
    lvl = [round(float(rms(b, b+0.25)-rms(b-0.25, b)),1) for b in bounds]
    print(p.split('/')[-2], "seam dist", [round(d,1) for d in d_b], "level jump dB", lvl,
          "| grid median %.1f p95 %.1f max %.1f" % (np.median(d_g), np.percentile(d_g,95), d_g.max()))
if __name__ == "__main__":
    argv = sys.argv[1:]; step = 255 / 24
    if argv and argv[0] == "--step":
        step = float(argv[1]); argv = argv[2:]
    for p in argv: main(p, step)
