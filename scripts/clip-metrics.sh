#!/bin/bash
# Per-second metrics of a continuation clip, for spotting the three decay axes and unrequested cuts:
#   motion  = mean absolute luma change between consecutive frames, per second (drops when the shot freezes)
#   detail  = variance of the Laplacian of the lower half of one frame per second (fine texture; falls when it blurs,
#             rises with streaks or noise, so read it with "coherence")
#   coherence = gradient-orientation coherence of the same half (0 = isotropic texture, 1 = one direction): rising with
#             detail flags directional streaking (idea: astra review round 7)
#   light   = mean luma of one frame per second (0-255; a slow rise or fall is light drift)
#   colour  = mean U and V (chroma, 0-255, 128 = neutral) per 5 s: a drift of more than a few units between windows
#             is a colour change the eye sees and no other metric here catches (owner's 60 s review, 2026-10-29)
#   audio   = mean volume per 10 s (dB)
#   cuts    = frames with an ffmpeg scene score > 0.1 (hard cuts only)
#   1s-change = mean absolute luma change between frame n and frame n+24, per second: a dissolve or a fast state
#             change (fog rolling in, sun going behind a cloud) shows here as a spike without a single-frame cut
#   seams   = luma change at every window hand-over (k*(window-overlap) + overlap) against the 27 frames before it,
#             plus the mean-luma step across the hand-over (3 frames before vs 3 after): the exposure jump the eye sees
# Usage: clip-metrics.sh <clip.mp4> [window_frames=277] [overlap_frames=22]   (needs ffmpeg + the H3 venv's numpy/PIL)
set -euo pipefail
F=$1; W=${2:-277}; O=${3:-22}
PY=${PY:-$(dirname "$0")/../../.venv/bin/python}; command -v "$PY" >/dev/null 2>&1 || PY=python3
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
ffmpeg -loglevel error -i "$F" -vf "fps=1,crop=iw:ih/2:0:ih/2,format=gray" "$T/d%03d.png"
ffmpeg -loglevel error -i "$F" -vf "scale=iw/4:ih/4,format=gray" "$T/m%05d.png"
ffmpeg -loglevel error -i "$F" -vf "fps=1,signalstats,metadata=print:file=$T/yavg.txt" -f null -
ffmpeg -loglevel error -nostats -i "$F" -vf "select=gte(scene\,0),metadata=print:key=lavfi.scene_score:file=$T/scene.txt" -f null -
n=$(ls "$T"/m*.png | wc -l); [ "$n" -gt 24 ] && [ -s "$T/scene.txt" ] && [ -s "$T/d001.png" ] || { echo "clip-metrics: frame extraction failed for $F" >&2; exit 1; }
dur=$(( (n + 23) / 24 ))
echo "audio (mean dB per 10 s):"
ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "$F" | grep -q audio || echo "  (no audio stream)"
for s in $(seq 0 10 $((dur - 6))); do ffmpeg -loglevel info -nostats -ss "$s" -t 10 -i "$F" -vn -af volumedetect -f null - 2>&1 | grep -o "mean_volume: [-0-9.]* dB" | sed "s/mean_volume:/  ${s}s/" || true; done
"$PY" - "$T" "$W" "$O" <<'PY'
import sys, glob, re, numpy as np
from PIL import Image
t, W, O = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
det, coh = [], []
for f in sorted(glob.glob(t + "/d*.png")):
    a = np.asarray(Image.open(f), dtype=np.float32)
    det.append((-4*a[1:-1,1:-1] + a[:-2,1:-1] + a[2:,1:-1] + a[1:-1,:-2] + a[1:-1,2:]).var())
    gx, gy = a[1:-1, 2:] - a[1:-1, :-2], a[2:, 1:-1] - a[:-2, 1:-1]
    jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    coh.append(float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)))
fr = [np.asarray(Image.open(f), dtype=np.float32) for f in sorted(glob.glob(t + "/m*.png"))]
d = [np.abs(fr[i] - fr[i-1]).mean() for i in range(1, len(fr))]
sec = [np.mean(d[s*24:(s+1)*24]) for s in range(len(d) // 24 + (1 if len(d) % 24 else 0))]
print("motion per second: " + " ".join(f"{v:.1f}" for v in sec))
d24 = [np.abs(fr[i + 24] - fr[i]).mean() for i in range(0, len(fr) - 24, 24)]
print("1s-change per second: " + " ".join(f"{v:.0f}" for v in d24))
stats = open(t + "/yavg.txt").read()
print("light per second: " + " ".join(f"{float(v):.0f}" for v in re.findall(r"YAVG=([\d.]+)", stats)))
u = [float(v) for v in re.findall(r"UAVG=([\d.]+)", stats)]; vv = [float(v) for v in re.findall(r"VAVG=([\d.]+)", stats)]
print("colour U/V per 5 s: " + " ".join(f"{np.mean(u[i:i+5]):.0f}/{np.mean(vv[i:i+5]):.0f}" for i in range(0, len(u), 5)))
print("detail per second: " + " ".join(f"{v:.0f}" for v in det))
print("coherence per second: " + " ".join(f"{v:.2f}" for v in coh))
seam = W                      # output switches to window k at k*(W-O) + O: 277, 532, 787 ... for 277/22
while seam < len(fr):
    i = seam - 1
    lb, la = np.mean([fr[j].mean() for j in range(seam-3, seam)]), np.mean([fr[j].mean() for j in range(seam, seam+3)])
    print(f"seam frame {seam} ({seam/24:.1f} s): {d[i-1]:.2f} [{d[i]:.2f}] {d[i+1]:.2f}  (mean of the 27 frames before: {np.mean(d[i-28:i-1]):.2f}); luma step {lb:.1f} -> {la:.1f} ({la-lb:+.1f}, frames -3..+3)")
    seam += W - O
cuts = []
txt = open(t + "/scene.txt").read()
for m in re.finditer(r"frame:(\d+)\s+pts:\d+\s+pts_time:([\d.]+)\s*\n\s*lavfi\.scene_score=([\d.]+)", txt):
    if float(m.group(3)) > 0.1: cuts.append(f"frame {m.group(1)} ({float(m.group(2)):.1f} s) score {m.group(3)}")
print("cuts (scene score > 0.1): " + ("; ".join(cuts) if cuts else "none"))
PY
