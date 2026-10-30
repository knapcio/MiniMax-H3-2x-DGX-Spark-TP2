#!/usr/bin/env python3
"""Compare frame changes in six regions at the known history5s flicker boundaries."""
import statistics
import subprocess
import sys

from PIL import Image, ImageChops, ImageStat

W, H = 336, 192
BOXES = [(col * W // 3, row * H // 2, (col + 1) * W // 3, (row + 1) * H // 2)
         for row in range(2) for col in range(3)]
BAD = set(range(357, 562, 17))


def main(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-an", "-vf",
           f"scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    previous = None
    prev_means = None
    boundary_deltas, boundary_abs, other_abs = [], [], []
    index = 0
    while True:
        buf = proc.stdout.read(W * H)
        if not buf:
            break
        if len(buf) != W * H:
            raise RuntimeError(f"short frame {index}")
        image = Image.frombytes("L", (W, H), buf)
        means = [ImageStat.Stat(image.crop(box)).mean[0] for box in BOXES]
        if previous is not None:
            difference = ImageChops.difference(image, previous)
            absvals = [ImageStat.Stat(difference.crop(box)).mean[0] for box in BOXES]
            if index in BAD:
                boundary_deltas.append([a - b for a, b in zip(means, prev_means)])
                boundary_abs.append(absvals)
            elif 357 <= index <= 561:
                other_abs.append(absvals)
        previous, prev_means = image, means
        index += 1
    if proc.wait() != 0 or len(boundary_deltas) != len(BAD):
        raise RuntimeError("ffmpeg failed or video is missing expected frames")
    for name, rows in (("boundary signed delta", boundary_deltas), ("boundary mean absolute delta", boundary_abs),
                       ("other mean absolute delta", other_abs)):
        medians = [statistics.median(row[k] for row in rows) for k in range(len(BOXES))]
        print(f"{name}: " + " ".join(f"{v:+.2f}" for v in medians))


if __name__ == "__main__":
    main(sys.argv[1])
