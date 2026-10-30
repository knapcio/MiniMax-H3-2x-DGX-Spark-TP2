#!/usr/bin/env python3
"""Experimental local-luma repair of the detected 17-frame flicker; preserves the input file.

Only the frames selected by experimental-deflicker.py are changed. The correction is a
blurred, capped per-pixel difference from the neighboring clean frames, not motion
compensation. It can change real light/motion and must be judged by eye.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageChops, ImageFilter


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("global_repair", HERE / "experimental-deflicker.py")
global_repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(global_repair)


def correct_luma(before, current, after, fraction, radius=16, limit=8):
    expected = Image.blend(before, after, fraction)
    delta = ImageChops.subtract(expected, current, offset=128).filter(ImageFilter.GaussianBlur(radius))
    # A spatially smoothed signed adjustment in [-limit, limit], centered on 128.
    delta = delta.point([max(128 - limit, min(128 + limit, x)) for x in range(256)])
    return ImageChops.add(current, delta, offset=-128)


def frames(command, frame_bytes):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if not buf:
                break
            if len(buf) != frame_bytes:
                raise RuntimeError("incomplete raw video frame")
            yield buf
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg decoding failed")
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.source.resolve() == args.output.resolve():
        parser.error("output must be a new file distinct from input")
    if args.radius < 1 or not 1 <= args.limit <= 32:
        parser.error("radius must be positive and limit must be 1..32")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,pix_fmt,avg_frame_rate", "-of", "json", str(args.source)],
                           check=True, capture_output=True, text=True)
    info = json.loads(probe.stdout)["streams"][0]
    w, h = info["width"], info["height"]
    if info["pix_fmt"] != "yuv420p" or w % 2 or h % 2:
        raise SystemExit("experiment requires an even-sized yuv420p source")
    groups, corrections = global_repair.make_adjustments(global_repair.measurements(args.source))
    if not groups or not corrections:
        raise SystemExit("no qualifying repeated drops; no file created")
    events = {frame + off: (frame - 1, frame + 5, (off + 1) / 6)
              for group in groups for frame, _ in group for off in range(5) if frame + off in corrections}
    endpoint_indices = sorted({i for left, right, _ in events.values() for i in (left, right)})
    select = "+".join(f"eq(n\\,{i})" for i in endpoint_indices)
    y_bytes, frame_bytes = w * h, w * h * 3 // 2
    decoded = ["ffmpeg", "-v", "error", "-i", str(args.source), "-map", "0:v:0", "-an"]
    raw = ["-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"]
    endpoints = {}
    for index, buf in zip(endpoint_indices, frames(decoded + ["-vf", f"select={select}"] + raw, frame_bytes), strict=True):
        endpoints[index] = Image.frombytes("L", (w, h), buf[:y_bytes])
    encode = ["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size",
              f"{w}x{h}", "-framerate", info["avg_frame_rate"], "-i", "pipe:0", "-i", str(args.source),
              "-map", "0:v:0", "-map", "1:a:0?", "-map_metadata", "1", "-c:v", "libx264", "-preset", "veryfast",
              "-crf", "0", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(args.output)]
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE)
    try:
        count = 0
        for index, buf in enumerate(frames(decoded + raw, frame_bytes)):
            if index in events:
                left, right, fraction = events[index]
                y = Image.frombytes("L", (w, h), buf[:y_bytes])
                y = correct_luma(endpoints[left], y, endpoints[right], fraction, args.radius, args.limit)
                buf = y.tobytes() + buf[y_bytes:]
                count += 1
            encoder.stdin.write(buf)
        encoder.stdin.close()
        if encoder.wait() != 0 or count != len(events):
            raise RuntimeError("ffmpeg encoding failed or selected frames were missing")
    except BaseException:
        try:
            encoder.stdin.close()
        except OSError:
            pass
        if encoder.poll() is None:
            encoder.terminate()
        encoder.wait()
        args.output.unlink(missing_ok=True)
        raise
    print(f"Experimental spatial luma repair: {count} frames, radius {args.radius}, limit +/-{args.limit}; "
          "chroma and audio copied, lossless video encoding. Visual review required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
