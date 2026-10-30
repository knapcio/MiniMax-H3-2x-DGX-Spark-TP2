#!/usr/bin/env python3
"""Experimental local-luma repair of the detected 17-frame flicker; preserves the input file.

By default only the frames selected by experimental-deflicker.py are changed.
--local-periodic is a more invasive experiment on the dominant repeated regional
phase. Corrections use neighboring frames, not motion compensation. They can
change real light/motion and must be judged by eye.
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


def regional_events(source):
    detector_path = HERE.parents[1] / "scripts/regional_flicker.py"
    detector_spec = importlib.util.spec_from_file_location("regional_flicker", detector_path)
    detector = importlib.util.module_from_spec(detector_spec)
    detector_spec.loader.exec_module(detector)
    _, frames, series = detector.inspect(source)
    return regional_events_from_series(frames, series, detector.periodic_spikes)


def regional_events_from_series(frames, series, find_spikes):
    groups = [group for changes in series for group in find_spikes(changes)]
    if not groups:
        return {}
    scores = {}
    for group in groups:
        phase = group[0][0] % 17
        scores[phase] = scores.get(phase, 0) + len(group)
    phase = max(scores, key=scores.get)
    boundaries = sorted({entry[0] for group in groups if group[0][0] % 17 == phase for entry in group
                         if entry[0] > 0 and entry[0] + 5 < frames})
    return {frame + off: (frame - 1, frame + 5, (off + 1) / 6) for frame in boundaries for off in range(5)}


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
    parser.add_argument("--local-periodic", action="store_true",
                        help="experimental: correct five frames at each dominant repeated regional boundary")
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
    if args.local_periodic:
        events = regional_events(args.source)
    else:
        groups, corrections = global_repair.make_adjustments(global_repair.measurements(args.source))
        events = {frame + off: (frame - 1, frame + 5, (off + 1) / 6)
                  for group in groups for frame, _ in group for off in range(5) if frame + off in corrections}
    if not events:
        raise SystemExit("no qualifying repeated changes; no file created")
    y_bytes, frame_bytes = w * h, w * h * 3 // 2
    decoded = ["ffmpeg", "-v", "error", "-i", str(args.source), "-map", "0:v:0", "-an"]
    raw = ["-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"]
    encode = ["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size",
              f"{w}x{h}", "-framerate", info["avg_frame_rate"], "-i", "pipe:0", "-i", str(args.source),
              "-map", "0:v:0", "-map", "1:a:0?", "-map_metadata", "1", "-c:v", "libx264", "-preset", "veryfast",
              "-crf", "0", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(args.output)]
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE)
    try:
        count = 0
        buffered = {}

        def write_frame(index):
            nonlocal count
            buf = buffered[index]
            if index in events:
                left, right, fraction = events[index]
                before = Image.frombytes("L", (w, h), buffered[left][:y_bytes])
                after = Image.frombytes("L", (w, h), buffered[right][:y_bytes])
                y = Image.frombytes("L", (w, h), buf[:y_bytes])
                y = correct_luma(before, y, after, fraction, args.radius, args.limit)
                buf = y.tobytes() + buf[y_bytes:]
                count += 1
            encoder.stdin.write(buf)
            buffered.pop(index - 5, None)

        total = 0
        for index, buf in enumerate(frames(decoded + raw, frame_bytes)):
            buffered[index] = buf
            total += 1
            if index >= 5:
                write_frame(index - 5)
        for index in range(max(0, total - 5), total):
            write_frame(index)
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
    print(f"Experimental spatial luma repair: {count} frames processed, radius {args.radius}, limit +/-{args.limit}; "
          "chroma and audio copied, lossless video encoding. Visual review required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
