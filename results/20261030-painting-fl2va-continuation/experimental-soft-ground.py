#!/usr/bin/env python3
"""Experimental ground-light attenuation for a locked-camera painting video.

The slowly varying luma difference from the first frame is damped below a
caller-adjustable boundary, with a soft transition and a strict correction limit.
This does not track clouds or prove physically correct shadows. It can suppress
real lighting changes and must be reviewed visually. The source is preserved.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageChops, ImageFilter


def lower_mask(width, height, start, end):
    top, bottom = round(height * start), round(height * end)
    if not 0 <= top < bottom <= height:
        raise ValueError("mask transition is invalid")
    column = Image.new("L", (1, height))
    column.putdata([max(0, min(255, round(255 * (y - top) / (bottom - top)))) for y in range(height)])
    return column.resize((width, height))


def correct(reference, current, mask, radius, strength, limit):
    difference = ImageChops.subtract(current, reference, offset=128).filter(ImageFilter.GaussianBlur(radius))
    lookup = [128 - round(max(-limit, min(limit, (value - 128) * strength))) for value in range(256)]
    adjustment = difference.point(lookup)
    adjusted = ImageChops.add(current, adjustment, offset=-128)
    return Image.composite(adjusted, current, mask)


def read_frames(command, frame_bytes):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        while True:
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                break
            if len(chunk) != frame_bytes:
                raise RuntimeError("incomplete video frame")
            yield chunk
        if proc.wait() != 0:
            raise RuntimeError("video decode failed")
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.62, help="top of ground transition as fraction of height")
    parser.add_argument("--end", type=float, default=0.77, help="bottom of ground transition as fraction of height")
    parser.add_argument("--radius", type=int, default=32, help="spatial blur radius in source pixels")
    parser.add_argument("--strength", type=float, default=0.35, help="fraction of smooth deviation to attenuate")
    parser.add_argument("--limit", type=int, default=10, help="maximum change in luma levels")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.source.resolve() == args.output.resolve():
        parser.error("output must be a new file distinct from the source")
    if not 0 <= args.start < args.end <= 1 or args.radius < 1 or not 0 <= args.strength <= 1 or not 1 <= args.limit <= 32:
        parser.error("invalid mask, radius, strength or limit")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,pix_fmt,avg_frame_rate", "-of", "json", str(args.source)],
                           check=True, capture_output=True, text=True)
    info = json.loads(probe.stdout)["streams"][0]
    width, height = info["width"], info["height"]
    if info["pix_fmt"] != "yuv420p" or width % 2 or height % 2:
        raise SystemExit("experiment requires an even-sized yuv420p source")
    mask = lower_mask(width, height, args.start, args.end)
    y_bytes, frame_bytes = width * height, width * height * 3 // 2
    decode = ["ffmpeg", "-v", "error", "-xerror", "-i", str(args.source), "-map", "0:v:0", "-an",
              "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".h3-soft-ground-", suffix=".mp4", dir=args.output.parent)
    os.close(fd)
    temp = Path(name)
    encode = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size",
              f"{width}x{height}", "-framerate", info["avg_frame_rate"], "-i", "pipe:0", "-i", str(args.source),
              "-map", "0:v:0", "-map", "1:a:0?", "-map_metadata", "1", "-c:v", "libx264",
              "-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv420p", "-c:a", "copy",
              "-movflags", "+faststart", "-f", "mp4", str(temp)]
    encoder = None
    try:
        encoder = subprocess.Popen(encode, stdin=subprocess.PIPE)
        reference = None
        count = 0
        for frame in read_frames(decode, frame_bytes):
            current = Image.frombytes("L", (width, height), frame[:y_bytes])
            if reference is None:
                reference = current.copy()
            adjusted = correct(reference, current, mask, args.radius, args.strength, args.limit)
            encoder.stdin.write(adjusted.tobytes() + frame[y_bytes:])
            count += 1
        encoder.stdin.close()
        if encoder.wait() != 0 or count == 0:
            raise RuntimeError("video encode failed or no frames decoded")
        os.link(temp, args.output)  # atomic, refuses an existing output
    except BaseException:
        if encoder is not None:
            try:
                encoder.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
            if encoder.poll() is None:
                encoder.terminate()
            encoder.wait()
        raise
    finally:
        temp.unlink(missing_ok=True)
    print(f"Experimental soft-ground copy: {count} frames, strength {args.strength}, "
          f"maximum +/-{args.limit} luma; copied chroma/audio. Visual review required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

