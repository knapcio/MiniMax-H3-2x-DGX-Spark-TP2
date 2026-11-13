#!/usr/bin/env python3
"""A still as the pinned first frame of a request: the ``source_video`` and ``video_noise_mask`` files of upstream's
latent-mask editing, made for one shot.

Why: a Ref2VA shot opens on its scene-frame reference's composition and cuts to the requested framing on its own
(``results/20261110-shots-and-close-ups/`` section 5), and a card-only shot opens static but in an invented room.
Pinning latent frame 0 to a still of the real room at the shot's framing (the model's own frame from an earlier shot)
and giving only the character's card as a reference asks for both at once. The server resizes the source to the
request and clones its last frame to the request's length (upstream ``prepare_edit_video``), so the source can be
one frame long. The mask is the token grid (latent frames x height/32 x width/32): 0 keeps the source, 1 generates.

  python scripts/first_frame_pin.py still.png --frames 192 --width 1344 --height 768 --out-dir pin
  H3_SOURCE_VIDEO=pin/source.mp4 H3_VIDEO_NOISE_MASK=pin/mask.json TASK=ref2va H3_REF_IMAGES=card.png bash scripts/render.sh ...

With ``--last other.png`` the last latent frame is pinned too (the FL2VA two-keyframe shot of
``results/20261110-shots-and-close-ups/`` section 8 on the Ref2VA path, so the character's card can come along): the
source is then the request's full length, the first still up to the last four frames and the other still on them
(the last latent covers four frames), and the mask keeps both ends.

Needs ffmpeg for the source file; the mask needs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

GRID = 17


def latent_frames(frames: int) -> int:
    """Latent frames of a 17n+5 request (the first latent covers one frame, every later one four)."""
    if frames < 5 or (frames - 5) % GRID:
        raise ValueError(f"frames must be 17n+5, got {frames}")
    return (frames - 5) // GRID * 5 + 2


def mask(latent_t: int, height: int, width: int, pinned: int = 1, pinned_tail: int = 0) -> list:
    """The token-grid mask: ``pinned`` leading and ``pinned_tail`` trailing latent frames at 0 (keep the source), the
    rest at 1 (generate)."""
    if height % 32 or width % 32:
        raise ValueError("height and width must be multiples of 32")
    if not 0 < pinned <= latent_t or pinned_tail < 0 or pinned + pinned_tail > latent_t:
        raise ValueError(f"pinned latent frames must be in 1..{latent_t} with the tail inside the rest")
    rows, cols = height // 32, width // 32
    return [[[0.0 if (t < pinned or t >= latent_t - pinned_tail) else 1.0] * cols for _ in range(rows)]
            for t in range(latent_t)]


def write_source(still: str, out: str, width: int, height: int, seconds: float = 1.0) -> None:
    """The still as a short H.264 clip at the request's size (the server pads it to the request's length)."""
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1", "-framerate", "24", "-i", still,
                    "-t", f"{seconds:g}", "-vf", f"scale={width}:{height}:flags=lanczos,format=yuv420p",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-an", out], check=True)


def write_source_both(first: str, last: str, out: str, width: int, height: int, frames: int, tail_frames: int) -> None:
    """The request's full length: ``first`` on frames 0..frames-tail_frames-1 and ``last`` on the final ``tail_frames``
    (what the pinned last latent covers), at the request's size."""
    if not 0 < tail_frames < frames:
        raise ValueError("the tail must be inside the clip")
    head = frames - tail_frames
    vf = f"scale={width}:{height}:flags=lanczos,format=yuv420p,setsar=1"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                    "-loop", "1", "-framerate", "24", "-i", first, "-loop", "1", "-framerate", "24", "-i", last,
                    "-filter_complex", f"[0:v]{vf},trim=end_frame={head},setpts=PTS-STARTPTS[a];"
                                       f"[1:v]{vf},trim=end_frame={tail_frames},setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0",
                    "-frames:v", str(frames), "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-an", out], check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("still")
    ap.add_argument("--frames", type=int, required=True, help="the request's frame count (17n+5)")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--pin-latents", type=int, default=1, help="leading latent frames to pin (default 1 = the first frame)")
    ap.add_argument("--last", default="", help="a still for the last latent frame too (the two-keyframe shot)")
    ap.add_argument("--last-latents", type=int, default=1, help="trailing latent frames to pin with --last (default 1 = four frames)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    latent_t = latent_frames(args.frames)
    os.makedirs(args.out_dir, exist_ok=True)
    source = os.path.join(args.out_dir, "source.mp4")
    mask_path = os.path.join(args.out_dir, "mask.json")
    tail = args.last_latents if args.last else 0
    if args.last:
        write_source_both(args.still, args.last, source, args.width, args.height, args.frames, 4 * tail)
    else:
        write_source(args.still, source, args.width, args.height)
    with open(mask_path, "w", encoding="utf-8") as f:
        json.dump(mask(latent_t, args.height, args.width, args.pin_latents, tail), f)
    print(f"{args.frames} frames = {latent_t} latent frames, {args.pin_latents} pinned"
          + (f" + the last {tail}" if tail else "") + f"; mask {latent_t}x{args.height // 32}x{args.width // 32}")
    print(f"H3_SOURCE_VIDEO={source} H3_VIDEO_NOISE_MASK={mask_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
