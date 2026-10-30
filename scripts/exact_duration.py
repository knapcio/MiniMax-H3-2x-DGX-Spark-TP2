#!/usr/bin/env python3
"""Make a separate H3 MP4 ending at the nearest 24 fps frame, preserving retained frames and AAC packets."""
import argparse
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def probe(path):
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                           "stream=index,codec_type,codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,"
                           "sample_rate,start_time,duration,sample_aspect_ratio,color_range,color_space,color_transfer,color_primaries:format=duration",
                           "-of", "json", str(path)], check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def target_frames(seconds):
    try:
        value = Decimal(str(seconds))
    except InvalidOperation as exc:
        raise ValueError("invalid duration") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError("duration must be finite and positive")
    count = int(Fraction(value) * 24 + Fraction(1, 2))
    if count < 1:
        raise ValueError("duration is shorter than half a frame")
    return count


def audio_packets(path):
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets",
                           "-show_entries", "packet=pts_time,duration_time,data_hash", "-show_data_hash", "sha256",
                           "-of", "json", str(path)], check=True, capture_output=True, text=True)
    packets = json.loads(proc.stdout)["packets"]
    if not packets:
        raise ValueError("audio stream has no packets")
    for p in packets:
        if any(not p.get(key) or p[key] == "N/A" for key in ("pts_time", "duration_time", "data_hash")):
            raise ValueError("audio packets lack timestamps, durations or hashes")
    return packets


def audio_cutoff(packets, end):
    ends = [Fraction(p["pts_time"]) + Fraction(p["duration_time"]) for p in packets]
    allowed = [v for v in ends if 0 < v <= end]
    if not allowed:
        raise ValueError("no complete audio packet fits the requested duration")
    return max(allowed)


def frame_hashes(path, count):
    proc = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-frames:v",
                           str(count), "-pix_fmt", "yuv420p", "-f", "framemd5", "-"],
                          check=True, capture_output=True, text=True)
    hashes = []
    for line in proc.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6 or int(fields[2]) != len(hashes) or int(fields[3]) != 1:
            raise ValueError("video frame timestamps are not a continuous 24 fps sequence")
        hashes.append(fields[-1])
    if len(hashes) != count:
        raise ValueError(f"expected {count} frames, decoded {len(hashes)}")
    return hashes


def trim(source, destination, seconds):
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if source == destination or destination.exists():
        raise ValueError("output must be a new file distinct from input")
    metadata = probe(source)
    videos = [s for s in metadata["streams"] if s["codec_type"] == "video"]
    audios = [s for s in metadata["streams"] if s["codec_type"] == "audio"]
    if len(videos) != 1 or len(audios) > 1:
        raise ValueError("expected one video and at most one audio stream")
    video = videos[0]
    if (video.get("pix_fmt") != "yuv420p" or video.get("sample_aspect_ratio") not in (None, "1:1", "N/A")
            or Fraction(video["r_frame_rate"]) != 24
            or Fraction(video["avg_frame_rate"]) != 24 or abs(float(video.get("start_time", 0))) > 0.001):
        raise ValueError("experiment requires a constant 24 fps, zero-start, yuv420p H3 video")
    count = target_frames(seconds)
    if not video.get("nb_frames") or count > int(video["nb_frames"]):
        raise ValueError("source has too few frames or no reliable frame count")
    end = Fraction(count, 24)
    packets = []
    cutoff = None
    if audios:
        if audios[0].get("codec_name") != "aac" or abs(float(audios[0].get("start_time", 0))) > 0.001:
            raise ValueError("experiment requires zero-start AAC audio")
        packets = audio_packets(source)
        cutoff = audio_cutoff(packets, end)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".h3-exact-", suffix=".mp4", dir=destination.parent)
    os.close(fd)
    temp = Path(name)
    try:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(source)]
        if cutoff is not None:
            cmd += ["-t", f"{float(cutoff):.9f}", "-i", str(source)]
        cmd += ["-map", "0:v:0"]
        if cutoff is not None:
            cmd += ["-map", "1:a:0", "-c:a", "copy"]
        cmd += ["-map_metadata", "0", "-vf", f"trim=end_frame={count},setpts=PTS-STARTPTS",
                "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv420p"]
        for key, option in (("color_range", "-color_range"), ("color_space", "-colorspace"),
                            ("color_transfer", "-color_trc"), ("color_primaries", "-color_primaries")):
            if video.get(key) and video[key] != "unknown":
                cmd += [option, video[key]]
        cmd += ["-movflags", "+faststart", str(temp)]
        subprocess.run(cmd, check=True)
        out = probe(temp)
        out_video = [s for s in out["streams"] if s["codec_type"] == "video"]
        out_audio = [s for s in out["streams"] if s["codec_type"] == "audio"]
        if len(out_video) != 1 or int(out_video[0].get("nb_frames", 0)) != count or len(out_audio) != len(audios):
            raise ValueError("output stream/frame count differs from the requested trim")
        if (out_video[0].get("width"), out_video[0].get("height")) != (video["width"], video["height"]):
            raise ValueError("output dimensions changed")
        if abs(float(out["format"]["duration"]) - float(end)) > 0.001:
            raise ValueError("container duration is not within 1 ms of the target frame boundary")
        if frame_hashes(source, count) != frame_hashes(temp, count):
            raise ValueError("retained decoded video frames changed")
        audio_gap = None
        if packets:
            result_packets = audio_packets(temp)
            source_hashes = [p["data_hash"] for p in packets]
            output_hashes = [p["data_hash"] for p in result_packets]
            if source_hashes[:len(output_hashes)] != output_hashes:
                raise ValueError("copied audio packets are not a source prefix")
            for original, copied in zip(packets, result_packets):
                if any(Fraction(original[key]) != Fraction(copied[key]) for key in ("pts_time", "duration_time")):
                    raise ValueError("copied audio packet timing changed")
            if out_audio[0].get("sample_rate") != audios[0].get("sample_rate"):
                raise ValueError("audio sample rate changed")
            audio_end = float(out_audio[0]["duration"])
            if audio_end > float(end) + 0.001:
                raise ValueError("audio extends past the target")
            audio_gap = max(0.0, float(end) - audio_end)
            source_end = max(Fraction(p["pts_time"]) + Fraction(p["duration_time"]) for p in packets)
            max_packet = max(Fraction(p["duration_time"]) for p in packets)
            if source_end >= end and audio_gap > float(max_packet) + 0.001:
                raise ValueError("audio was truncated by more than one packet")
        os.link(temp, destination)  # atomic; refuses an output created by another process
        return count, float(end), audio_gap
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("seconds", help="target duration, rounded to the nearest 1/24 second")
    args = parser.parse_args()
    try:
        count, seconds, gap = trim(args.source, args.output, args.seconds)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, ZeroDivisionError, InvalidOperation) as exc:
        print("exact-duration:", exc, file=sys.stderr)
        return 1
    suffix = "no audio" if gap is None else f"AAC ends {gap * 1000:.1f} ms before video"
    print(f"saved {args.output}: {count} frames, {seconds:.6f} s; retained frames and AAC packets verified, {suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
