"""Check decoded audio PTS against sample counts, before timestamp-flattening decode.

Packet durations are deliberately unused: a muxer can stretch a packet's
duration across a gap without adding any decoded samples. Decoder-trimmed
priming is handled by anchoring at the first decoded frame, not timestamp zero.
"""
from fractions import Fraction
import json
from pathlib import Path
import subprocess


def assess_frames(data: dict) -> dict:
    result = {'continuous': False, 'status': 'unavailable', 'first_decoded_pts_s': None,
              'frames': 0, 'max_cumulative_delta_s': None, 'max_adjacent_delta_s': None,
              'tolerance_s': None}
    try:
        streams, frames = data['streams'], data['frames']
        if len(streams) != 1 or not frames:
            raise ValueError('one audio stream and decoded frames are required')
        rate = int(streams[0]['sample_rate'])
        time_base = Fraction(streams[0]['time_base'])
        if rate <= 0 or time_base <= 0:
            raise ValueError('invalid sample rate or time base')
        tolerance = 2 * time_base + Fraction(1, rate)
        if tolerance > Fraction(1, 1000):
            raise ValueError('audio timestamp resolution is coarser than 1 ms')
        result['tolerance_s'] = float(tolerance)
        origin = previous = None
        cumulative = previous_samples = Fraction(0)
        worst = adjacent = Fraction(0)
        monotonic = True
        for frame in frames:
            if (type(frame.get('pts')) is not int or type(frame.get('nb_samples')) is not int
                    or frame['nb_samples'] <= 0):
                raise ValueError('missing or invalid decoded-frame PTS/sample count')
            pts = frame['pts'] * time_base
            if origin is None:
                origin = pts
            if previous is not None:
                monotonic &= pts > previous
                adjacent = max(adjacent, abs(pts - previous - previous_samples))
            worst = max(worst, abs(pts - origin - cumulative))
            previous_samples = Fraction(frame['nb_samples'], rate)
            cumulative += previous_samples
            previous = pts
        result.update(continuous=bool(monotonic and worst <= tolerance and adjacent <= tolerance),
                      status='measured', first_decoded_pts_s=float(origin), frames=len(frames),
                      max_cumulative_delta_s=float(worst), max_adjacent_delta_s=float(adjacent))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        result['reason'] = str(exc)
    return result


def inspect(path: Path) -> dict:
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_frames', '-show_streams',
         '-show_entries', 'frame=pts,nb_samples:stream=sample_rate,time_base', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True, timeout=1800)
    return assess_frames(json.loads(probe.stdout))


def relative_start(frame_timing: dict, track_timing: dict):
    """Keep absolute presentation offset separate from internal continuity."""
    if (frame_timing['status'] != 'measured' or not frame_timing['continuous']
            or track_timing['audio_relative_to_timeline_s'] is None):
        return None
    return frame_timing['first_decoded_pts_s'] - track_timing['timeline_origin_s']
