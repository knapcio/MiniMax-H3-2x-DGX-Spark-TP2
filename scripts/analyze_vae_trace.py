#!/usr/bin/env python3
"""Compare approximate VAE temporal-overlap traces with encoded global luma changes."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys


spec = importlib.util.spec_from_file_location('luma_flicker', Path(__file__).with_name('luma_flicker.py'))
detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detector)
LINE = re.compile(r'h3_tp2 VAE temporal blend (\d+) approx-frame (\d+): '
                  r'old (\[[^\]]*\]) new (\[[^\]]*\]) blended (\[[^\]]*\])')


def parse_trace(text):
    entries, seen = [], set()
    for line in text.splitlines():
        found = LINE.search(line)
        if not found:
            continue
        index, frame = int(found[1]), int(found[2])
        if index in seen:
            raise ValueError('duplicate overlap index; separate trace streams before analysis')
        seen.add(index)
        values = [json.loads(found[i]) for i in (3, 4, 5)]
        if (not values[0] or len({len(x) for x in values}) != 1 or
                any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                    for group in values for v in group)):
            raise ValueError('invalid overlap luma samples')
        entries.append({'overlap': index, 'approx_frame': frame, 'old': values[0],
                        'new': values[1], 'blended': values[2]})
    if not entries:
        raise ValueError('no temporal blend traces found')
    return sorted(entries, key=lambda x: x['overlap'])


def encoded_means(video):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video), '-map', '0:v:0', '-an', '-vf',
                             'scale=320:-2:flags=area,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-',
                             '-fps_mode', 'passthrough', '-f', 'null', '-'],
                            check=True, capture_output=True, text=True)
    means = [float(x) for x in re.findall(r'lavfi\.signalstats\.YAVG=([0-9.]+)', result.stdout)]
    if len(means) < 2:
        raise ValueError('fewer than two encoded video frames')
    return means


def phase_changes(means, *, start=1, end=None, period=17):
    """Describe signed frame changes by phase in the half-open frame interval."""
    end = len(means) if end is None else end
    if period < 1 or start < 1 or end > len(means) or start >= end:
        raise ValueError('invalid phase interval or period')
    stats = []
    for phase in range(period):
        frames = [i for i in range(start, end) if i % period == phase]
        changes = [means[i] - means[i - 1] for i in frames]
        rebounds = [means[i + 1] - means[i] for i in frames if i + 1 < len(means)]
        stats.append({'phase': phase, 'frames': len(changes),
                      'median_change': statistics.median(changes) if changes else None,
                      'negative_fraction': sum(value < 0 for value in changes) / len(changes) if changes else None,
                      'median_next_change': statistics.median(rebounds) if rebounds else None})
    return stats


def compare(entries, means):
    if len(means) < 2:
        raise ValueError('fewer than two encoded video frames')
    drops = detector.periodic_drops(means, period=17, threshold=2.0, minimum=2)
    drop_frames = {frame for group in drops for frame, _ in group}
    rows = []
    for item in entries:
        frame = item['approx_frame']
        if not 0 < frame < len(means):
            continue
        nearby = [offset for offset in range(-8, 9) if frame + offset in drop_frames]
        nearest = min(nearby, key=lambda offset: (abs(offset), offset)) if nearby else None
        rows.append({'overlap': item['overlap'], 'approx_frame': frame,
                     'encoded_frame_change': means[frame] - means[frame - 1],
                     'encoded_rebound_after': means[frame + 1] - means[frame] if frame + 1 < len(means) else None,
                     'encoded_repeated_drop': frame in drop_frames,
                     'nearest_repeated_drop_offset': nearest,
                     'old_first': item['old'][0], 'new_first': item['new'][0],
                     'blend_first': item['blended'][0],
                     'old_rebound_first': item['old'][1] - item['old'][0] if len(item['old']) > 1 else None,
                     'new_rebound_first': item['new'][1] - item['new'][0] if len(item['new']) > 1 else None,
                     'blend_rebound_first': item['blended'][1] - item['blended'][0] if len(item['blended']) > 1 else None,
                     'new_minus_old_first': item['new'][0] - item['old'][0],
                     'encoded_minus_blend_first': means[frame] - item['blended'][0]})
    if not rows:
        raise ValueError('no overlap trace maps into the encoded frames')
    offsets = {offset: sum(row['approx_frame'] + offset in drop_frames for row in rows)
               for offset in range(-8, 9)}
    most_matches = max(offsets.values())
    # Show every phase, not only the largest detected drops: a gradual change
    # in scene brightness can produce negative steps at every boundary too.
    phase_stats = phase_changes(means)
    return {'encoded_frames': len(means), 'trace_entries': len(entries), 'overlaps_in_range': len(rows),
            'periodic_global_drop_runs': [[{'frame': frame, 'delta': delta} for frame, delta in group]
                                          for group in drops],
            'encoded_changes_by_phase': phase_stats,
            'drop_matches_by_candidate_offset': offsets,
            'candidate_offsets_with_most_matches': ([offset for offset, matches in offsets.items()
                                                      if matches == most_matches] if most_matches else []),
            'median_abs_encoded_vs_sampled_blend': statistics.median(abs(row['encoded_minus_blend_first']) for row in rows),
            'overlaps': rows,
            'caveat': ('Trace frame indices and sampled luma are approximate; spatial sampling, crop, color conversion '
                       'and encoding may differ. Offset matching is exploratory and can align periodic events by chance. '
                       'A match is diagnostic, not proof of cause; global averages can miss regional flicker or real movement.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('trace', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(parse_trace(args.trace.read_text()), encoded_means(args.video)), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f'analyze-vae-trace: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
