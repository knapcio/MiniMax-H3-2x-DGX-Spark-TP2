#!/usr/bin/env python3
"""Associate spatial face detections with anchored positions for exploratory QA.

This is a position tracker, not face recognition. Crossings, occlusions,
scene cuts or changed appearance can give the wrong identity.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import sys


def point(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in value)):
        raise ValueError(name + ' must be two finite normalized coordinates')
    return float(value[0]), float(value[1])


def match(anchors: list[tuple[float, float]], faces: list[dict], max_distance: float) -> list[int | None]:
    """Maximize assigned tracks, then minimize distance without reusing a face."""
    centers = [(f['face_box'][0] + f['face_box'][2] / 2, f['face_box'][1] + f['face_box'][3] / 2) for f in faces]
    @lru_cache(None)
    def solve(i, used):
        if i == len(anchors):
            return 0, 0.0, ()
        count, cost, choices = solve(i + 1, used)
        best = (count, cost, (None,) + choices)
        for j, c in enumerate(centers):
            if used & (1 << j):
                continue
            distance = math.hypot(c[0] - anchors[i][0], c[1] - anchors[i][1])
            if distance > max_distance:
                continue
            n, total, rest = solve(i + 1, used | (1 << j))
            candidate = (n + 1, total + distance, (j,) + rest)
            if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
        return best
    return list(solve(0, 0)[2])


def track(records: list[dict], anchors: list[tuple[float, float]], *, fps: float = 2,
          min_area: float = .0025, max_distance: float = .18) -> list[list[dict]]:
    if (not isinstance(records, list) or not records or len(records) > 100000 or not 1 <= len(anchors) <= 4
            or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
            or type(min_area) not in (int, float) or not math.isfinite(min_area) or not 0 <= min_area <= 1
            or type(max_distance) not in (int, float) or not math.isfinite(max_distance) or not 0 < max_distance <= 1):
        raise ValueError('invalid frames, anchors or tracking options')
    locations = [point(a, 'anchor') for a in anchors]
    tracks = [[] for _ in locations]
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get('faces'), list):
            raise ValueError('each frame must have a faces list')
        candidates = []
        for face in record['faces']:
            box = face.get('face_box') if isinstance(face, dict) else None
            if (not isinstance(box, list) or len(box) != 4 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
                continue
            x, y, w, h = box
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.001 or y + h > 1.001 or w * h < min_area:
                continue
            candidates.append(face)
        candidates.sort(key=lambda f: f['face_box'][2] * f['face_box'][3], reverse=True)
        candidates = candidates[:8]
        assigned = match(locations, candidates, max_distance)
        for i, j in enumerate(assigned):
            row = {'time_s_approx': round((index + .5) / fps, 3), 'face_count': len(candidates), 'detected': j is not None}
            if j is not None:
                face = candidates[j]; row['face_box'] = face['face_box']; row['confidence'] = face.get('confidence')
                if 'inner_lip_height_over_width' in face:
                    row['inner_lip_height_over_width'] = face['inner_lip_height_over_width']
                x, y, w, h = face['face_box']; locations[i] = (x + w / 2, y + h / 2)
            tracks[i].append(row)
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('detections', type=Path, help='JSON array emitted by face_landmarks_all.swift')
    parser.add_argument('--anchor', nargs=2, type=float, action='append', required=True, metavar=('X', 'Y'))
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--min-area', type=float, default=.0025)
    parser.add_argument('--max-distance', type=float, default=.18)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        records = json.loads(args.detections.read_text())
        tracks = track(records, args.anchor, fps=args.fps, min_area=args.min_area, max_distance=args.max_distance)
        if args.output_dir.exists():
            raise FileExistsError('output directory already exists')
        args.output_dir.mkdir()
        for i, rows in enumerate(tracks, 1):
            data = {'track': i, 'sample_fps': args.fps, 'samples': rows,
                    'note': 'Position association only; not identity or lip-sync verification.'}
            with (args.output_dir / f'track-{i:02d}.json').open('x') as f:
                json.dump(data, f, indent=2); f.write('\n')
        print(f'tracked {len(tracks)} positions across {len(records)} frames')
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'track-face-detections: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
