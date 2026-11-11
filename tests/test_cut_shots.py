"""Tests of scripts/cut_shots.py: the trim plan on the CPU, the film on synthetic clips when ffmpeg is installed."""
import argparse
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("cut_shots", Path(__file__).resolve().parents[1] / "scripts" / "cut_shots.py")
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)


class TrimArgTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(cs.parse_trims(["5=2.333", "7=1"], 9), {5: 2.333, 7: 1.0})
        self.assertEqual(cs.parse_trims([], 3), {})

    def test_bad(self):
        for items in (["x=1"], ["2"], ["0=1"], ["4=1"], ["1="]):
            with self.assertRaises(argparse.ArgumentTypeError):
                cs.parse_trims(items, 3)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.measured = {"a.mp4": ([1.04], [(0.0, 1.69)]), "b.mp4": ([], [(0.0, 1.97)]), "c.mp4": ([2.6], [(0.0, 0.78)])}
        self._orig = (cs.shot_cuts.scene_cuts, cs.shot_cuts.silences, cs.shot_cuts.frame_rate)
        cs.shot_cuts.scene_cuts = lambda clip, threshold=0.3: self.measured[clip][0]
        cs.shot_cuts.silences = lambda clip, noise_db=-32.0, min_s=0.35: self.measured[clip][1]
        cs.shot_cuts.frame_rate = lambda clip: 24.0

    def tearDown(self):
        cs.shot_cuts.scene_cuts, cs.shot_cuts.silences, cs.shot_cuts.frame_rate = self._orig

    def test_measured_rule_per_shot(self):
        rows = cs.plan(["a.mp4", "b.mp4", "c.mp4"], {}, True, 2.0)
        self.assertEqual([c for c, _ in rows], ["a.mp4", "b.mp4", "c.mp4"])
        self.assertAlmostEqual(rows[0][1], 25 / 24)
        self.assertEqual([t for _, t in rows[1:]], [0.0, 0.0])

    def test_hand_values_win_and_no_head_trim_zeroes_the_rest(self):
        rows = cs.plan(["a.mp4", "b.mp4", "c.mp4"], {2: 1.0}, True, 2.0)
        self.assertEqual(rows[1], ("b.mp4", 1.0))
        self.assertAlmostEqual(rows[0][1], 25 / 24)
        rows = cs.plan(["a.mp4", "b.mp4", "c.mp4"], {3: 0.5}, False, 2.0)
        self.assertEqual([t for _, t in rows], [0.0, 0.0, 0.5])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class FilmTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.clips = []
        for i, (colour, d) in enumerate([("blue", 2.0), ("orange", 3.0)]):
            clip = os.path.join(self.dir, f"s{i}.mp4")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=160x96:r=24:d={d}",
                            "-f", "lavfi", "-i", f"sine=f=440:r=16000:d={d}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", clip], check=True)
            self.clips.append(clip)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_film_is_the_shots_minus_the_trims(self):
        out = os.path.join(self.dir, "film.mp4")
        cs.main(self.clips + ["--out", out, "--no-head-trim", "--trim", "2=1.0"])
        self.assertAlmostEqual(cs.shot_cuts.duration(out), 2.0 + 2.0, delta=0.15)
        self.assertEqual(len(cs.shot_cuts.scene_cuts(out)), 1)


if __name__ == "__main__":
    unittest.main()
