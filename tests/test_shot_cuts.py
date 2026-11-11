"""Tests of scripts/shot_cuts.py: the head-trim rule on the CPU, and the measurement on a synthetic clip when ffmpeg
is installed."""
import importlib.util
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("shot_cuts", Path(__file__).resolve().parents[1] / "scripts" / "shot_cuts.py")
sc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sc)


class HeadTrimTests(unittest.TestCase):
    def test_cut_with_leading_silence_trims_at_the_cut(self):
        # queue144 shot 2: cut at 1.0417 s, silence 0-1.69 s
        self.assertAlmostEqual(sc.head_trim([1.041667], [(0.0, 1.69), (2.2, 3.1)]), 25 / 24)

    def test_no_cut_no_trim(self):
        self.assertEqual(sc.head_trim([], [(0.0, 1.97)]), 0.0)

    def test_cut_after_the_line_started_stays(self):
        # queue144 shot 6: cut at 2.625 s, silence only 0-0.78 s (the line starts in the head)
        self.assertEqual(sc.head_trim([2.625], [(0.0, 0.78)]), 0.0)
        self.assertEqual(sc.head_trim([1.5], [(0.0, 0.78)]), 0.0)

    def test_late_cut_is_not_a_head(self):
        # queue144 shot 5: the cut at 3.54 s is a cut back to the wide shot, not an establishing head
        self.assertEqual(sc.head_trim([3.541667], [(0.25, 1.42), (3.68, 6.18)]), 0.0)
        self.assertEqual(sc.head_trim([2.5], [(0.0, 3.0)], max_head=2.0), 0.0)
        self.assertAlmostEqual(sc.head_trim([2.5], [(0.0, 3.0)], max_head=3.0), 60 / 24)

    def test_silence_must_start_at_zero(self):
        self.assertEqual(sc.head_trim([1.0], [(0.3, 2.0)]), 0.0)
        self.assertAlmostEqual(sc.head_trim([1.0], [(0.04, 2.0)]), 1.0)

    def test_snaps_to_a_frame(self):
        self.assertAlmostEqual(sc.head_trim([1.3], [(0.0, 1.5)], fps=24.0), 31 / 24)
        self.assertAlmostEqual(sc.head_trim([1.3], [(0.0, 1.5)], fps=25.0), 32 / 25)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class SyntheticClipTests(unittest.TestCase):
    """A 4 s clip: 1 s of one colour, then another colour (the cut), silence for 1.5 s, then a tone."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.clip = os.path.join(self.dir, "shot.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-f", "lavfi", "-i", "color=c=blue:s=160x96:r=24:d=1",
                        "-f", "lavfi", "-i", "color=c=orange:s=160x96:r=24:d=3",
                        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1.5",
                        "-f", "lavfi", "-i", "sine=f=440:r=16000:d=2.5",
                        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a][3:a]concat=n=2:v=0:a=1[a]",
                        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                        self.clip], check=True)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_measure_finds_the_cut_the_silence_and_the_trim(self):
        row = sc.measure(self.clip, max_head=2.0, threshold=0.3, noise_db=-32.0)
        self.assertEqual(len(row["cuts_s"]), 1)
        self.assertAlmostEqual(row["cuts_s"][0], 1.0, delta=0.05)
        self.assertTrue(row["silences_s"] and row["silences_s"][0][0] < 0.05 and abs(row["silences_s"][0][1] - 1.5) < 0.15)
        self.assertAlmostEqual(row["trim_s"], 1.0, delta=0.05)

    def test_apply_shortens_by_the_trim_and_fades_the_new_start(self):
        out = os.path.join(self.dir, "trimmed.mp4")
        sc.apply(self.clip, out, 2.0)   # the tone starts at 1.5 s, so the new start is inside the tone
        self.assertAlmostEqual(sc.duration(out), sc.duration(self.clip) - 2.0, delta=0.1)
        self.assertEqual(sc.scene_cuts(out), [])
        self.assertLess(rms_db(out, 0.0, 0.01) + 10, rms_db(out, 0.2, 0.01))   # faded in: the first 10 ms are quiet

    def test_trim_filters(self):
        vf, af = sc.trim_filters(1.0417, 8.0, 0.04, fps=24.0, size="1344x768")
        self.assertEqual(vf, "trim=start=1.0417,setpts=PTS-STARTPTS,fps=24,scale=1344:768:flags=lanczos")
        self.assertEqual(af, "atrim=start=1.0417,asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.04,afade=t=out:st=6.9183:d=0.04")
        self.assertEqual(sc.trim_filters(0.0, 4.0)[0], "trim=start=0.0000,setpts=PTS-STARTPTS")


def rms_db(path: str, start: float, length: float) -> float:
    """RMS in dBFS of the mono 16 kHz PCM decode of ``path`` between ``start`` and ``start + length`` seconds."""
    import array
    import math

    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True).stdout
    samples = array.array("h", pcm)
    chunk = samples[int(start * 16000):int((start + length) * 16000)]
    if not chunk:
        return -120.0
    return 20 * math.log10(max(math.sqrt(sum(v * v for v in chunk) / len(chunk)) / 32768.0, 1e-6))


if __name__ == "__main__":
    unittest.main()
