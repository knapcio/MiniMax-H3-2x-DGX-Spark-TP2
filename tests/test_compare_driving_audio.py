import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_driving_audio.py"
spec = importlib.util.spec_from_file_location("compare_driving_audio", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompareDrivingAudioTests(unittest.TestCase):
    def test_shift_and_gain_are_measured_with_full_windows(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 0.1, 8000 * 5)
        output = np.pad(reference, (240, 0))[:len(reference)] * 1.7
        rows = module.compare(reference, output, 8000, window_seconds=2, max_shift_seconds=0.1)
        for row in rows:
            self.assertEqual(row["status"], "measured")
            self.assertEqual(row["shift_samples"], 240)
            self.assertAlmostEqual(row["correlation"], 1, places=5)
            self.assertAlmostEqual(row["fitted_gain"], 1.7, places=4)

    def test_leading_output_and_silent_input(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(0, .1, 10000)
        output = np.pad(reference[100:], (0, 100))
        result = module.compare_segment(reference, output, 200, 4000, 200)
        self.assertEqual(result["shift_samples"], -100)
        self.assertAlmostEqual(result["correlation"], 1, places=5)
        silent = module.compare_segment(np.zeros(10000), output, 200, 4000, 200)
        self.assertEqual(silent["status"], "low_reference_energy")
        self.assertIsNone(silent["shift_samples"])

    def test_short_or_silent_output_is_not_treated_as_match(self):
        reference = np.linspace(-1, 1, 1000)
        result = module.compare_segment(reference, np.zeros(1000), 0, 500, 100)
        self.assertEqual(result["status"], "low_output_energy")
        with self.assertRaises(ValueError):
            module.compare(reference[:10], reference[:10], 8000)

    def test_reports_quiet_seconds_and_above_scale_samples(self):
        result = module.signal_levels(np.array([0, 0, .5, 1.1, 0, 0], dtype=np.float32), 2)
        self.assertEqual(result["rms_by_second"][0], 0)
        self.assertEqual(result["rms_by_second"][2], 0)
        self.assertGreater(result["rms_by_second"][1], 0)
        self.assertEqual(result["samples_above_full_scale_at_analysis_rate"], 1)
        self.assertAlmostEqual(result["peak_abs_at_analysis_rate"], 1.1)

    def test_container_timestamps_count_toward_audio_delay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.wav", root / "delayed.mp4"
            raw = np.random.default_rng(42).normal(0, .08, 24000)
            raw = np.convolve(raw, np.ones(5) / 5, mode="same")
            pcm = np.int16(np.clip(raw, -1, 1) * 32767)
            with wave.open(str(source), "wb") as wav:
                wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm.tobytes())
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=24:d=3.5",
                            "-itsoffset", "0.5", "-i", str(source), "-c:v", "libx264", "-preset", "ultrafast",
                            "-c:a", "aac", "-t", "3.5", str(output)], check=True)
            result = json.loads(subprocess.check_output([sys.executable, str(SCRIPT), str(source), str(output),
                                                         "--window-seconds", "1"], text=True))
            self.assertIsNotNone(result["reference_timing"]["assumption"])
            self.assertGreater(result["segments"][0]["correlation"], .95)
            self.assertAlmostEqual(result["segments"][0]["presentation_shift_seconds"], .5, delta=.015)


if __name__ == "__main__":
    unittest.main()
