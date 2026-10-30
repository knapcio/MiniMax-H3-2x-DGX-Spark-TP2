import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("compare_speed", Path(__file__).with_name("compare_speed.py"))
speed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speed)


class SpeedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fake_flicker = patch.object(speed, "periodic_report", return_value=["periodic luma drops: none"])
        fake_flicker.start()
        self.addCleanup(fake_flicker.stop)
        for name, seconds, window, steps in ((speed.BASE, 1000, None, 20), (speed.SMALL, 800, 175, 20),
                                              (speed.TEN, 550, None, 10)):
            d = self.root / name
            d.mkdir()
            tag = speed.TAG if steps == 20 else speed.TAG10
            (d / "STATUS").write_text(tag + " ok\n")
            (d / (tag + ".mp4")).write_bytes(b"video")
            extra = {"task": "fl2va", "long_video": True}
            if window is not None:
                extra["continuation_window_frames"] = window
            timing = {"http": "200", "curl_rc": 0, "elapsed_s": seconds, "duration": 16,
                      "effective_duration": 16, "size": "1344x768", "steps": steps, "seed": 1101,
                      "task": "fl2va", "image": "painting.png", "prompt": "same prompt", "extra_params": extra}
            (d / (tag + ".mp4.timing.json")).write_text(json.dumps(timing))
            for node in ("head", "worker"):
                (d / (tag + f".mem-{node}.csv")).write_text("1,16000,0,0\n2,15000,0,0\n")
            # Match the old deployed metrics script: it has cuts but no luma report.
            (d / "metrics.txt").write_text("cuts (scene score > 0.1): none\n")
            (d / "phases.txt").write_text("phase diffuse 500.0 s\nphase diffuse 100.0 s\n")

    def test_matched_controls_produce_speed_and_memory_without_promotion(self):
        report = speed.compare(self.root)
        self.assertEqual(report["wall_change_percent"], -20.0)
        self.assertEqual(report["memory"]["175"]["worker"]["min_available_mib"], 15000)
        self.assertIn("Review visual quality", report["caveat"])

    def test_changed_prompt_makes_comparison_invalid(self):
        path = self.root / speed.SMALL / (speed.TAG + ".mp4.timing.json")
        data = json.loads(path.read_text())
        data["prompt"] = "different prompt"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "prompt"):
            speed.compare(self.root)

    def test_matched_ten_step_control_reports_speed_without_promotion(self):
        report = speed.compare(self.root, "steps")
        self.assertEqual(report["wall_change_percent"], -45.0)
        self.assertEqual(report["memory"]["10"]["head"]["min_available_mib"], 15000)
        self.assertIn("Review visual quality", report["caveat"])

    def test_ten_step_control_must_change_only_the_step_count(self):
        path = self.root / speed.TEN / (speed.TAG10 + ".mp4.timing.json")
        data = json.loads(path.read_text()); data["extra_params"]["other"] = True
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "beyond the 10-step"):
            speed.compare(self.root, "steps")

    def test_missing_memory_or_invalid_status_rejected(self):
        (self.root / speed.SMALL / (speed.TAG + ".mem-worker.csv")).write_text("1,15000,0,0\n")
        with self.assertRaisesRegex(ValueError, "memory samples"):
            speed.compare(self.root)
        (self.root / speed.SMALL / "STATUS").write_text(speed.TAG + " render-failed\n")
        with self.assertRaisesRegex(ValueError, "verification"):
            speed.compare(self.root)

    def test_standalone_flicker_report_checks_hash_and_requests_two_drop_screen(self):
        detector = self.root / "luma-flicker-20261030.py"
        repo = Path(__file__).resolve().parents[2]
        shutil.copy2(repo / "scripts/luma_flicker.py", detector)
        # Stop the setUp mock to exercise the real wrapper.
        patch.stopall()
        output = subprocess.CompletedProcess([], 0, stdout="periodic luma drops: none\n")
        with patch.object(speed.subprocess, "run", return_value=output) as run:
            self.assertEqual(speed.periodic_report(self.root, self.root / "video.mp4"), ["periodic luma drops: none"])
            self.assertEqual(run.call_args.args[0][-2:], ["--minimum", "2"])
        detector.write_text("different")
        with self.assertRaisesRegex(ValueError, "detector changed"):
            speed.periodic_report(self.root, self.root / "video.mp4")


if __name__ == "__main__":
    unittest.main()
