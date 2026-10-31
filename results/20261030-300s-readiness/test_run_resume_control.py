import json
import os
from pathlib import Path
import subprocess
import unittest

import test_queue_quality as fixture


SCRIPT = Path(__file__).with_name("run-resume-control.sh")
NAME = "quality16-960-resume-20261031"


class ResumeControlTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.QueueQualityTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.command("curl", '''
case "$*" in
  */health*) [ "${H3_TEST_HEALTH:-ok}" = ok ] && printf 200 || exit 1;;
  */metrics*) printf 'vllm_omni:num_requests_running{x="0"} 0\\nvllm_omni:num_requests_waiting{x="0"} 0\\n';;
  */v1/videos*) if [ -n "${H3_TEST_JOBS:-}" ]; then printf '%s' "$H3_TEST_JOBS"; else printf '%s' '{"data":[],"has_more":false}'; fi;;
  *) exit 1;;
esac''')
        self.env = self.f.env.copy()
        self.env.update(H3_SINGLE_RESULTS=str(self.f.results), H3_SINGLE_BENCH=str(self.f.bench),
                        H3_SINGLE_METRICS=str(self.f.metrics), H3_SINGLE_PYTHON="python3",
                        H3_SINGLE_FLICKER=str(fixture.DETECTOR), H3_SINGLE_MEMINFO=str(self.f.root / "meminfo"),
                        H3_SINGLE_WAIT_ATTEMPTS="1", H3_SINGLE_IDLE_ATTEMPTS="3",
                        H3_SINGLE_WAIT_S="0", H3_SINGLE_IDLE_S="0")

    def run_control(self, **changes):
        return subprocess.run(["bash", str(SCRIPT)], env=dict(self.env, **changes),
                              text=True, capture_output=True, timeout=20)

    def events(self):
        path = self.f.results / "resume-events-20261031.log"
        return path.read_text() if path.exists() else ""

    def test_healthy_run_records_control_parameters(self):
        result = self.run_control()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"H3-RESUME-OK {NAME}", self.events())
        args = json.loads((self.f.results / NAME / "args.json").read_text())
        self.assertEqual(args["steps"], 20)
        self.assertTrue(args["extra"]["preencode_mp4"])
        self.assertIn("move very slowly", args["prompt"])
        self.assertIn("without strong moving cloud shadows", args["prompt"])
        self.assertTrue((self.f.results / NAME / "d16_960x576_s20_seed1101_fl2va.mp4").is_file())

    def test_never_overwrites_existing_output(self):
        (self.f.results / NAME).mkdir()
        self.assertNotEqual(self.run_control().returncode, 0)
        self.assertEqual(self.f.calls(), [])
        self.assertIn("H3-RESUME-HOLD", self.events())

    def test_unready_pair_and_async_jobs_fail_closed(self):
        for change in ({"H3_TEST_HEAD_GUARD": "off"}, {"H3_TEST_WORKER_GUARD": "off"},
                       {"H3_TEST_GPU": "40"}, {"H3_TEST_HEALTH": "down"},
                       {"H3_TEST_JOBS": '{"data":[{"id":"busy"}],"has_more":false}'}):
            with self.subTest(change=change):
                result = self.run_control(**change)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.f.calls(), [])

    def test_invalid_detector_or_benchmark_failure_never_reports_success(self):
        self.assertNotEqual(self.run_control(H3_SINGLE_FLICKER="/nonexistent").returncode, 0)
        self.assertNotEqual(self.run_control(H3_TEST_FAIL=NAME).returncode, 0)
        self.assertNotIn("H3-RESUME-OK", self.events())

    def test_failed_status_never_reports_success(self):
        bench = self.f.bench
        bench.write_text(bench.read_text() + '\nprintf "%s failed\\n" "$tag" > "$OUT/STATUS"\n')
        result = self.run_control()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("H3-RESUME-OK", self.events())


if __name__ == "__main__":
    unittest.main()
