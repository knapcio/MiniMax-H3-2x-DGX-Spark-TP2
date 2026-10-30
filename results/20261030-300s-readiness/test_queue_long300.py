import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "queue-long300.sh"
TXN = HERE / "timeout_env_txn.py"
spec = importlib.util.spec_from_file_location("trace_queue_fixture", HERE / "test_queue_vae_trace.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)

ORIGINAL = ("The painting comes alive: a wide Dutch landscape under a huge sky. The clouds drift slowly across the sky, "
            "their shadows glide over the plain, the river and the pond in the distance glint and ripple, leaves on the trees "
            "tremble in a light breeze, a few birds fly across the sky, the tiny figures on the road walk on. The camera is "
            "locked on a tripod and does not move at all; one continuous take; soft natural ambience.")
COMBINED = ORIGINAL.replace("the tiny figures on the road walk on",
                            "the tiny figures on the road move very slowly, taking small natural steps without sudden movements").replace(
                                "their shadows glide over the plain",
                                "the light on the landscape stays soft and nearly constant, without strong moving cloud shadows").replace(
                                "The clouds drift slowly across the sky",
                                "The clouds drift slowly across the sky, keeping their broad soft painted shapes")


class Long300QueueTests(unittest.TestCase):
    def setUp(self):
        self.t = trace.VaeTraceQueueTests()
        self.t.setUp()
        self.addCleanup(self.t.doCleanups)
        self.root, self.results = self.t.root, self.t.results
        self.envfile = self.root / "head.env"
        self.original_env = ("export H3_TP2_REQUEST_TIMEOUT_S=28800\n"
                             "export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=28800\n"
                             "export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=29400\n"
                             "export H3_RENDER_TIMEOUT_S=30000\n")
        self.envfile.write_text(self.original_env)
        self.envfile.chmod(0o600)
        self.set_proc_env(False)
        (self.results / "gen.log").write_text(
            "H3-AUTO-SKIP-300 timeout or speed margin insufficient 2026-10-31 02:00 CET\n"
            "H3-AUTO-PATCH-DONE fc14580 2026-10-31 03:00 CET\n")
        tag960 = "d16_960x576_s20_seed1101_fl2va"
        tag60 = "d60_960x576_s20_seed1101_fl2va"
        self.t.output_case("quality16-960-stream", tag960)
        self.t.output_case("quality60-960-stream", tag60)
        self.t.output_case("quality5-stream", "d5_1344x768_s20_seed1101_fl2va")
        self.t.output_case("quality16-stream", "d16_1344x768_s20_seed1101_fl2va")
        control = self.results / "quality16-960-stream"
        (control / "metrics.txt").write_text("cuts (scene score > 0.1): none\n")
        (control / "luma-flicker.txt").write_text("periodic luma drops: none >= 2 YAVG for 2 successive 17-frame periods (400 frames checked)\n")
        long_control = self.results / "quality60-960-stream"
        (long_control / "metrics.txt").write_text("cuts (scene score > 0.1): none\n")
        (long_control / "phases.txt").write_text("phase diffuse 900.0 s\nphase diffuse 1000.0 s\n")
        for folder, tag in ((control, tag960), (long_control, tag60)):
            for rank in ("head", "worker"):
                (folder / f"{tag}.mem-{rank}.csv").write_text("1,15000,0,0\n2,14000,0,0\n")
        timing = {"http": "200", "curl_rc": 0, "size": "960x576", "steps": 20, "seed": 1101,
                  "task": "fl2va", "prompt": COMBINED, "image": str(self.results / "koninck.png"),
                  "extra_params": {"long_video": True, "long_video_mode": "continuation", "preencode_mp4": True}}
        for folder, tag, duration in ((control, tag960, 16), (long_control, tag60, 60)):
            (folder / f"{tag}.mp4.timing.json").write_text(json.dumps(dict(timing, duration=duration)))
        self.t.env.update(H3_LONG300_RESULTS=str(self.results), H3_LONG300_BENCH=str(self.t.bench),
                          H3_LONG300_METRICS=str(self.t.metrics), H3_LONG300_PYTHON="python3",
                          H3_LONG300_FLICKER=str(trace.FLICKER), H3_LONG300_TXN=str(TXN),
                          H3_LONG300_ENV_FILE=str(self.envfile), H3_LONG300_META=str(self.root / "meta.json"),
                          H3_LONG300_PROC_ROOT=str(self.root / "proc"), H3_LONG300_MEMINFO=str(self.root / "meminfo"),
                          H3_LONG300_WAIT_ATTEMPTS="1", H3_LONG300_IDLE_ATTEMPTS="3", H3_LONG300_READY_ATTEMPTS="3",
                          H3_LONG300_WAIT_S="0", H3_LONG300_IDLE_S="0", H3_LONG300_READY_S="0")
        systemctl = self.t.bin / "systemctl"
        text = systemctl.read_text()
        old = '''    if [ -e "$H3_TRACE_DROPIN" ]; then printf 'H3_TP2_VAE_BLEND_TRACE=1\\0' > "$H3_TRACE_PROC_ROOT/123/environ"
    else printf 'H3_TP2_VAE_BLEND_TRACE=0\\0' > "$H3_TRACE_PROC_ROOT/123/environ"; fi'''
        replacement = '''    : > "$H3_LONG300_PROC_ROOT/123/environ"
    for key in H3_TP2_REQUEST_TIMEOUT_S VLLM_OMNI_DLO_DP_WAVE_TIMEOUT VLLM_OMNI_VIDEO_SYNC_TIMEOUT; do
      value=$(sed -n "s/^export $key=//p" "$H3_LONG300_ENV_FILE")
      printf '%s=%s\\0' "$key" "$value" >> "$H3_LONG300_PROC_ROOT/123/environ"
    done
    printf 'H3_TP2_VAE_BLEND_TRACE=0\\0' >> "$H3_LONG300_PROC_ROOT/123/environ"'''
        assert old in text
        systemctl.write_text(text.replace(old, replacement))
        bench_text = self.t.bench.read_text()
        needle = 'if [ "$name" = "${H3_TEST_FAIL_CASE:-}" ]; then\n'
        assert needle in bench_text
        self.t.bench.write_text(bench_text.replace(
            needle, needle + '  [ "${H3_TEST_CHANGE_ENV:-0}" != 1 ] || printf "# external change\\n" >> "$H3_LONG300_ENV_FILE"\n'
            '  [ "${H3_TEST_CHANGE_TXN:-0}" != 1 ] || printf "# changed\\n" >> "$H3_LONG300_TXN"\n'))

    def set_proc_env(self, new):
        values = ("57600", "57600", "58200") if new else ("28800", "28800", "29400")
        names = ("H3_TP2_REQUEST_TIMEOUT_S", "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT", "VLLM_OMNI_VIDEO_SYNC_TIMEOUT")
        data = b"".join(f"{name}={value}\0".encode() for name, value in zip(names, values))
        (self.t.proc / "environ").write_bytes(data + b"H3_TP2_VAE_BLEND_TRACE=0\0")

    def run_queue(self, **overrides):
        return subprocess.run(["bash", str(SCRIPT)], env=dict(self.t.env, **overrides),
                              text=True, capture_output=True, timeout=25)

    def test_time_only_skip_runs_300_and_retains_verified_16h_timeouts(self):
        result = self.run_queue()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.t.cases(), ["long300-960-stream"])
        self.assertIn("H3_TP2_REQUEST_TIMEOUT_S=57600", self.envfile.read_text())
        self.assertEqual(self.t.state.read_text(), "active")
        self.assertIn("H3-AUTO-LONG300-DONE", (self.results / "gen.log").read_text())
        self.assertIn('"preencode_mp4":true', (self.results / "long300-960-stream/extra.json").read_text())

    def test_skip_if_already_completed_or_if_quality_not_safe(self):
        for quality in (False, True):
            with self.subTest(quality=quality):
                test = Long300QueueTests(); test.setUp()
                try:
                    lines = "H3-AUTO-PATCH-DONE fc14580 2026-10-31 03:00 CET\n"
                    if quality:
                        test.t.output_case("quality300-960-stream", "d300_960x576_s20_seed1101_fl2va")
                        lines += "H3-AUTO-QUALITY-DONE 2026-10-31 02:00 CET\n"
                    else:
                        lines += "H3-AUTO-SKIP-300 960 control quality or memory samples did not pass 2026-10-31 02:00 CET\n"
                    (test.results / "gen.log").write_text(lines)
                    result = test.run_queue(PYTHONOPTIMIZE="1")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(test.t.cases(), [])
                    self.assertNotIn("stop h3-tp2-head.service", test.t.systemctl())
                    self.assertIn("H3-AUTO-LONG300-SKIPPED", (test.results / "gen.log").read_text())
                finally:
                    test.doCleanups()

    def test_too_slow_projection_skips_without_changing_config(self):
        (self.results / "quality60-960-stream/phases.txt").write_text("phase diffuse 2000.0 s\n")
        self.assertEqual(self.run_queue().returncode, 0)
        self.assertEqual(self.envfile.read_text(), self.original_env)
        self.assertNotIn("stop h3-tp2-head.service", self.t.systemctl())

    def test_quality_memory_or_request_mismatch_prevents_restart(self):
        for mutation in ("flicker", "cuts", "memory", "prompt", "cuts60", "memory60", "prompt60"):
            with self.subTest(mutation=mutation):
                test = Long300QueueTests(); test.setUp()
                try:
                    p = test.results / "quality16-960-stream"
                    if mutation == "flicker": (p / "luma-flicker.txt").write_text("periodic luma drops: frames 1..35\n")
                    if mutation == "cuts": (p / "metrics.txt").write_text("cuts (scene score > 0.1): frame 1\n")
                    if mutation == "memory":
                        (p / "d16_960x576_s20_seed1101_fl2va.mem-head.csv").write_text("1,5000\n2,5000\n")
                    if mutation == "prompt":
                        path = p / "d16_960x576_s20_seed1101_fl2va.mp4.timing.json"
                        item = json.loads(path.read_text()); item["prompt"] = "different"; path.write_text(json.dumps(item))
                    q = test.results / "quality60-960-stream"
                    if mutation == "cuts60": (q / "metrics.txt").write_text("cuts (scene score > 0.1): frame 1\n")
                    if mutation == "memory60":
                        (q / "d60_960x576_s20_seed1101_fl2va.mem-head.csv").write_text("1,5000\n2,5000\n")
                    if mutation == "prompt60":
                        path = q / "d60_960x576_s20_seed1101_fl2va.mp4.timing.json"
                        item = json.loads(path.read_text()); item["prompt"] = "different"; path.write_text(json.dumps(item))
                    result = test.run_queue(PYTHONOPTIMIZE="1")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("stop h3-tp2-head.service", test.t.systemctl())
                    self.assertEqual(test.envfile.read_text(), test.original_env)
                finally:
                    test.doCleanups()

    def test_gpu_async_or_bad_env_prevents_restart(self):
        for override in ({"H3_TEST_GPU": "40"}, {"H3_TEST_JOBS": "1"}, {"H3_TEST_WORKER": "bad"}):
            with self.subTest(override=override):
                test = Long300QueueTests(); test.setUp()
                try:
                    self.assertNotEqual(test.run_queue(**override).returncode, 0)
                    self.assertNotIn("stop h3-tp2-head.service", test.t.systemctl())
                finally:
                    test.doCleanups()
        self.envfile.write_text(self.original_env.replace("28800", "20000", 1))
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertNotIn("stop h3-tp2-head.service", self.t.systemctl())

    def test_failed_300_restores_original_timeouts_when_idle(self):
        result = self.run_queue(H3_TEST_FAIL_CASE="long300-960-stream")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.envfile.read_text(), self.original_env)
        self.assertEqual(self.t.state.read_text(), "active")
        self.assertEqual(self.t.systemctl().count("stop h3-tp2-head.service"), 2)
        self.assertIn("H3-AUTO-HOLD long300", (self.results / "gen.log").read_text())

    def test_busy_or_async_work_after_failure_is_never_stopped(self):
        for flag in ("H3_TEST_BUSY_ON_FAIL", "H3_TEST_JOBS_ON_FAIL"):
            with self.subTest(flag=flag):
                test = Long300QueueTests(); test.setUp()
                try:
                    result = test.run_queue(H3_TEST_FAIL_CASE="long300-960-stream", **{flag: "1"})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(test.t.state.read_text(), "active")
                    self.assertIn("H3_TP2_REQUEST_TIMEOUT_S=57600", test.envfile.read_text())
                    self.assertEqual(test.t.systemctl().count("stop h3-tp2-head.service"), 1)
                finally:
                    test.doCleanups()

    def test_failed_start_restores_but_activating_start_is_left_untouched(self):
        result = self.run_queue(H3_TEST_FAIL_START_ONCE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.envfile.read_text(), self.original_env)
        self.assertEqual(self.t.state.read_text(), "active")
        test = Long300QueueTests(); test.setUp()
        try:
            result = test.run_queue(H3_TEST_START_ACTIVATING="1")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(test.t.state.read_text(), "activating")
            self.assertIn("H3_TP2_REQUEST_TIMEOUT_S=57600", test.envfile.read_text())
            self.assertEqual(test.t.systemctl().count("stop h3-tp2-head.service"), 1)
        finally:
            test.doCleanups()

    def test_nonquiescent_stop_does_not_apply_timeouts(self):
        result = self.run_queue(H3_TEST_STOP_ACTIVATING="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.t.state.read_text(), "activating")
        self.assertEqual(self.envfile.read_text(), self.original_env)

    def test_external_environment_change_is_not_overwritten_or_stopped_again(self):
        result = self.run_queue(H3_TEST_FAIL_CASE="long300-960-stream", H3_TEST_CHANGE_ENV="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.t.state.read_text(), "active")
        self.assertIn("# external change", self.envfile.read_text())
        self.assertEqual(self.t.systemctl().count("stop h3-tp2-head.service"), 1)

    def test_changed_transaction_helper_is_not_executed_for_restore(self):
        helper = self.root / "transaction.py"
        shutil.copy2(TXN, helper)
        result = self.run_queue(H3_TEST_FAIL_CASE="long300-960-stream", H3_TEST_CHANGE_TXN="1",
                                H3_LONG300_TXN=str(helper))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.t.state.read_text(), "active")
        self.assertIn("# changed", helper.read_text())
        self.assertEqual(self.t.systemctl().count("stop h3-tp2-head.service"), 1)

    def test_failed_predecessor_never_starts(self):
        (self.results / "gen.log").write_text("H3-AUTO-HOLD prompt-patch-fc14580 phase=healthy rc=1\n")
        self.assertNotEqual(self.run_queue().returncode, 0)
        self.assertEqual(self.t.cases(), [])
        self.assertNotIn("stop h3-tp2-head.service", self.t.systemctl())


if __name__ == "__main__":
    unittest.main()

