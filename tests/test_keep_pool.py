"""CPU tests of h3_tp2/keep_pool.py: the release gate and its installation on a stand-in torch module."""
import importlib.util
import logging
import types
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("h3_tp2_keep_pool", Path(__file__).resolve().parents[1] / "h3_tp2" / "keep_pool.py")
kp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kp)


def fake_torch(with_accelerator=True):
    calls = []
    state = {"reserved": 30 * kp.GIB, "allocs": 7, "frees": 2}

    def release(tag):
        def _release():
            calls.append(tag)
            state["reserved"] = 4 * kp.GIB
            state["frees"] += 1
        return _release

    cuda = types.SimpleNamespace(empty_cache=release("cuda"), memory_stats=lambda: {
        "reserved_bytes.all.current": state["reserved"], "allocated_bytes.all.current": 3 * kp.GIB,
        "num_device_alloc": state["allocs"], "num_device_free": state["frees"]})
    module = types.SimpleNamespace(cuda=cuda)
    if with_accelerator:
        module.accelerator = types.SimpleNamespace(empty_cache=release("accelerator"))
    return module, calls


class ParseTests(unittest.TestCase):
    def test_off_values(self):
        for text in (None, "", "0", "off", " OFF ", "false", "no"):
            self.assertIsNone(kp.parse(text))

    def test_observe_and_floor(self):
        self.assertEqual(kp.parse("observe"), ("observe", 0.0))
        self.assertEqual(kp.parse(" Observe "), ("observe", 0.0))
        self.assertEqual(kp.parse("24"), ("keep", 24.0))
        self.assertEqual(kp.parse("17.5"), ("keep", 17.5))

    def test_bad_values_fail_loudly(self):
        for text in ("yes", "-3", "nan", "inf", "24GiB"):
            with self.assertRaises(ValueError):
                kp.parse(text)


class MemAvailableTests(unittest.TestCase):
    def test_reads_the_line(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".meminfo") as handle:
            handle.write("MemTotal:       127000000 kB\nMemAvailable:   52428800 kB\n")
            handle.flush()
            self.assertAlmostEqual(kp.mem_available_gib(handle.name), 50.0)

    def test_missing_file_or_line(self):
        import tempfile
        self.assertIsNone(kp.mem_available_gib("/nonexistent/meminfo"))
        with tempfile.NamedTemporaryFile("w") as handle:
            handle.write("MemTotal: 1 kB\n")
            handle.flush()
            self.assertIsNone(kp.mem_available_gib(handle.name))


class GateTests(unittest.TestCase):
    def test_observe_always_releases(self):
        done = []
        gate = kp.Gate("observe", 0.0, available=lambda: 80.0)
        self.assertFalse(gate(lambda: done.append(1), "cuda"))
        self.assertEqual((done, gate.kept, gate.released), ([1], 0, 1))

    def test_keep_above_the_floor_releases_below(self):
        done = []
        level = [50.0]
        gate = kp.Gate("keep", 24.0, available=lambda: level[0])
        self.assertTrue(gate(lambda: done.append(1), "cuda"))
        self.assertEqual(done, [])
        level[0] = 24.0
        self.assertTrue(gate(lambda: done.append(1), "cuda"))
        level[0] = 23.9
        self.assertFalse(gate(lambda: done.append(1), "cuda"))
        self.assertEqual((done, gate.kept, gate.released, gate.calls), ([1], 2, 1, 3))

    def test_keep_releases_when_memory_cannot_be_read(self):
        done = []
        gate = kp.Gate("keep", 24.0, available=lambda: None)
        self.assertFalse(gate(lambda: done.append(1), "cuda"))
        self.assertEqual(done, [1])

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            kp.Gate("always", 1.0)

    def test_log_line_and_thinning(self):
        log = logging.getLogger("test_keep_pool")
        gate = kp.Gate("keep", 10.0, available=lambda: 40.0, stats=lambda: (30.0, 3.0, 7, 2), log=log)
        with self.assertLogs(log, level="INFO") as captured:
            for _ in range(kp.LOG_FIRST + kp.LOG_EVERY * 2):
                gate(lambda: None, "accelerator")
        self.assertEqual(len(captured.records), kp.LOG_FIRST + 2)
        first = captured.records[0].getMessage()
        self.assertIn("release 1 via accelerator KEPT", first)
        self.assertIn("reserved 30.00 -> 30.00 GiB", first)
        self.assertIn("MemAvailable 40.0 GiB", first)

    def test_stats_failure_is_tolerated(self):
        gate = kp.Gate("observe", 0.0, available=lambda: None, stats=lambda: None)
        with self.assertLogs(kp.logger, level="INFO") as captured:
            gate(lambda: None, "cuda")
        self.assertIn("reserved ? -> ? GiB", captured.records[0].getMessage())
        self.assertIn("MemAvailable ? GiB", captured.records[0].getMessage())


class InstallTests(unittest.TestCase):
    def test_observe_passes_every_call_through(self):
        module, calls = fake_torch()
        gate = kp.install(module, "observe", 0.0, available=lambda: 60.0)
        module.cuda.empty_cache()
        module.accelerator.empty_cache()
        self.assertEqual(calls, ["cuda", "accelerator"])
        self.assertEqual((gate.calls, gate.released, gate.kept), (2, 2, 0))

    def test_keep_skips_both_entry_points(self):
        module, calls = fake_torch()
        gate = kp.install(module, "keep", 24.0, available=lambda: 60.0)
        module.cuda.empty_cache()
        module.accelerator.empty_cache()
        self.assertEqual(calls, [])
        self.assertEqual(gate.kept, 2)

    def test_keep_releases_below_the_floor(self):
        module, calls = fake_torch()
        kp.install(module, "keep", 24.0, available=lambda: 12.0)
        module.accelerator.empty_cache()
        self.assertEqual(calls, ["accelerator"])

    def test_installed_once_and_stock_reachable(self):
        module, calls = fake_torch()
        first = kp.install(module, "keep", 24.0, available=lambda: 60.0)
        wrapped = module.cuda.empty_cache
        second = kp.install(module, "observe", 0.0, available=lambda: 60.0)
        self.assertIs(first, second)
        self.assertIs(module.cuda.empty_cache, wrapped)
        wrapped.__wrapped_stock__()
        self.assertEqual(calls, ["cuda"])

    def test_module_without_accelerator(self):
        module, calls = fake_torch(with_accelerator=False)
        kp.install(module, "keep", 24.0, available=lambda: 60.0)
        module.cuda.empty_cache()
        self.assertEqual(calls, [])

    def test_stats_come_from_the_module(self):
        module, _ = fake_torch()
        self.assertEqual(kp.cuda_stats(module), (30.0, 3.0, 7, 2))
        self.assertIsNone(kp.cuda_stats(types.SimpleNamespace(cuda=types.SimpleNamespace())))


if __name__ == "__main__":
    unittest.main()
