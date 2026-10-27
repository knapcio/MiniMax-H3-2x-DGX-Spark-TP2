"""CPU-only tests of the control channel: wire framing + HMAC, and the agent session state machine
(ready delivery, heartbeats, worker death before/after ready, head shutdown, head loss, lease expiry).
Run: python -m unittest discover -s tests -v
"""
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from h3_tp2.wire import connect, listen, recv_obj, send_obj  # noqa: E402


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class WireTests(unittest.TestCase):
    def _pair(self):
        srv = listen("127.0.0.1:0")
        c = connect(f"127.0.0.1:{srv.getsockname()[1]}")
        a, _ = srv.accept()
        srv.close()
        self.addCleanup(c.close); self.addCleanup(a.close)
        return c, a

    def test_roundtrip(self):
        c, a = self._pair()
        send_obj(c, {"x": list(range(1000))})
        self.assertEqual(recv_obj(a, timeout=2)["x"][-1], 999)

    def test_hmac_mismatch_rejected(self):
        c, a = self._pair()
        os.environ["H3_TP2_SECRET"] = "one"
        send_obj(c, {"cmd": "launch"})
        os.environ["H3_TP2_SECRET"] = "two"
        with self.assertRaises(ConnectionError):
            recv_obj(a, timeout=2)
        os.environ["H3_TP2_SECRET"] = "one"
        send_obj(c, {"cmd": "ping"})
        self.assertEqual(recv_obj(a, timeout=2)["cmd"], "ping")
        del os.environ["H3_TP2_SECRET"]

    def test_unsigned_frame_rejected_when_secret_set(self):
        c, a = self._pair()
        send_obj(c, {"cmd": "launch"})           # unsigned
        os.environ["H3_TP2_SECRET"] = "s"
        try:
            with self.assertRaises(Exception):
                recv_obj(a, timeout=2)
        finally:
            del os.environ["H3_TP2_SECRET"]


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.logdir = tempfile.mkdtemp(prefix="h3tp2-agent-test-")
        env = dict(os.environ, PYTHONPATH=ROOT, H3_TP2_HEAD_SILENCE_S="8")
        cls.agent = subprocess.Popen(
            [sys.executable, "-m", "h3_tp2.agent", "--bind", f"127.0.0.1:{cls.port}", "--log-dir", cls.logdir,
             "--worker-module", "tests.fake_worker"],
            env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(50):
            try:
                connect(f"127.0.0.1:{cls.port}").close(); break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("agent did not start")

    @classmethod
    def tearDownClass(cls):
        cls.agent.terminate(); cls.agent.wait(5)

    def _launch(self, mode):
        sock = connect(f"127.0.0.1:{self.port}")
        send_obj(sock, {"cmd": "launch", "job": {"rank": 1, "env": {"FAKE_MODE": mode}}})
        return sock

    def _collect(self, sock, until_status, timeout=15, ping=True):
        """Read until a message with until_status; by default keep the agent's lease alive with pings (as the head does)."""
        seen = []
        deadline = time.monotonic() + timeout
        last_ping = 0.0
        while time.monotonic() < deadline:
            if ping and time.monotonic() - last_ping > 1.0:
                try:
                    send_obj(sock, {"cmd": "ping"})
                except (ConnectionError, OSError):
                    break
                last_ping = time.monotonic()
            try:
                m = recv_obj(sock, timeout=1.0)
            except socket.timeout:
                continue
            except (ConnectionError, OSError):
                break
            seen.append(m)
            if m.get("status") == until_status:
                break
        return seen

    def _worker_pid(self, msgs):
        for m in msgs:
            if m.get("status") == "alive":
                return m["pid"]
        return None

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0); return True
        except ProcessLookupError:
            return False

    def test_ready_heartbeat_shutdown(self):
        sock = self._launch("sleep")
        msgs = self._collect(sock, "alive", timeout=12)
        statuses = [m["status"] for m in msgs]
        self.assertIn("ready", statuses); self.assertIn("alive", statuses)
        self.assertEqual(msgs[statuses.index("ready")]["result_handle"]["rank"], 1)
        pid = self._worker_pid(msgs)
        send_obj(sock, {"cmd": "ping"})
        send_obj(sock, {"cmd": "shutdown"})
        msgs = self._collect(sock, "dead", timeout=20)
        self.assertEqual(msgs[-1]["status"], "dead")
        time.sleep(0.5)
        self.assertFalse(self._alive(pid))
        with self.assertRaises((ConnectionError, OSError)):
            recv_obj(sock, timeout=5)   # session closed after dead
        sock.close()

    def test_worker_dies_after_ready(self):
        sock = self._launch("die_after_ready")
        msgs = self._collect(sock, "dead", timeout=15)
        statuses = [m["status"] for m in msgs]
        self.assertIn("ready", statuses)
        self.assertEqual(msgs[-1], {"status": "dead", "exitcode": 3})
        sock.close()

    def test_worker_dies_before_ready(self):
        sock = self._launch("die_before_ready")
        msgs = self._collect(sock, "dead", timeout=15)
        self.assertNotIn("ready", [m["status"] for m in msgs])
        self.assertEqual(msgs[-1]["status"], "dead"); self.assertEqual(msgs[-1]["exitcode"], 4)
        sock.close()

    def test_head_loss_terminates_worker(self):
        sock = self._launch("sleep")
        msgs = self._collect(sock, "alive", timeout=12)
        pid = self._worker_pid(msgs); self.assertIsNotNone(pid)
        sock.close()                      # head vanishes
        deadline = time.monotonic() + 10
        while self._alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertFalse(self._alive(pid))

    def test_lease_expiry_terminates_worker(self):
        sock = self._launch("sleep")
        msgs = self._collect(sock, "alive", timeout=12)
        pid = self._worker_pid(msgs); self.assertIsNotNone(pid)
        # keep the socket open but silent: H3_TP2_HEAD_SILENCE_S=8 for the test agent
        deadline = time.monotonic() + 20
        while self._alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertFalse(self._alive(pid))
        sock.close()

    def test_sigkill_path(self):
        sock = self._launch("ignore_term")
        msgs = self._collect(sock, "alive", timeout=12)
        pid = self._worker_pid(msgs); self.assertIsNotNone(pid)
        send_obj(sock, {"cmd": "shutdown"})
        msgs = self._collect(sock, "dead", timeout=30)
        self.assertEqual(msgs[-1]["status"], "dead")
        self.assertFalse(self._alive(pid))
        sock.close()

    def test_second_session_after_first(self):
        for _ in range(2):
            sock = self._launch("die_after_ready")
            msgs = self._collect(sock, "dead", timeout=15)
            self.assertEqual(msgs[-1]["status"], "dead")
            sock.close()
            time.sleep(0.3)


if __name__ == "__main__":
    unittest.main()
