"""Worker-host agent: launches and supervises the rank-1 diffusion worker on request of the head.

Run as a user service on the worker Spark, bound to its fabric address:
    python -m h3_tp2.agent --bind 10.100.96.5:18095

Protocol (length-prefixed pickle, optional HMAC, see wire.py), one head session at a time:
    head  -> {"cmd": "launch", "job": {...}}
    agent -> {"status": "ready", "result_handle": Handle}      (forwarded from the worker)
    agent -> {"status": "alive", "pid": int}                    every 5 s
    head  -> {"cmd": "ping"}                                    every 15 s (the lease: 90 s of silence ends the session)
    agent -> {"status": "dead", "exitcode": int}                then the connection closes
    head  -> {"cmd": "shutdown"}                                SIGTERM, 15 s, SIGKILL, then "dead"

One thread owns all writes to the head socket (an outbound queue), one thread owns all reads; the forwarding thread
for the worker's ready message and the main supervision loop only enqueue.  Any session end terminates the worker.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import queue
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

from h3_tp2.wire import listen, recv_obj, send_obj, tune

log = logging.getLogger("h3_tp2.agent")

HEARTBEAT_S = 5.0
TERM_GRACE_S = 15.0
HEAD_SILENCE_S = float(os.environ.get("H3_TP2_HEAD_SILENCE_S", "90"))
FLUSH_S = 5.0


def _terminate(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    log.info("terminating rank worker pid %s", child.pid)
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + TERM_GRACE_S
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def _serve_one(conn: socket.socket, python: str, log_dir: str, worker_module: str) -> None:
    msg = recv_obj(conn, timeout=60)
    if msg.get("cmd") != "launch":
        send_obj(conn, {"status": "error", "error": f"unexpected command {msg.get('cmd')!r}"})
        return
    job = msg["job"]
    fd, jobfile = tempfile.mkstemp(prefix="h3_tp2_job_", suffix=".pkl")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(job, fh, protocol=pickle.HIGHEST_PROTOCOL)

    reply_srv = listen("127.0.0.1:0")  # the worker reports "ready" here (loopback)
    reply_addr = f"127.0.0.1:{reply_srv.getsockname()[1]}"
    os.makedirs(log_dir, exist_ok=True)
    logfile = open(os.path.join(log_dir, f"worker-rank{job['rank']}-{int(time.time())}.log"), "ab")
    env = dict(os.environ)
    env.update({k: str(v) for k, v in job["env"].items()})
    child = subprocess.Popen(
        [python, "-m", worker_module, jobfile, reply_addr],
        env=env,
        stdout=logfile,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.info("launched rank %s pid %s log %s", job["rank"], child.pid, logfile.name)
    pidfile = os.path.join(log_dir, f"rank{job['rank']}.pid")  # process-group id for the memory guard
    with open(pidfile, "w") as fh:
        fh.write(str(child.pid))

    outq: queue.Queue = queue.Queue()
    lost = threading.Event()        # head socket unusable (write failed, read failed, lease expired)
    shutdown_req = threading.Event()
    done = threading.Event()
    last_rx = [time.monotonic()]

    def _writer() -> None:
        while True:
            item = outq.get()
            if item is None:
                return
            try:
                send_obj(conn, item)
            except (ConnectionError, OSError) as exc:
                log.warning("write to head failed (%s)", exc)
                lost.set()
                return

    def _reader() -> None:
        while not done.is_set():
            try:
                m = recv_obj(conn, timeout=2.0)
            except socket.timeout:
                if time.monotonic() - last_rx[0] > HEAD_SILENCE_S:
                    log.warning("no message from the head for %.0f s: lease expired", HEAD_SILENCE_S)
                    lost.set()
                    return
                continue
            except (ConnectionError, OSError) as exc:
                if not done.is_set():
                    log.warning("read from head failed (%s)", exc)
                    lost.set()
                return
            last_rx[0] = time.monotonic()
            cmd = m.get("cmd")
            if cmd == "shutdown":
                shutdown_req.set()
                return
            if cmd != "ping":
                log.warning("ignoring unexpected message from head: %r", cmd)

    def _forward_ready() -> None:
        reply_srv.settimeout(1.0)
        while not done.is_set():
            try:
                c, _ = reply_srv.accept()
            except socket.timeout:
                if child.poll() is not None:
                    return
                continue
            except OSError:
                return
            tune(c)
            try:
                while not done.is_set():
                    outq.put(recv_obj(c))
            except (ConnectionError, OSError):
                return

    threads = [
        threading.Thread(target=_writer, daemon=True, name="agent-writer"),
        threading.Thread(target=_reader, daemon=True, name="agent-reader"),
        threading.Thread(target=_forward_ready, daemon=True, name="agent-forward"),
    ]
    for t in threads:
        t.start()

    last_beat = time.monotonic()
    try:
        while True:
            code = child.poll()
            if code is not None:
                log.warning("rank worker exited with %s", code)
                outq.put({"status": "dead", "exitcode": code})
                return
            if lost.is_set():
                log.warning("head session lost; terminating worker")
                _terminate(child)
                return
            if shutdown_req.is_set():
                _terminate(child)
                outq.put({"status": "dead", "exitcode": child.returncode})
                return
            if time.monotonic() - last_beat >= HEARTBEAT_S:
                outq.put({"status": "alive", "pid": child.pid})
                last_beat = time.monotonic()
            time.sleep(0.2)
    finally:
        # let the writer flush the last message, then tear everything down
        outq.put(None)
        threads[0].join(FLUSH_S)
        done.set()
        _terminate(child)
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        for t in threads[1:]:
            t.join(3.0)
        for path in (jobfile, pidfile):
            try:
                os.unlink(path)
            except OSError:
                pass
        try:
            reply_srv.close()
            logfile.close()
        except OSError:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", required=True, help="fabric host:port, e.g. 10.100.96.5:18095")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--log-dir", default=os.path.expanduser("~/h3-tp2-logs"))
    ap.add_argument("--worker-module", default="h3_tp2.remote_worker", help=argparse.SUPPRESS)  # tests only
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    srv = listen(args.bind)
    log.info("h3_tp2 agent listening on %s (python %s, hmac=%s)", args.bind, args.python, bool(os.environ.get("H3_TP2_SECRET")))
    while True:
        conn, peer = srv.accept()
        tune(conn)
        log.info("head connected from %s", peer)
        try:
            _serve_one(conn, args.python, args.log_dir, args.worker_module)
        except Exception:
            log.exception("session failed")
        finally:
            try:
                conn.close()
            except OSError:
                pass
            log.info("session closed")


if __name__ == "__main__":
    main()
