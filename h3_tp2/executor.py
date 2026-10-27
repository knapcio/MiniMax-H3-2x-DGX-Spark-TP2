"""Cross-host diffusion executor: rank 0 local, rank 1 on the second Spark.

Drop-in for ``MultiprocDiffusionExecutor`` (``--distributed-executor-backend
h3_tp2.executor.RemoteReaderExecutor``).  The scheduler/RPC/result-pump logic
of the stock executor is untouched; only the transport of rank 1 changes:

* the broadcast ``MessageQueue`` gets one local reader (rank 0, shared memory)
  and one remote reader (rank 1, ZMQ over the fabric, ``connect_ip`` =
  ``H3_TP2_HEAD_IP``);
* rank 1 is started by the worker-host agent (``H3_TP2_WORKER_AGENT`` =
  ``host:port``) which returns the worker's result-queue handle; the head
  subscribes to that queue over ZMQ;
* both ZMQ channels are handshaken (XPUB subscription -> ``READY``) before
  the first message, because PUB/SUB drops messages sent before a
  subscription is active;
* startup is bounded by one deadline (``H3_TP2_WORKER_READY_TIMEOUT``, default
  3600 s) covering rank 0, rank 1, both handshakes and the result-queue
  subscription; a failure anywhere in it terminates whatever was started;
* liveness of rank 1 comes from the agent's heartbeat (and the head's pings
  keep the agent's lease alive); a request or RPC older than
  ``H3_TP2_REQUEST_TIMEOUT_S`` (default 14400) fails the executor closed, so a
  rank that is alive but stuck cannot hang the engine forever;
* every failure path (rank 0 sentinel, rank 1 death, lost agent, watchdog) goes
  through one serialized ``_fail``: callbacks fire once, never during an
  intentional stop;
* shutdown lets the stock SHUTDOWN message reach both ranks, SIGKILLs a rank 0
  that survives the stock grace, then tells the agent to terminate rank 1.

Environment (head process, inherited by rank 0; an allowlist is forwarded to rank 1):
    H3_TP2_ROLE=head  H3_TP2_HEAD_IP  H3_TP2_WORKER_IP  H3_TP2_WORKER_AGENT
    H3_TP2_MASTER_ADDR (= head fabric IP)  H3_TP2_WORLD_SIZE (2)
    NCCL_* / GLOO_SOCKET_IFNAME as in the recipe.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.connection
import os
import queue
import select
import threading
import time
from typing import cast

from vllm.distributed.device_communicators.shm_broadcast import Handle, MessageQueue
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.worker import WorkerProc

from h3_tp2.wire import connect, recv_obj, send_obj

logger = logging.getLogger("vllm_omni.h3_tp2.executor")  # under vllm_omni so the engine's handlers pick it up

# What rank 1 inherits from the head.  Shared knobs only: the worker host owns its devices, fabric interfaces,
# caches, paths and credentials (they come from the agent's own environment, i.e. scripts/env.sh on that host).
_FORWARD_PREFIXES = ("NCCL_", "GLOO_", "H3_TP2_", "VLLM_OMNI_", "TORCH_", "OMP_", "PYTORCH_")
_FORWARD_NAMES = {"HF_HUB_OFFLINE", "VLLM_LOGGING_LEVEL", "VLLM_ATTENTION_BACKEND", "VLLM_USE_V1"}
_NEVER_FORWARD = {
    "H3_TP2_SECRET", "H3_TP2_ROLE", "H3_TP2_RESULT_CONNECT_IP", "VLLM_HOST_IP", "CUDA_VISIBLE_DEVICES",
    "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_FAMILY",
}
_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "PRIVATE")

PING_S = 15.0


def forwarded_env(environ=os.environ) -> dict[str, str]:
    out = {}
    for k, v in environ.items():
        if k in _NEVER_FORWARD or any(m in k.upper() for m in _SECRET_MARKERS):
            continue
        if k in _FORWARD_NAMES or k.startswith(_FORWARD_PREFIXES):
            out[k] = v
    return out


class RemoteReaderExecutor(MultiprocDiffusionExecutor):
    """One local rank plus one remote rank reached through the fabric."""

    # ------------------------------------------------------------------ queues
    def _init_broadcast_queue(self, num_workers: int) -> MessageQueue:
        # First hook upstream calls: make shutdown() safe at every later point of _init_executor.
        self._lifecycle_lock = threading.Lock()
        self._sock_lock = threading.Lock()
        self._stopping = False
        self._agent_sock = None
        self._local_procs: list[mp.Process] = []
        self._sync_inflight: dict[int, float] = {}
        self._pump_stop = threading.Event()
        self._futures_lock = threading.RLock()
        self._rpc_futures = {}
        self._output_futures = {}
        self._batch_split_map = {}
        self._result_pump_threads = []
        self._pump_running = False
        self._shutdown_cleaner = None
        self._finalizer = lambda: None
        ready_timeout = float(os.environ.get("H3_TP2_WORKER_READY_TIMEOUT", "3600"))
        self._startup_deadline = time.monotonic() + ready_timeout

        head_ip = os.environ["H3_TP2_HEAD_IP"]
        if num_workers != 2:
            raise ValueError(f"h3_tp2 supports exactly two ranks, got num_gpus={num_workers}")
        mq = MessageQueue(
            n_reader=num_workers,
            n_local_reader=1,
            local_reader_ranks=[0],
            connect_ip=head_ip,
        )
        self._bcast_ready = threading.Event()

        def _handshake() -> None:
            # XPUB_VERBOSE: one subscription frame per remote subscriber, then READY.
            for _ in range(mq.n_remote_reader):
                mq.remote_socket.recv()
            mq.remote_socket.send(b"READY")
            self._bcast_ready.set()
            logger.info("h3_tp2: broadcast queue handshake with remote reader done")

        self._bcast_hs_thread = threading.Thread(target=_handshake, daemon=True, name="h3-tp2-bcast-handshake")
        self._bcast_hs_thread.start()
        logger.info("h3_tp2: broadcast queue bound for remote reader at %s", mq.export_handle().remote_subscribe_addr)
        return mq

    def _init_result_queue(self, result_handle: Handle | None) -> MessageQueue:
        if result_handle is None:
            raise RuntimeError("Failed to get result queue handle from workers")
        mq = MessageQueue.create_from_handle(result_handle, 0)
        if not result_handle.local_reader_ranks:
            # Remote worker: wait for its READY (no reply may be lost), under the startup deadline, while watching
            # the agent for a death notice and rank 0 for an exit.  Same protocol as MessageQueue.wait_until_ready.
            while True:
                remaining = self._startup_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("rank 1 result queue never became ready")
                if mq.remote_socket.poll(int(min(remaining, 2.0) * 1000)):
                    recv = mq.remote_socket.recv()
                    assert recv == b"READY", recv
                    break
                self._poll_agent_once()
                for proc in self._local_procs:
                    if not proc.is_alive():
                        raise RuntimeError(f"rank 0 exited during startup (exitcode={proc.exitcode})")
            logger.info("h3_tp2: result queue of the remote rank ready (%s)", result_handle.remote_subscribe_addr)
        return mq

    # ------------------------------------------------------------- agent I/O
    def _send_agent(self, obj) -> None:
        sock = self._agent_sock
        if sock is None:
            raise ConnectionError("no agent connection")
        with self._sock_lock:
            send_obj(sock, obj)

    def _poll_agent_once(self) -> None:
        """Non-blocking: consume one pending agent message during startup; raise on death/error."""
        sock = self._agent_sock
        if sock is None or not select.select([sock], [], [], 0)[0]:
            return
        msg = recv_obj(sock, timeout=30)
        status = msg.get("status")
        if status == "alive":
            return
        raise RuntimeError(f"rank 1 failed during startup: {msg}")

    def _start_pinger(self) -> None:
        def _ping() -> None:
            while not (self._closed or self._stopping):
                time.sleep(PING_S)
                try:
                    self._send_agent({"cmd": "ping"})
                except Exception:
                    return  # the monitor reports the loss

        threading.Thread(target=_ping, daemon=True, name="h3-tp2-agent-ping").start()

    # ----------------------------------------------------------------- launch
    def _launch_workers(self, broadcast_handle: Handle, wake_events) -> tuple[list[mp.Process], list[Handle]]:
        od_config = self.od_config
        num_gpus = cast(int, od_config.num_gpus)
        if num_gpus != 2:
            raise ValueError(f"h3_tp2 supports exactly two ranks, got num_gpus={num_gpus}")
        mp.set_start_method("spawn", force=True)
        worker_extension_cls = od_config.worker_extension_cls
        custom_pipeline_args = getattr(od_config, "custom_pipeline_args", None)

        head_ip = os.environ["H3_TP2_HEAD_IP"]
        worker_ip = os.environ["H3_TP2_WORKER_IP"]
        agent_addr = os.environ["H3_TP2_WORKER_AGENT"]
        os.environ.setdefault("H3_TP2_MASTER_ADDR", head_ip)
        os.environ["H3_TP2_MASTER_PORT"] = str(od_config.master_port)
        os.environ.setdefault("H3_TP2_WORLD_SIZE", str(num_gpus))
        os.environ.setdefault("VLLM_HOST_IP", head_ip)
        deadline = self._startup_deadline

        # rank 0: local, exactly like upstream
        reader, writer = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=WorkerProc.worker_main,
            args=(0, od_config, writer, broadcast_handle, wake_events[0], worker_extension_cls, custom_pipeline_args),
            name="DiffusionWorker-0",
            daemon=True,
        )
        proc.start()
        writer.close()
        self._local_procs = [proc]

        try:
            # rank 1: remote, through the agent
            env = forwarded_env()
            env.update(
                {
                    "H3_TP2_ROLE": "worker",
                    "H3_TP2_RESULT_CONNECT_IP": worker_ip,
                    "VLLM_HOST_IP": worker_ip,
                    "RANK": "1",
                    "LOCAL_RANK": "0",
                    "WORLD_SIZE": str(num_gpus),
                }
            )
            job = {
                "rank": 1,
                "od_config": od_config,
                "broadcast_handle": broadcast_handle,
                "worker_extension_cls": worker_extension_cls,
                "custom_pipeline_args": custom_pipeline_args,
                "env": env,
            }
            logger.info("h3_tp2: requesting rank 1 from agent %s", agent_addr)
            self._agent_sock = connect(agent_addr)
            self._send_agent({"cmd": "launch", "job": job})
            self._start_pinger()

            # one deadline for both ranks; whichever reports first is handled first
            local_handle = remote_handle = None
            local_ready = remote_ready = False
            while not (local_ready and remote_ready):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"ranks not ready within the startup deadline (rank0={local_ready}, rank1={remote_ready})"
                    )
                waitables = ([] if local_ready else [reader]) + ([] if remote_ready else [self._agent_sock])
                for obj in multiprocessing.connection.wait(waitables, timeout=min(remaining, 5.0)):
                    if obj is reader:
                        try:
                            data = reader.recv()
                        except EOFError:
                            proc.join(5)
                            raise RuntimeError(f"rank 0 died during initialisation (exitcode={proc.exitcode})") from None
                        if data.get("status") != "ready":
                            raise RuntimeError(f"rank 0 failed to initialise: {data}")
                        local_handle = data.get("result_handle")
                        local_ready = True
                        reader.close()
                    else:
                        msg = recv_obj(self._agent_sock, timeout=30)
                        status = msg.get("status")
                        if status == "alive":
                            continue
                        if status == "ready":
                            remote_handle = msg.get("result_handle")
                            if remote_handle is None:
                                raise RuntimeError("rank 1 did not provide a result queue handle")
                            remote_ready = True
                        else:
                            raise RuntimeError(f"rank 1 failed to start: {msg}")
            if not self._bcast_ready.wait(timeout=max(0.0, deadline - time.monotonic())):
                raise RuntimeError("rank 1 never subscribed to the broadcast queue")
            logger.info("h3_tp2: both ranks ready")
            return [proc], [local_handle, remote_handle]
        except BaseException:
            self._abort_launch()
            raise

    def _abort_launch(self) -> None:
        with self._lifecycle_lock:
            self._stopping = True
        logger.error("h3_tp2: aborting startup; terminating both ranks")
        self._close_agent(send_shutdown=True)
        self._kill_local()

    def _close_agent(self, send_shutdown: bool) -> None:
        sock = self._agent_sock
        if sock is None:
            return
        if send_shutdown:
            try:
                self._send_agent({"cmd": "shutdown"})  # idempotent: terminating an exited worker is a no-op
            except Exception:
                pass
        try:
            sock.close()
        except OSError:
            pass
        self._agent_sock = None

    def _kill_local(self) -> None:
        """Escalate on rank 0: the stock cleaner stops at SIGTERM; a rank stuck in native code needs SIGKILL."""
        for proc in self._local_procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(10)
            if proc.is_alive():
                logger.warning("h3_tp2: rank 0 survived SIGTERM; sending SIGKILL")
                proc.kill()
                proc.join(5)

    # ---------------------------------------------------------------- monitor
    def _fail(self, reason: str) -> None:
        with self._lifecycle_lock:
            if self._closed or self._stopping or self._is_failed:
                return
            self._is_failed = True
        logger.error("h3_tp2: %s; shutting down the executor", reason)
        self.shutdown()
        for cb in self._failure_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("failure_callback raised")

    def _start_worker_monitor(self) -> None:
        # rank 0 sentinel (replaces the stock monitor so that it goes through _fail like every other path)
        sentinels = [p.sentinel for p in self._processes]

        def _local_monitor() -> None:
            try:
                finished = multiprocessing.connection.wait(sentinels)
            except OSError:
                return
            if self._closed or self._stopping:
                return
            dead = [p for p in self._processes if p.sentinel in finished]
            self._fail("rank 0 died unexpectedly: " + ", ".join(f"{p.name}(exitcode={p.exitcode})" for p in dead))

        if sentinels:
            threading.Thread(target=_local_monitor, daemon=True, name="diffusion-worker-monitor").start()

        sock = self._agent_sock
        if sock is None:
            return

        def _monitor() -> None:
            while not (self._closed or self._stopping):
                try:
                    msg = recv_obj(sock, timeout=60)
                except Exception as exc:  # timeout, closed, EOF, bad signature
                    if self._closed or self._stopping:
                        return
                    self._fail(f"lost the worker agent ({exc}): rank 1 considered dead")
                    return
                status = msg.get("status")
                if status == "alive":
                    continue
                if status == "dead":
                    if self._closed or self._stopping:
                        return
                    self._fail(f"rank 1 died (exitcode={msg.get('exitcode')})")
                    return
                self._fail(f"unexpected message from the worker agent: {msg}")
                return

        threading.Thread(target=_monitor, daemon=True, name="h3-tp2-remote-monitor").start()

        timeout_s = float(os.environ.get("H3_TP2_REQUEST_TIMEOUT_S", "14400"))
        if timeout_s > 0:
            def _watchdog() -> None:
                first_seen: dict[str, float] = {}
                while not (self._closed or self._stopping):
                    time.sleep(10)
                    now = time.monotonic()
                    with self._futures_lock:
                        pending = set(self._rpc_futures) | set(self._output_futures)
                        sync_pending = dict(self._sync_inflight)
                    first_seen = {k: first_seen.get(k, now) for k in pending}
                    stuck = [k for k, t in first_seen.items() if now - t > timeout_s]
                    stuck += [f"rpc#{k}" for k, t in sync_pending.items() if now - t > timeout_s]
                    if stuck:
                        self._fail(f"request(s) {stuck} pending for more than {timeout_s:.0f} s")
                        return

            threading.Thread(target=_watchdog, daemon=True, name="h3-tp2-request-watchdog").start()

    def collective_rpc(self, *args, **kwargs):
        """Stock RPC, with the synchronous wait registered for the request watchdog."""
        key = id(threading.current_thread()) ^ int(time.monotonic_ns())
        with self._futures_lock:
            self._sync_inflight[key] = time.monotonic()
        try:
            return super().collective_rpc(*args, **kwargs)
        finally:
            with self._futures_lock:
                self._sync_inflight.pop(key, None)

    def shutdown(self) -> None:
        with self._lifecycle_lock:
            self._stopping = True
        try:
            super().shutdown()  # stock SHUTDOWN through the broadcast queue (rank 1 is a reader), local join
        finally:
            self._kill_local()
            self._close_agent(send_shutdown=True)
