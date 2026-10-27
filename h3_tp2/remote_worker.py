"""Rank-1 entry point on the worker Spark.

Started by ``h3_tp2.agent`` as ``python -m h3_tp2.remote_worker <job.pkl> <reply host:port>``.
It replays exactly what ``MultiprocDiffusionExecutor._launch_workers`` does for a local rank
(``WorkerProc.worker_main``) with three differences:

* the "ready" pipe message goes back over TCP instead of an ``mp.Pipe``;
* before entering the busy loop the two ZMQ channels are handshaken
  (result queue: writer side; broadcast queue: remote-reader side) so that no
  PUB/SUB message is lost to a late subscription;
* ``wake_event`` is ``None`` (the stock loop tolerates it).
"""

from __future__ import annotations

import os
import pickle
import sys

from h3_tp2.wire import connect, send_obj


class _TcpPipe:
    def __init__(self, addr: str) -> None:
        self._sock = connect(addr)

    def send(self, obj) -> None:
        send_obj(self._sock, obj)

    def close(self) -> None:  # mirror mp.Connection
        try:
            self._sock.close()
        except OSError:
            pass


def main(jobfile: str, reply_addr: str) -> None:
    with open(jobfile, "rb") as fh:
        job = pickle.load(fh)
    os.environ.update({k: str(v) for k, v in job["env"].items()})
    os.environ["H3_TP2_ROLE"] = "worker"

    # Import after the env is set: the plugin reads it when vLLM-Omni loads plugins.
    from vllm_omni.diffusion.worker import diffusion_worker as dw

    _orig_loop = dw.WorkerProc._worker_busy_loop

    def _loop(self):  # noqa: ANN001
        # 1) result queue (we are the writer, the head is a remote reader)
        self.result_mq.wait_until_ready()
        # 2) broadcast queue (we are a remote reader of the head's queue)
        self.mq.wait_until_ready()
        dw.logger.info("h3_tp2: rank %s ZMQ channels ready", self.gpu_id)
        return _orig_loop(self)

    dw.WorkerProc._worker_busy_loop = _loop

    pipe = _TcpPipe(reply_addr)
    dw.WorkerProc.worker_main(
        job["rank"],
        job["od_config"],
        pipe,
        job["broadcast_handle"],
        None,
        job.get("worker_extension_cls"),
        job.get("custom_pipeline_args"),
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
