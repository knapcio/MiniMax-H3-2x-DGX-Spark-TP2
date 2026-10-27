"""Stand-in for h3_tp2.remote_worker in the transport tests (no GPU, no vLLM).

Behaviour comes from job["env"]["FAKE_MODE"]:
  sleep            report ready, then live until SIGTERM (exit 0)
  die_after_ready  report ready, exit 3 one second later
  die_before_ready exit 4 without reporting
  ignore_term      report ready, ignore SIGTERM (exercise the SIGKILL path)
"""
import pickle
import signal
import sys
import time

from h3_tp2.wire import connect, send_obj


def main(jobfile, reply_addr):
    with open(jobfile, "rb") as fh:
        job = pickle.load(fh)
    mode = job["env"].get("FAKE_MODE", "sleep")
    if mode == "die_before_ready":
        sys.exit(4)
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sock = connect(reply_addr)
    send_obj(sock, {"status": "ready", "result_handle": {"fake": True, "rank": job["rank"]}})
    if mode == "die_after_ready":
        time.sleep(1)
        sys.exit(3)
    while True:
        time.sleep(0.5)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
