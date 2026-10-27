"""Length-prefixed pickle frames between the head executor, the worker agent and the remote worker.

Security model: the control channel carries pickles, so anyone who can reach the agent port can run code as the
agent's user.  Bind the agent to the fabric address of a point-to-point cable only (the recipe does), and set the
same ``H3_TP2_SECRET`` on both Sparks to have every frame authenticated with HMAC-SHA256 (a frame with a bad tag is
rejected before it is unpickled).  Without a secret the channel is as trusting as NCCL itself.

Sockets get keepalives and ``TCP_USER_TIMEOUT`` so a black-holed peer surfaces as an error within ~30 s instead of the
kernel's retransmission limit (minutes).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import select
import socket
import struct
import time

_HDR = struct.Struct("!Q")
_TAG_LEN = hashlib.sha256().digest_size
_MAX_FRAME = 1 << 30
USER_TIMEOUT_MS = int(os.environ.get("H3_TP2_TCP_USER_TIMEOUT_MS", "30000"))
FRAME_S = 60.0  # once the first byte of a frame arrived, the rest must follow within this long


def _secret() -> bytes | None:
    s = os.environ.get("H3_TP2_SECRET", "")
    return s.encode() if s else None


def tune(sock: socket.socket) -> socket.socket:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, val in (("TCP_KEEPIDLE", 10), ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 3), ("TCP_USER_TIMEOUT", USER_TIMEOUT_MS)):
        opt = getattr(socket, name, None)
        if opt is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, opt, val)
            except OSError:
                pass
    return sock


def send_obj(sock: socket.socket, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sec = _secret()
    tag = hmac.new(sec, payload, hashlib.sha256).digest() if sec else b""
    sock.sendall(_HDR.pack(len(payload)) + tag + payload)


def _wait_readable(sock: socket.socket, deadline: float | None) -> None:
    if deadline is None:
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not select.select([sock], [], [], remaining)[0]:
        raise socket.timeout("no data")


def _recv_exact(sock: socket.socket, n: int, deadline: float | None) -> bytes:
    """Read exactly n bytes.  The socket stays in blocking mode (a writer thread may share it); waiting is done
    with select against an absolute deadline, so a timeout never leaves half a frame consumed: before the first
    byte it raises socket.timeout (retryable), after it the frame deadline makes a stall a ConnectionError."""
    chunks: list[bytes] = []
    while n:
        _wait_readable(sock, deadline)
        chunk = sock.recv(min(n, 1 << 20))
        if not chunk:
            raise ConnectionError("peer closed")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def recv_obj(sock: socket.socket, timeout: float | None = None):
    """Receive one frame.  ``timeout`` bounds the wait for the frame's first byte (socket.timeout if none arrives);
    the rest of the frame has FRAME_S."""
    _wait_readable(sock, None if timeout is None else time.monotonic() + timeout)
    first = sock.recv(1)
    if not first:
        raise ConnectionError("peer closed")
    deadline = time.monotonic() + FRAME_S
    try:
        (n,) = _HDR.unpack(first + _recv_exact(sock, _HDR.size - 1, deadline))
        if n > _MAX_FRAME:
            raise ConnectionError(f"frame too large ({n} bytes)")
        sec = _secret()
        tag = _recv_exact(sock, _TAG_LEN, deadline) if sec else b""
        payload = _recv_exact(sock, n, deadline)
    except socket.timeout:
        raise ConnectionError("peer stalled in the middle of a frame") from None
    if sec and not hmac.compare_digest(tag, hmac.new(sec, payload, hashlib.sha256).digest()):
        raise ConnectionError("frame signature mismatch (H3_TP2_SECRET differs between the two hosts?)")
    return pickle.loads(payload)


def connect(addr: str, timeout: float = 30.0) -> socket.socket:
    host, port = addr.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=timeout)
    tune(sock)
    sock.settimeout(None)
    return sock


def listen(addr: str) -> socket.socket:
    host, port = addr.rsplit(":", 1)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, int(port)))
    srv.listen(4)
    return srv
