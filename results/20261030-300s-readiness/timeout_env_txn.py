#!/usr/bin/env python3
"""Change only four numeric H3 timeouts, with a hash-checked reversible transaction.

The manifest contains hashes and timeout numbers, never the environment file's
other contents. This helper does not start or stop services.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile


OLD = {
    "H3_TP2_REQUEST_TIMEOUT_S": 28800,
    "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT": 28800,
    "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": 29400,
    "H3_RENDER_TIMEOUT_S": 30000,
}
NEW = {
    "H3_TP2_REQUEST_TIMEOUT_S": 57600,
    "VLLM_OMNI_DLO_DP_WAVE_TIMEOUT": 57600,
    "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": 58200,
    "H3_RENDER_TIMEOUT_S": 58800,
}
NAMES = "|".join(map(re.escape, OLD))
SIMPLE = re.compile(rf"^([ \t]*(?:export[ \t]+)?)({NAMES})([ \t]*=[ \t]*)([0-9]+)([ \t]*)$")
MENTION = re.compile(rf"\b(?:{NAMES})[ \t]*=")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(path):
    if path.is_symlink() or not path.is_file() or path.parent.is_symlink():
        raise ValueError("environment must be a regular, non-symlinked file")
    st = path.stat()
    if st.st_uid != os.getuid() or st.st_nlink != 1:
        raise ValueError("environment owner or link count is unexpected")
    data = path.read_bytes()
    if fingerprint(st) != fingerprint(path.stat()):
        raise ValueError("environment changed while reading")
    return data, st


def fingerprint(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def replace(data, expected, target):
    lines = data.decode("utf-8").splitlines(keepends=True)
    seen = {}
    output = []
    for line in lines:
        content = line.rstrip("\r\n")
        ending = line[len(content):]
        match = SIMPLE.fullmatch(content)
        if match:
            prefix, name, equals, number, suffix = match.groups()
            if name in seen:
                raise ValueError("duplicate timeout assignment: " + name)
            seen[name] = int(number)
            output.append(prefix + name + equals + str(target[name]) + suffix + ending)
        else:
            if not content.lstrip().startswith("#") and MENTION.search(content):
                raise ValueError("non-simple timeout assignment")
            output.append(line)
    if seen != expected:
        raise ValueError("timeout assignments differ from expected numeric values")
    return "".join(output).encode("utf-8")


def metadata(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked transaction metadata")
    obj = json.loads(path.read_text())
    if set(obj) != {"old_sha256", "new_sha256", "old", "new"} or obj["old"] != OLD or obj["new"] != NEW:
        raise ValueError("unexpected timeout transaction metadata")
    for key in ("old_sha256", "new_sha256"):
        if not isinstance(obj[key], str) or not re.fullmatch(r"[0-9a-f]{64}", obj[key]):
            raise ValueError("invalid transaction digest")
    return obj


def atomic_write(path, new, before):
    fd, name = tempfile.mkstemp(prefix=".h3-timeout-", dir=path.parent)
    temp = Path(name)
    try:
        os.fchmod(fd, stat.S_IMODE(before.st_mode))
        with os.fdopen(fd, "wb") as f:
            f.write(new); f.flush(); os.fsync(f.fileno())
        if path.is_symlink() or fingerprint(path.stat()) != fingerprint(before):
            raise ValueError("environment changed before replacement")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def prepare(env, meta):
    original, _ = snapshot(env)
    updated = replace(original, OLD, NEW)
    if replace(updated, NEW, OLD) != original:
        raise ValueError("timeout assignments cannot be restored byte-for-byte")
    record = {"old_sha256": digest(original), "new_sha256": digest(updated), "old": OLD, "new": NEW}
    fd = os.open(meta, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())


def apply(env, meta):
    record = metadata(meta)
    current, st = snapshot(env)
    if digest(current) != record["old_sha256"]:
        raise ValueError("environment differs from prepared baseline")
    updated = replace(current, OLD, NEW)
    if digest(updated) != record["new_sha256"]:
        raise ValueError("prepared updated hash mismatch")
    atomic_write(env, updated, st)
    if digest(snapshot(env)[0]) != record["new_sha256"]:
        raise ValueError("updated environment verification failed")


def restore(env, meta):
    record = metadata(meta)
    current, st = snapshot(env)
    if digest(current) == record["old_sha256"]:
        return
    if digest(current) != record["new_sha256"]:
        raise ValueError("environment changed; refusing to restore over it")
    previous = replace(current, NEW, OLD)
    if digest(previous) != record["old_sha256"]:
        raise ValueError("reconstructed baseline hash mismatch")
    atomic_write(env, previous, st)
    if digest(snapshot(env)[0]) != record["old_sha256"]:
        raise ValueError("restored environment verification failed")


def state(env, meta):
    record = metadata(meta)
    current, _ = snapshot(env)
    checksum = digest(current)
    if checksum == record["old_sha256"]:
        return "old"
    if checksum == record["new_sha256"]:
        return "new"
    return "changed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "apply", "restore", "state"))
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "state":
        print(state(args.env, args.meta))
    else:
        {"prepare": prepare, "apply": apply, "restore": restore}[args.operation](args.env, args.meta)
        print(args.operation + ": verified")


if __name__ == "__main__":
    main()
