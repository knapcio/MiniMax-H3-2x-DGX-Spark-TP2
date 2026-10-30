#!/usr/bin/env python3
"""Stage, install or restore only the tested H3 prompt/trace files on one idle node.

This utility does not stop or start services. Both ranks must be stopped/idle and
coordinated by the caller before applying it; never run against a live workload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


# old = deployed 0ed86ba; new = staged fc14580. None means the file was absent.
SPEC = {
    "h3_tp2/plugin.py": ("32f8f26806c43aa918262103ea41d81230d4db3279d22d6205fb8cdb623cc960", "78cff682ea92aae3544410ab2cc1b5f1114d22ca236c16bc61ad717d073459d1"),
    "h3_tp2/cont_prompts.py": ("4911479465707faaa4b92b6a8cf5adf7ac74557fe5fd5c7cbbe92db9f0cca294", "aeda8cf7b0a416b5e5d656b894964cb4b6ff766d5ead4b12be8bd3cbc146bb26"),
    "scripts/render.sh": ("3e13b2dfbea05491aded0b7ac5f6aa287894222005f712b42b8e94653787328e", "c444b7668282bf0718af1d17e0bdb7d02b06ba0c0550d33e6e0e8999f36ae4f8"),
    "h3_tp2/h3_output.py": ("22d6a6f070ad46b53a5119601517264e8359b8f386f28fe2a5eeace5a7622068", "1d055fee9291f57f73053cc8f34306f82b4be42f486578908bb1872734b24e24"),
    "h3_tp2/vae_trace.py": (None, "f4b5e2cd4a0da235b9de7f14ef589fa418db06826bed21cd5953c183f34e9c90"),
}


def digest(path):
    if path.is_symlink():
        raise ValueError("refusing symlink: " + str(path))
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("not a regular file: " + str(path))
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_parent(root, name):
    parent = root
    for part in Path(name).parts[:-1]:
        parent = parent / part
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("missing or symlinked parent directory: " + str(parent))


def atomic_copy(source, destination):
    fd, name = tempfile.mkstemp(prefix=".h3-patch-", dir=destination.parent)
    temp = Path(name)
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(source, temp)
        # Avoid a stale timestamp-based .pyc surviving an atomic swap, even
        # when two different source revisions happen to have the same size.
        old_mtime = int(destination.stat().st_mtime) if destination.exists() else 0
        new_mtime = max(int(time.time()) + 2, old_mtime + 2)
        os.utime(temp, (new_mtime, new_mtime))
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def verify(live, stage, spec=SPEC):
    if live.is_symlink() or stage.is_symlink():
        raise ValueError("live/staging directory is a symlink")
    for name, (old, new) in spec.items():
        check_parent(live, name)
        check_parent(stage, name)
        if digest(live / name) != old:
            raise ValueError("deployed file differs from expected baseline: " + name)
        if digest(stage / name) != new:
            raise ValueError("staged file differs from tested revision: " + name)


def verify_new(live, spec=SPEC):
    if live.is_symlink():
        raise ValueError("live directory is a symlink")
    for name, (_, new) in spec.items():
        check_parent(live, name)
        if digest(live / name) != new:
            raise ValueError("deployed file differs from tested revision: " + name)


def restore(live, backup, spec=SPEC):
    if live.is_symlink() or backup.is_symlink():
        raise ValueError("live/backup directory is a symlink")
    if json.loads((backup / "manifest.json").read_text()) != {k: list(v) for k, v in spec.items()}:
        raise ValueError("backup manifest differs from the expected patch")
    # Validate every file and backup before touching any; refuse to overwrite
    # unexpected external changes or use a damaged backup.
    for name, (old, new) in spec.items():
        check_parent(live, name)
        if digest(live / name) not in (old, new):
            raise ValueError("cannot restore over an unexpected file: " + name)
        if old is not None and digest(backup / name) != old:
            raise ValueError("backup is missing or damaged: " + name)
    for name, (old, _) in spec.items():
        target = live / name
        if old is None:
            target.unlink(missing_ok=True)
        else:
            atomic_copy(backup / name, target)
        if digest(target) != old:
            raise ValueError("restore verification failed: " + name)


def apply(live, stage, backup, spec=SPEC):
    verify(live, stage, spec)
    backup.mkdir(mode=0o700)  # refuses to reuse or overwrite a backup
    for name, (old, _) in spec.items():
        if old is not None:
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(live / name, saved)
            if digest(saved) != old:
                raise ValueError("backup verification failed: " + name)
    with (backup / "manifest.json").open("x") as f:
        json.dump(spec, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        for name, (old, new) in spec.items():
            if digest(live / name) != old or digest(stage / name) != new:
                raise ValueError("file changed during deployment: " + name)
            atomic_copy(stage / name, live / name)
            if digest(live / name) != new:
                raise ValueError("deployment verification failed: " + name)
    except BaseException:
        restore(live, backup, spec)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("verify", "verify-new", "apply", "restore"))
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "verify":
        verify(args.live, args.stage)
    elif args.operation == "verify-new":
        verify_new(args.live)
    elif args.operation == "apply":
        apply(args.live, args.stage, args.backup)
    else:
        restore(args.live, args.backup)
    print(args.operation + ": verified")


if __name__ == "__main__":
    main()
