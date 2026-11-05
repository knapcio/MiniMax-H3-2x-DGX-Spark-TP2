"""Identity guard for continued windows (opt-in ``H3_TP2_IDENTITY_GUARD``, not output-preserving).

After a continued window is denoised, its latents are decoded, a few frames spread over the window are scored
against a bank of face embeddings built from the request's reference images (``scripts/face_bank.py`` in a separate
Python with facenet-pytorch), and a window in which more than ``max_unknown`` of the sampled frames show a principal
face that matches nobody in the bank is sampled again with another seed, like the seam and cut checks in
``cont_match``.  It catches a character replaced by another person, not a face that drifted a little.  This module holds the parts
that run without a GPU: the configuration, the frame sampling, the subprocess calls and the verdict.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

logger = logging.getLogger("vllm_omni.h3_tp2.identity")


@dataclass(frozen=True)
class Config:
    max_unknown: float      # re-sample when more than this fraction of the sampled frames show an unknown principal face
    min_sim: float = 0.4    # a face is "known" when its best cosine similarity to the bank reaches this
    min_face: float = 0.05  # principal faces: at least this fraction of the frame width
    frames: int = 8         # frames scored per window, spread over it
    python: str = ""        # interpreter with facenet-pytorch (default ~/facenv/bin/python)
    script: str = ""        # scripts/face_bank.py (default: next to this package)
    width: int = 960


def config_from_env(env=None) -> Config | None:  # noqa: ANN001
    env = os.environ if env is None else env
    raw = (env.get("H3_TP2_IDENTITY_GUARD") or "").strip()
    if not raw or raw == "0":
        return None
    max_unknown = float(raw)
    if not 0.0 < max_unknown <= 1.0:
        raise ValueError("H3_TP2_IDENTITY_GUARD must be a fraction in (0, 1]: the tolerated share of unknown faces")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return Config(
        max_unknown=max_unknown,
        min_sim=float(env.get("H3_TP2_IDENTITY_MIN_SIM") or 0.4),
        min_face=float(env.get("H3_TP2_IDENTITY_MIN_FACE") or 0.05),
        frames=max(1, int(env.get("H3_TP2_IDENTITY_FRAMES") or 8)),
        python=(env.get("H3_TP2_IDENTITY_PY") or os.path.expanduser("~/facenv/bin/python")),
        script=(env.get("H3_TP2_IDENTITY_SCRIPT") or os.path.join(here, "scripts", "face_bank.py")),
        width=int(env.get("H3_TP2_IDENTITY_WIDTH") or 960),
    )


def sample_indices(total: int, count: int, skip_first: int = 0) -> list[int]:
    """``count`` frame indices spread over [skip_first, total), the last frame included (the re-entry happens late);
    the hand-over frames can be skipped because the previous window already covered them."""
    if total <= 0:
        return []
    lo = min(max(0, skip_first), total - 1)
    if count <= 1 or total - lo <= 1:
        return [total - 1]
    span = total - 1 - lo
    return sorted({lo + round(i * span / (count - 1)) for i in range(count)})


def verdict(summary: dict, max_unknown: float) -> tuple[bool, float]:
    """(ok, score): score = the fraction of frames with an unknown principal face relative to the limit (>= 1.0 fails);
    no faces = nothing to judge."""
    faces = int(summary.get("faces") or 0)
    if faces <= 0:
        return True, 0.0
    frac = float(summary.get("unknown_frac") or 0.0)
    score = frac / max(max_unknown, 1e-9)
    return frac <= max_unknown, score


def build_bank(cfg: Config, images: list[str], out: str) -> dict:
    """Run ``face_bank.py build`` on the reference image files; returns its JSON (faces, sources)."""
    cmd = [cfg.python, cfg.script, "build", "--out", out, "--width", str(cfg.width), *images]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"face bank build failed (rc {res.returncode}): {res.stderr.strip()[-400:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def score_frames(cfg: Config, bank: str, frame_files: list[str]) -> dict:
    cmd = [cfg.python, cfg.script, "score", "--bank", bank, "--min-sim", str(cfg.min_sim), "--min-face", str(cfg.min_face),
           "--width", str(cfg.width), *frame_files]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"face bank score failed (rc {res.returncode}): {res.stderr.strip()[-400:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def save_frames(frames_u8, indices: list[int], directory: str) -> list[str]:  # noqa: ANN001
    """Write the selected frames of a (F, H, W, 3) uint8 array as PNGs; returns the file names."""
    from PIL import Image

    names = []
    for i in indices:
        path = os.path.join(directory, f"frame{i:05d}.png")
        Image.fromarray(frames_u8[i]).save(path)
        names.append(path)
    return names


class Guard:
    """Per-request state: the bank file (built once from the reference images on the deciding rank)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = tempfile.mkdtemp(prefix="h3-identity-")
        self.bank: str | None = None
        self.bank_faces = 0

    def set_references(self, images) -> int:  # noqa: ANN001  (PIL images or file names)
        files = []
        for i, im in enumerate(images):
            if isinstance(im, str):
                files.append(im)
            else:
                path = os.path.join(self.dir, f"ref{i}.png")
                im.save(path)
                files.append(path)
        if not files:
            self.bank = None
            return 0
        out = os.path.join(self.dir, "bank.npz")
        info = build_bank(self.cfg, files, out)
        self.bank, self.bank_faces = out, int(info.get("faces") or 0)
        logger.info("h3_tp2 identity bank: %d face(s) from %d reference image(s): %s", self.bank_faces, len(files),
                    ", ".join(os.path.basename(s) for s in info.get("sources", [])))
        return self.bank_faces

    def score(self, frames_u8, skip_first: int = 0) -> dict:  # noqa: ANN001
        """Score a decoded window ((F, H, W, 3) uint8 numpy); {} when there is no bank."""
        if not self.bank:
            return {}
        idx = sample_indices(int(frames_u8.shape[0]), self.cfg.frames, skip_first)
        with tempfile.TemporaryDirectory(dir=self.dir) as d:
            files = save_frames(frames_u8, idx, d)
            out = score_frames(self.cfg, self.bank, files)
        out["indices"] = idx
        return out
