"""Identity guard for continued windows (opt-in ``H3_TP2_IDENTITY_GUARD``, not output-preserving).

After a continued window is denoised, its latents are decoded, a few frames spread over the window are scored
against a bank of face embeddings built from the request's reference images (``scripts/face_bank.py`` in a separate
Python with facenet-pytorch), and a window in which more than ``max_unknown`` of the sampled frames show a principal
face that matches nobody in the bank is sampled again with another seed, like the seam and cut checks in
``cont_match``.  It catches a character replaced by another person, not a face that drifted a little.  With
``H3_TP2_IDENTITY_GROW=<similarity>`` the bank grows after every accepted window by the principal faces that matched it
at least that well, so it follows the character through the clip instead of holding only the reference frame.  With
``H3_TP2_IDENTITY_PERSON_SIM=<similarity>`` every window is also judged as whole people (``scripts/person_bank.py``:
a person detector and DINOv2 embeddings of each figure against the character cards, so hair, hats and clothes count,
not only the face); a principal person matching no card makes the frame unknown like an unknown face.  This
module holds the parts that run without a GPU: the configuration, the frame sampling, the subprocess calls, the
verdict and the bank growth.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

logger = logging.getLogger("vllm_omni.h3_tp2.identity")
KEEP_WINDOWS = 2  # scored frames of this many windows are kept on disk


@dataclass(frozen=True)
class Config:
    max_unknown: float      # re-sample when more than this fraction of the sampled frames show an unknown principal face
    min_sim: float = 0.4    # a face is "known" when its best cosine similarity to the bank reaches this
    min_face: float = 0.05  # principal faces: at least this fraction of the frame width
    frames: int = 8         # frames scored per window, spread over it
    python: str = ""        # interpreter with facenet-pytorch (default ~/facenv/bin/python)
    script: str = ""        # scripts/face_bank.py (default: next to this package)
    width: int = 960
    grow_sim: float = 0.0   # > 0: after an accepted window, principal faces with at least this similarity join the bank
    grow_max: int = 24      # at most this many grown faces in the bank (oldest grown ones go first; references stay)
    person_sim: float = 0.0  # > 0: whole-person check (scripts/person_bank.py), a principal person below this is unknown
    person_script: str = ""  # scripts/person_bank.py (default: next to this package)
    person_min_height: float = 0.4


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
        grow_sim=float(env.get("H3_TP2_IDENTITY_GROW") or 0.0),
        grow_max=max(1, int(env.get("H3_TP2_IDENTITY_GROW_MAX") or 24)),
        person_sim=float(env.get("H3_TP2_IDENTITY_PERSON_SIM") or 0.0),
        person_script=(env.get("H3_TP2_IDENTITY_PERSON_SCRIPT") or os.path.join(here, "scripts", "person_bank.py")),
        person_min_height=float(env.get("H3_TP2_IDENTITY_PERSON_MIN_HEIGHT") or 0.4),
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
    """(ok, score): score = the fraction of frames with an unknown principal face or person relative to the limit
    (>= 1.0 fails); no faces and no people = nothing to judge."""
    if int(summary.get("faces") or 0) <= 0 and int(summary.get("persons") or 0) <= 0:
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


def score_frames(cfg: Config, bank: str, frame_files: list[str], save_faces: str = "") -> dict:
    cmd = [cfg.python, cfg.script, "score", "--bank", bank, "--min-sim", str(cfg.min_sim), "--min-face", str(cfg.min_face),
           "--width", str(cfg.width)]
    if save_faces:
        cmd += ["--save-faces", save_faces]
    cmd += frame_files
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"face bank score failed (rc {res.returncode}): {res.stderr.strip()[-400:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def grow_bank(bank: str, faces_file: str, min_sim: float, max_grown: int) -> tuple[int, int, int]:
    """Append the faces of ``faces_file`` (``face_bank.py score --save-faces``) whose best similarity to the bank
    reaches ``min_sim`` to the bank file; at most ``max_grown`` grown faces are kept (the oldest grown go first, the
    reference faces never).  Returns (added, bank size, grown faces in the bank).

    The bank then follows a character through the clip (lighting, angle, expression) instead of holding only the
    reference frame; a stranger (best similarity below ``min_sim``) is never added, so the guard still trips on a
    replacement.
    """
    import numpy as np

    data = np.load(bank, allow_pickle=False)
    emb, src = data["emb"], data["src"].astype(str)
    grown = data["grown"] if "grown" in data.files else np.zeros(len(emb), dtype=bool)
    who = data["who"] if "who" in data.files else None  # person banks: the character of each card (-1 = grown)
    cand = np.load(faces_file, allow_pickle=False)
    keep = [i for i, b in enumerate(cand["best"]) if float(b) >= min_sim]
    if not keep:
        return 0, int(len(emb)), int(grown.sum())
    emb = np.concatenate([emb, cand["emb"][keep].astype(np.float32)])
    src = np.concatenate([src, cand["src"].astype(str)[keep]])
    grown = np.concatenate([grown, np.ones(len(keep), dtype=bool)])
    if who is not None:
        who = np.concatenate([who, np.full(len(keep), -1, dtype=who.dtype)])
    excess = int(grown.sum()) - int(max_grown)
    if excess > 0:
        drop = set(np.flatnonzero(grown)[:excess].tolist())
        sel = np.array([i not in drop for i in range(len(emb))])
        emb, src, grown = emb[sel], src[sel], grown[sel]
        who = who[sel] if who is not None else None
    extra = {"who": who} if who is not None else {}
    np.savez(bank, emb=emb, src=src, grown=grown, **extra)
    return len(keep), int(len(emb)), int(grown.sum())


def build_person_bank(cfg: Config, cards: list[str], out: str) -> dict:
    """Run ``person_bank.py build`` with every card as its own character; returns its JSON."""
    cmd = [cfg.python, cfg.person_script, "build", "--out", out]
    for c in cards:
        cmd += ["--cards", c]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"person bank build failed (rc {res.returncode}): {res.stderr.strip()[-400:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def score_persons(cfg: Config, bank: str, frame_files: list[str], save_persons: str = "") -> dict:
    cmd = [cfg.python, cfg.person_script, "score", "--bank", bank, "--min-sim", str(cfg.person_sim),
           "--min-height", str(cfg.person_min_height), "--width", str(cfg.width)]
    if save_persons:
        cmd += ["--save-persons", save_persons]
    cmd += frame_files
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        raise RuntimeError(f"person bank score failed (rc {res.returncode}): {res.stderr.strip()[-400:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def frames_to_uint8(frames, convert):  # noqa: ANN001
    """Decoded frames -> (B, F, H, W, 3) uint8.

    ``convert`` is the float path ((B, C, F, H, W) in [0, 1] -> uint8, ``h3_output.to_uint8_frames``); a decoder that
    already returns uint8 (the streaming write-back) is only re-laid out, never re-scaled (scaling 0-255 values again
    would clamp every frame to white and the scorer would find no face).
    """
    if "uint8" in str(getattr(frames, "dtype", "")):
        if frames.ndim == 5 and frames.shape[-1] != 3 and frames.shape[1] == 3:
            return frames.permute(0, 2, 3, 4, 1)
        return frames
    return convert(frames)


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
        self.pbank: str | None = None   # whole-person bank (Config.person_sim > 0)
        self.pbank_cards = 0
        self.windows = 0

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
        if self.cfg.person_sim > 0:
            # the character cards are the reference images after the scene frame (all of them when there is one image)
            cards = files[1:] if len(files) > 1 else files
            pout = os.path.join(self.dir, "pbank.npz")
            pinfo = build_person_bank(self.cfg, cards, pout)
            self.pbank, self.pbank_cards = pout, int(pinfo.get("cards") or 0)
            logger.info("h3_tp2 identity person bank: %d card(s): %s", self.pbank_cards,
                        ", ".join(os.path.basename(s) for s in pinfo.get("sources", [])))
        return self.bank_faces

    def score(self, frames_u8, skip_first: int = 0) -> dict:  # noqa: ANN001
        """Score a decoded window ((F, H, W, 3) uint8 numpy); {} when there is no bank.

        The sampled frames of the last ``KEEP_WINDOWS`` scored windows stay in ``<dir>/wNNN`` next to the bank (a
        judgement can be checked by eye afterwards); older ones are removed.
        """
        if not self.bank:
            return {}
        idx = sample_indices(int(frames_u8.shape[0]), self.cfg.frames, skip_first)
        self.windows += 1
        d = os.path.join(self.dir, f"w{self.windows:03d}")
        os.makedirs(d, exist_ok=True)
        files = save_frames(frames_u8, idx, d)
        out = score_frames(self.cfg, self.bank, files, os.path.join(d, "faces.npz") if self.cfg.grow_sim > 0 else "")
        if self.pbank:
            # whole person (hair, hat, clothes): a principal person matching no card makes the frame unknown too
            pout = score_persons(self.cfg, self.pbank, files, os.path.join(d, "persons.npz") if self.cfg.grow_sim > 0 else "")
            out["persons"] = pout.get("persons", 0)
            out["person_unknown_frac"] = pout.get("unknown_frac", 0.0)
            out["person_mean_best"] = pout.get("mean_best")
            out["person_min_best"] = pout.get("min_best")
            if pout.get("persons_file"):
                out["persons_file"] = pout["persons_file"]
            out["face_unknown_frac"] = out.get("unknown_frac", 0.0)
            out["unknown_frac"] = max(float(out.get("unknown_frac") or 0.0), float(pout.get("unknown_frac") or 0.0))
        out["indices"] = idx
        out["dir"] = d
        sample = frames_u8[idx[0]]
        logger.info("h3_tp2 identity frames %s: %d of %d frames, frame %s %s mean %.1f, scorer %s", d, len(idx),
                    int(frames_u8.shape[0]), tuple(sample.shape), sample.dtype, float(sample.mean()),
                    json.dumps({k: v for k, v in out.items() if k not in ("indices", "dir", "bank_best", "faces_file", "persons_file")}))
        for old in sorted(x for x in os.listdir(self.dir) if x.startswith("w"))[:-KEEP_WINDOWS]:
            shutil.rmtree(os.path.join(self.dir, old), ignore_errors=True)
        return out

    def grow(self, summary: dict) -> int:
        """After an accepted window: add its well-matched principal faces to the bank (``Config.grow_sim`` > 0)."""
        if not summary or self.cfg.grow_sim <= 0:
            return 0
        total = 0
        f = summary.get("faces_file")
        if self.bank and f and os.path.exists(f):
            added, size, grown = grow_bank(self.bank, f, self.cfg.grow_sim, self.cfg.grow_max)
            if added:
                logger.info("h3_tp2 identity bank grown: +%d face(s) at similarity >= %.2f, bank %d (%d grown, %d reference)",
                            added, self.cfg.grow_sim, size, grown, size - grown)
            total += added
        pf = summary.get("persons_file")
        if self.pbank and pf and os.path.exists(pf):
            # whole-person embeddings: the same rule with the person threshold (a figure at >= person_sim joins)
            added, size, grown = grow_bank(self.pbank, pf, max(self.cfg.person_sim, self.cfg.grow_sim), self.cfg.grow_max)
            if added:
                logger.info("h3_tp2 identity person bank grown: +%d figure(s), bank %d (%d grown, %d card)",
                            added, size, grown, size - grown)
            total += added
        return total
