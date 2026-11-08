"""Opt-in identity re-anchoring for FL2VA continuation (``H3_TP2_ID_LOCK``); off unless the variable is set.

Character identity drifts window by window in long FL2VA continuations (results/20261103-identity-drift/): window 1
holds the faces (its keyframe is the reference image), windows 2+ see only the previous tail and the Qwen3-VL picture
tokens of the prompt, so every hand-over re-samples the faces from an already drifted copy. Two levers live here:

``end`` (trained mechanism): every window after the first is sampled as FL2VA with the request's reference image as
its LAST keyframe (``keyframe_frame_indices=[-1]``, the same rows window 1 used as its first keyframe), on top of the
pinned prefix. ``H3_TP2_ID_END_EXTEND=<frames, multiple of 17>`` renders each such window that much longer, puts the
keyframe at the extended end and crops the result back to the planned window, so the clip is pulled toward the
reference faces without ever showing the return to the reference pose ("hidden end anchor").

``toff0`` (Ref2VA continuation only): every window keeps local RoPE time (``temporal_offset`` 0) so the static reference
image stays as close in time to window 6 as to window 1.  The older ``H3_TP2_WINDOW_TEMPORAL_OFFSET`` switch wraps
upstream's loop, which the default edited loop (``H3_TP2_CONT_EDIT=2``) replaces, so it is a no-op on the current stack.
2026-11-08 (results/20261106-voice/ section 13): on Polish dialogue chains ``toff0`` halves the real slips of the tongue
of windows 2-6 but the voice drifts along the chain again (UTMOS -0.15 to -0.25 over windows 2-6, growing).

``toffcap`` (Ref2VA continuation only): the same shift capped at ``H3_TP2_ID_LOCK_TOFF_CAP`` RoPE units (default 425,
the shift of window 2 at 24 fps: 255 frames * 40 / 24), so every later window sits where window 2 sits, inside the
time span of a training clip and still not at a clip's opening.  The middle ground between upstream's shift (window 6
at 2125 units, 3.5 training clips away from its reference) and ``toff0``.

``stats`` / ``blend`` (latent face lock, post-hoc on the returned window, after the look-lock): inside the face boxes
(``H3_TP2_ID_LOCK_BOXES``, normalised x0,y0,x1,y1 per face, ``;``-separated) each free token of a later window is
aligned to window 1's opening block by an integer latent shift (search radius ``H3_TP2_ID_LOCK_RADIUS`` cells) and
either its per-channel statistics (``stats``) or its content (``blend``, optionally low-passed on the 2x2 DiT token grid
with ``H3_TP2_ID_LOCK_LOWPASS=1``) are moved toward the reference of the same VAE token class (block-start token vs
the other four) by ``H3_TP2_ID_LOCK_STRENGTH``, feathered over ``H3_TP2_ID_LOCK_FEATHER`` cells and ramped per 17-frame
block over ``H3_TP2_ID_LOCK_RAMP`` of the free frames.  ``measure`` only logs the aligned latent face similarity.

Everything is deterministic elementwise work on tensors both ranks already hold identically, so the two TP ranks stay
in lockstep (the same contract as the look-lock).  Not checkpoint-aware: the wrapper state is not part of
``snapshot_state``.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

logger = logging.getLogger("vllm_omni.h3_tp2.identity_lock")

BLOCK_TOKENS = 5          # VAE: 5 latent tokens per 17-frame block, token 5k is the padded block-start token
MODES = ("end", "toff0", "toffcap", "stats", "blend", "measure")


def video_t(frames: int) -> int:
    return (frames - 5) // 17 * 5 + 2


@dataclass
class Config:
    modes: frozenset = frozenset()
    extend: int = 0                       # frames added after the planned window for the hidden end anchor (17k)
    boxes: tuple = ()                     # ((x0, y0, x1, y1), ...) normalised to the canvas
    strength: float = 0.3
    radius: int = 2                       # alignment search, latent cells (16 px)
    feather: int = 2                      # cosine falloff outside the box, latent cells
    ramp: float = 0.25                    # fraction of the free frames until full strength (block-constant steps)
    lowpass: bool = True
    toff_cap: float = 425.0               # toffcap: the largest per-window RoPE shift, units of 40/24 per frame

    @property
    def active(self) -> bool:
        return bool(self.modes)

    @classmethod
    def from_env(cls, environ=os.environ) -> "Config":
        raw = environ.get("H3_TP2_ID_LOCK", "").strip().lower()
        if not raw or raw in ("0", "off"):
            return cls()
        modes = frozenset(m.strip() for m in raw.split(",") if m.strip())
        bad = modes - set(MODES)
        if bad:
            raise ValueError(f"H3_TP2_ID_LOCK: unknown mode(s) {sorted(bad)}; use {', '.join(MODES)}")
        extend = int(environ.get("H3_TP2_ID_END_EXTEND", "0").strip() or 0)
        if extend < 0 or extend % 17:
            raise ValueError("H3_TP2_ID_END_EXTEND must be a non-negative multiple of 17 frames")
        boxes = parse_boxes(environ.get("H3_TP2_ID_LOCK_BOXES", ""))
        if modes & {"stats", "blend", "measure"} and not boxes:
            raise ValueError("H3_TP2_ID_LOCK stats/blend/measure need H3_TP2_ID_LOCK_BOXES")
        if {"toff0", "toffcap"} <= modes:
            raise ValueError("H3_TP2_ID_LOCK: toff0 and toffcap exclude each other")
        toff_cap = float(environ.get("H3_TP2_ID_LOCK_TOFF_CAP", "425") or 425)
        if toff_cap < 0:
            raise ValueError("H3_TP2_ID_LOCK_TOFF_CAP must be >= 0")
        return cls(modes=modes, extend=extend, boxes=boxes, toff_cap=toff_cap,
                   strength=float(environ.get("H3_TP2_ID_LOCK_STRENGTH", "0.3") or 0.3),
                   radius=int(environ.get("H3_TP2_ID_LOCK_RADIUS", "2") or 2),
                   feather=int(environ.get("H3_TP2_ID_LOCK_FEATHER", "2") or 2),
                   ramp=float(environ.get("H3_TP2_ID_LOCK_RAMP", "0.25") or 0.25),
                   lowpass=environ.get("H3_TP2_ID_LOCK_LOWPASS", "1").strip() == "1")


def parse_boxes(text: str) -> tuple:
    out = []
    for part in (p.strip() for p in text.split(";")):
        if not part:
            continue
        x0, y0, x1, y1 = (float(v) for v in part.split(","))
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"H3_TP2_ID_LOCK_BOXES: invalid box {part!r} (normalised x0,y0,x1,y1)")
        out.append((x0, y0, x1, y1))
    return tuple(out)


# ----------------------------------------------------------------------------------------------- end anchor (args)

def _later_window(args: dict) -> bool:
    return args.get("video_edit_mask_rows") is not None and args.get("task") == "t2va"


def end_anchor_args(args: dict, ref: dict, extend: int) -> tuple[dict, tuple[int, int] | None]:
    """Return (args for the real diffuse, crop) for a later window: FL2VA with the reference as the last keyframe and,
    with ``extend`` > 0, a longer canvas whose surplus is cropped off afterwards (crop = (latent_t, audio_t))."""
    a = dict(args)
    shape = tuple(int(v) for v in ref["shape"])
    a.update(task="fl2va", keyframe_frame_indices=[-1], visual_condition=ref["rows"], visual_condition_shape=shape,
             visual_condition_shapes=[shape], ref_blocks=None, audio_condition=None, audio_condition_lengths=None,
             ref_audio_t=None)
    if not extend:
        return a, None
    if a.get("locked_audio_rows") is not None or a.get("pad_seq_len") is not None:
        logger.warning("h3_tp2 identity end anchor: extension skipped (locked source audio or pinned sequence length)")
        return a, None
    lt, at = int(a["latent_t"]), int(a["audio_t"])
    nf = int(a["num_frames"]) + extend
    lt2, at2 = video_t(nf), at + round(extend * 5 / 3)
    clean, mask, restore = a["video_edit_clean_rows"], a["video_edit_mask_rows"], a["video_edit_restore_mask_rows"]
    rpf = clean.shape[0] // lt
    pad = (lt2 - lt) * rpf
    a["video_edit_clean_rows"] = torch.cat((clean, clean.new_zeros((pad, clean.shape[1]))))
    a["video_edit_mask_rows"] = torch.cat((mask, mask.new_ones(pad)))
    a["video_edit_restore_mask_rows"] = torch.cat((restore, restore.new_ones((pad, *restore.shape[1:]))))
    if a.get("audio_edit_mask_rows") is not None:   # channel-major [c * audio_t + i]
        ca, ma, ra = a["audio_edit_clean_rows"], a["audio_edit_mask_rows"], a["audio_edit_restore_mask_rows"]
        nc = ca.shape[0] // at
        ca2 = ca.new_zeros((nc * at2, ca.shape[1])); ma2 = ma.new_ones(nc * at2); ra2 = ra.new_ones((nc * at2, *ra.shape[1:]))
        for c in range(nc):
            ca2[c * at2:c * at2 + at] = ca[c * at:(c + 1) * at]
            ma2[c * at2:c * at2 + at] = ma[c * at:(c + 1) * at]
            ra2[c * at2:c * at2 + at] = ra[c * at:(c + 1) * at]
        a["audio_edit_clean_rows"], a["audio_edit_mask_rows"], a["audio_edit_restore_mask_rows"] = ca2, ma2, ra2
    a.update(num_frames=nf, latent_t=lt2, audio_t=at2)
    return a, (lt, at)


def wrap_inner(diffuse, cfg: Config):
    """Around the real ``diffuse``: capture window 1's keyframe rows, re-use them as the last keyframe later."""
    state = {"ref": None, "index": 0}

    def _diffuse(**args):  # noqa: ANN003
        state["index"] += 1
        if state["index"] == 1 and args.get("task") == "fl2va" and list(args.get("keyframe_frame_indices") or []) == [0]:
            shape = args.get("visual_condition_shape") or (args.get("visual_condition_shapes") or [None])[0]
            if args.get("visual_condition") is not None and shape is not None:
                state["ref"] = {"rows": args["visual_condition"], "shape": shape}
        if "toff0" in cfg.modes and args.get("task") == "ref2va" and args.get("temporal_offset"):
            args = {**args, "temporal_offset": 0.0}
        if "toffcap" in cfg.modes and args.get("task") == "ref2va" and (args.get("temporal_offset") or 0) > cfg.toff_cap:
            args = {**args, "temporal_offset": float(cfg.toff_cap)}
        if "end" in cfg.modes and state["ref"] is not None and _later_window(args):
            a, crop = end_anchor_args(args, state["ref"], cfg.extend)
            logger.info("h3_tp2 identity end anchor (call %d, seed %s): reference as last keyframe, canvas %d -> %d frames",
                        state["index"], args.get("seed"), int(args["num_frames"]), int(a["num_frames"]))
            video, audio = diffuse(**a)
            if crop is not None:
                video, audio = video[:, :, :crop[0]].contiguous(), audio[..., :crop[1]].contiguous()
            return video, audio
        return diffuse(**args)

    return _diffuse


# ---------------------------------------------------------------------------------------- latent face lock (post)

def face_cells(box, h: int, w: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    r0, r1 = int(math.floor(y0 * h)), int(math.ceil(y1 * h))
    c0, c1 = int(math.floor(x0 * w)), int(math.ceil(x1 * w))
    return r0, max(r1, r0 + 1), c0, max(c1, c0 + 1)


def feather_mask(cells, h: int, w: int, feather: int, device=None) -> torch.Tensor:
    """1 inside the box, cosine falloff to 0 at ``feather`` cells outside it."""
    r0, r1, c0, c1 = cells
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    dy = torch.clamp(torch.maximum(r0 - ys, ys - (r1 - 1)), min=0)
    dx = torch.clamp(torch.maximum(c0 - xs, xs - (c1 - 1)), min=0)
    d = torch.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    if feather <= 0:
        return (d == 0).float()
    return torch.where(d >= feather + 1, torch.zeros_like(d), 0.5 * (1 + torch.cos(math.pi * torch.clamp(d / (feather + 1), max=1.0))))


def _centered_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    b = b - b.mean(dim=(-2, -1), keepdim=True)
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8))


def align_shift(x: torch.Tensor, ref: torch.Tensor, cells, radius: int) -> tuple[tuple[int, int], float]:
    """Integer (dy, dx) such that x[box + shift] best matches ref[box] (centered cosine over channels and cells)."""
    r0, r1, c0, c1 = cells
    h, w = x.shape[-2:]
    target = ref[:, r0:r1, c0:c1]
    best, score = (0, 0), -2.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if r0 + dy < 0 or r1 + dy > h or c0 + dx < 0 or c1 + dx > w:
                continue
            s = _centered_cos(x[:, r0 + dy:r1 + dy, c0 + dx:c1 + dx], target)
            if s > score + 1e-9:   # strict: ties keep the smaller-|shift| candidate found first in this order
                best, score = (dy, dx), s
    return best, score


def _shift_map(m: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """m shifted by (dy, dx) with zero fill (no wrap-around)."""
    out = torch.zeros_like(m)
    h, w = m.shape[-2:]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[..., yd, xd] = m[..., ys, xs]
    return out


def _lowpass(d: torch.Tensor) -> torch.Tensor:
    """Average over the 2x2 DiT token cells, constant inside each token (pads an odd edge)."""
    h, w = d.shape[-2:]
    p = F.pad(d[None], (0, w % 2, 0, h % 2), mode="replicate")
    return F.interpolate(F.avg_pool2d(p, 2), scale_factor=2, mode="nearest")[0, ..., :h, :w]


def reference_tokens(first_window: torch.Tensor) -> dict:
    """Window 1's opening block: token 0 (class A, the reference keyframe re-rendered) and tokens 1..4 (class B)."""
    v = first_window[0].float()
    return {"A": v[:, 0].clone(), "B": v[:, 1:BLOCK_TOKENS].mean(dim=1).clone()}


def block_ramp(n_tokens: int, ov: int, ramp: float, device=None) -> torch.Tensor:
    """Weight per token of the free part: constant within a 17-frame block, rising by block, full after ``ramp``."""
    blocks = (torch.arange(ov, ov + n_tokens, device=device) // BLOCK_TOKENS).float()
    first = float(ov // BLOCK_TOKENS)
    n_ramp = max(1.0, math.ceil(ramp * (n_tokens / BLOCK_TOKENS)))
    return torch.clamp((blocks - first + 1) / n_ramp, max=1.0)


def lock_faces(ref: dict, window: torch.Tensor, ov: int, cfg: Config, *, index: int, apply: set) -> torch.Tensor:
    """Apply ``stats`` and/or ``blend`` (``apply``) to the free tokens of ``window`` [1, C, T, H, W]; always logs."""
    x = window[0].float().clone()
    c, t, h, w = x.shape
    wt = block_ramp(t - ov, ov, cfg.ramp, device=x.device)
    for fi, box in enumerate(cfg.boxes):
        cells = face_cells(box, h, w)
        base = feather_mask(cells, h, w, cfg.feather, device=x.device)
        shifts, before = [], []
        for j in range(ov, t):
            cls = "A" if j % BLOCK_TOKENS == 0 else "B"
            sh, s = align_shift(x[:, j], ref[cls], cells, cfg.radius)
            shifts.append(sh)
            before.append(s)
        # one shift per 17-frame block (median of its tokens): the face may move between blocks, not jitter inside one
        for b0 in range(ov, t, BLOCK_TOKENS):
            idx = list(range(b0 - b0 % BLOCK_TOKENS, b0 - b0 % BLOCK_TOKENS + BLOCK_TOKENS))
            sel = [shifts[k - ov] for k in idx if ov <= k < t]
            med = (sorted(s[0] for s in sel)[len(sel) // 2], sorted(s[1] for s in sel)[len(sel) // 2])
            for k in idx:
                if ov <= k < t:
                    shifts[k - ov] = med
        if "stats" in apply:
            for cls in ("A", "B"):
                toks = [j for j in range(ov, t) if (j % BLOCK_TOKENS == 0) == (cls == "A")]
                if not toks:
                    continue
                r0, r1, c0, c1 = cells
                patches = torch.stack([x[:, j, r0 + shifts[j - ov][0]:r1 + shifts[j - ov][0], c0 + shifts[j - ov][1]:c1 + shifts[j - ov][1]] for j in toks], 1)
                m, sd = patches.mean(dim=(1, 2, 3)), patches.std(dim=(1, 2, 3))
                rp = ref[cls][:, r0:r1, c0:c1]
                mr, sr = rp.mean(dim=(1, 2)), rp.std(dim=(1, 2))
                gain = 1.0 + cfg.strength * ((sr / sd.clamp_min(1e-6)).clamp(0.8, 1.25) - 1.0)
                shift = cfg.strength * (mr - m)
                for j in toks:
                    mk = _shift_map(base, *shifts[j - ov]) * wt[j - ov]
                    y = (x[:, j] - m[:, None, None]) * gain[:, None, None] + m[:, None, None] + shift[:, None, None]
                    x[:, j] = x[:, j] + mk * (y - x[:, j])
        if "blend" in apply:
            for j in range(ov, t):
                cls = "A" if j % BLOCK_TOKENS == 0 else "B"
                dy, dx = shifts[j - ov]
                target = _shift_map(ref[cls], dy, dx)        # the reference face moved onto this token's face position
                d = target - x[:, j]
                if cfg.lowpass:
                    d = _lowpass(d)
                mk = _shift_map(base, dy, dx) * (cfg.strength * wt[j - ov])
                x[:, j] = x[:, j] + mk * d
        after = [align_shift(x[:, j], ref["A" if j % BLOCK_TOKENS == 0 else "B"], cells, cfg.radius)[1] for j in range(ov, t)]
        moved = sum(1 for s in shifts if s != (0, 0))
        logger.info("h3_tp2 identity lock window %d face %d: latent face similarity %.3f -> %.3f (%s), shifted tokens %d/%d",
                    index, fi, sum(before) / max(1, len(before)), sum(after) / max(1, len(after)),
                    "+".join(sorted(apply)) or "measure", moved, len(shifts))
    if not apply:
        return window
    out = window.clone()
    out[0] = x.to(window.dtype)
    return out


def _pinned_tokens(args: dict, video: torch.Tensor) -> int | None:
    mask = args.get("video_edit_mask_rows")
    if mask is None or not isinstance(video, torch.Tensor) or video.dim() != 5:
        return None
    rpf = (video.shape[3] // 2) * (video.shape[4] // 2)
    return int((mask == 0).sum().item()) // rpf


def wrap_outer(diffuse, cfg: Config):
    """Around the look-locked ``diffuse``: the latent face lock on every later window."""
    state = {"ref": None, "index": 0}
    apply = set(cfg.modes) & {"stats", "blend"}

    def _diffuse(**args):  # noqa: ANN003
        video, audio = diffuse(**args)
        if not (cfg.modes & {"stats", "blend", "measure"}) or not isinstance(video, torch.Tensor) or video.dim() != 5:
            return video, audio
        ov = _pinned_tokens(args, video)
        state["index"] += 1
        if state["ref"] is None and ov is None:
            state["ref"] = reference_tokens(video)
            return video, audio
        if state["ref"] is not None and ov:
            try:
                video = lock_faces(state["ref"], video, ov, cfg, index=state["index"], apply=apply)
            except Exception as e:  # noqa: BLE001  (an experiment never breaks a render)
                logger.warning("h3_tp2 identity lock skipped (%s)", e)
        return video, audio

    for name in ("snapshot_state", "restore_state"):
        if hasattr(diffuse, name):
            setattr(_diffuse, name, getattr(diffuse, name))
    return _diffuse


# ------------------------------------------------------------------------------------------------------- install

def install(cm_module, environ=os.environ) -> bool:
    """Compose the identity wrappers around ``cm_module.wrap_diffuse`` (the look-lock / seam-retry wrapper) when
    ``H3_TP2_ID_LOCK`` is set.  Returns True when installed.  Idempotent."""
    cfg = Config.from_env(environ)
    if not cfg.active or getattr(cm_module, "_h3_tp2_identity_lock", False):
        return False
    orig = cm_module.wrap_diffuse

    def wrap_diffuse(diffuse, **kw):  # noqa: ANN001
        return wrap_outer(orig(wrap_inner(diffuse, cfg), **kw), cfg)

    cm_module.wrap_diffuse = wrap_diffuse
    cm_module._h3_tp2_identity_lock = True
    logger.warning("h3_tp2: EXPERIMENT identity lock %s (end extend %d, boxes %d, strength %.2f, radius %d, lowpass %s, toff cap %.0f; changes outputs)",
                   ",".join(sorted(cfg.modes)), cfg.extend, len(cfg.boxes), cfg.strength, cfg.radius, cfg.lowpass, cfg.toff_cap)
    return True
