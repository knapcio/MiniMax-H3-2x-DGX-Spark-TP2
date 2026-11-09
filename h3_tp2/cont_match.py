"""Continuation hand-over diagnostics and an optional per-channel latent match (H3_TP2_CONT_MATCH=1).

Upstream renders every continuation window fresh under a latent-tail guide and appends only the frames after the
hidden overlap.  Measured 2026-10-29 (owner's eye + per-frame luma): the first frames of window 2 are ~19 % brighter
than the last frames of window 1 and the texture changes, so the hand-over is visible.  This wraps the per-window
``diffuse`` call to log, for every window after the first, how the regenerated overlap compares with the previous
window's tail (per-channel mean/std of the video latents) and, when enabled, to apply the per-channel affine map that
makes the regenerated overlap's statistics equal to the tail's, to the whole new window (video latents only; audio
untouched).  The matched window then continues from the exposure and contrast it was supposed to continue from.
"""
from __future__ import annotations

import logging
import math
import os

import torch

from h3_tp2 import pace as _pace
from h3_tp2 import speech_early as _early

logger = logging.getLogger("vllm_omni.h3_tp2.cont_match")


def _stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [B, C, T, H, W] -> per-channel mean/std over B, T, H, W
    xf = x.float()
    dims = (0, 2, 3, 4)
    return xf.mean(dim=dims), xf.std(dim=dims)


BLOCK_TOKENS = 5   # the VAE decodes 5 latent tokens per 17-frame block; token 5k is the padded clip-start token


def block_start_mask(n_tokens: int, device, offset: int = 0) -> torch.Tensor:
    """True for the tokens the VAE treats as a block start (local index + offset = 0 mod 5).  A continuation window
    shifts by 75 tokens (255 frames = 15 blocks) in the stitched latent, so the local phase is the stitched phase."""
    return ((torch.arange(n_tokens, device=device) + offset) % BLOCK_TOKENS) == 0


def class_stats(x: torch.Tensor, offset: int = 0) -> dict:
    """Per-channel mean/std over all tokens and separately over the block-start tokens (A) and the others (B).
    The two classes have different statistics (A encodes one real frame + 3 padding frames); a per-channel affine
    fitted on all tokens moves A differently from B and the decoder renders that as a darker first frame per block
    (the 17-frame flicker: -0.66 Y raw -> -2.5 Y with the lock on 2026-11-02)."""
    a = block_start_mask(x.shape[2], x.device, offset)
    out = {"all": _stats(x)}
    out["A"] = _stats(x[:, :, a]) if int(a.sum()) >= 2 else out["all"]
    out["B"] = _stats(x[:, :, ~a]) if int((~a).sum()) >= 2 else out["all"]
    return out


def pos_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position first and second moments over batch and time of [B, C, T, H, W] latents: two [C, H, W] maps."""
    xf = x.float()
    return xf.mean(dim=(0, 2)), (xf * xf).mean(dim=(0, 2))


def class_pos_moments(x: torch.Tensor, offset: int = 0) -> dict:
    """``pos_moments`` over all tokens and over the two token classes of ``class_stats``."""
    a = block_start_mask(x.shape[2], x.device, offset)
    out = {"all": pos_moments(x)}
    out["A"] = pos_moments(x[:, :, a]) if int(a.sum()) >= 2 else out["all"]
    out["B"] = pos_moments(x[:, :, ~a]) if int((~a).sum()) >= 2 else out["all"]
    return out


def robust_params(ref_maps: tuple[torch.Tensor, torch.Tensor], free: torch.Tensor, k: float = 3.0):
    """Reference and window statistics estimated on the positions that did NOT change between them.

    The look-lock's per-channel affine compares the free frames' mean/std with the reference tail's.  Both are
    content-dependent: a character who is hidden in the reference tail and back in the new window raises the channel
    means of the new window by her share of the frame, and the lock then shifts the whole frame (and her) away from
    her own colours (the orange coat and the changed face of 2026-11-06, `results/20261106-whole-person/`).  Drift,
    the thing the lock is for, is a whole-frame change; a character is a local one.  So the difference of the
    per-position temporal means (reference - window) is taken per channel, its median and median absolute deviation
    define the inliers (|d - median| <= k * MAD: the positions that moved like the majority), and the means and stds
    of both tensors are computed over time and those positions only.  Returns ``(mean_ref, std_ref, mean_free,
    std_free, inlier_fraction)``, each per channel.
    """
    m1_r, m2_r = ref_maps
    m1_f, m2_f = pos_moments(free)
    d = (m1_r - m1_f).flatten(1)
    med = d.median(dim=1).values
    mad = (d - med[:, None]).abs().median(dim=1).values
    inl = ((d - med[:, None]).abs() <= (k * mad + 1e-6)[:, None]).float()   # [C, P]
    n = inl.sum(dim=1).clamp_min(1.0)

    def _joint(m1, m2):
        mu = (m1.flatten(1) * inl).sum(dim=1) / n
        ex2 = (m2.flatten(1) * inl).sum(dim=1) / n
        return mu, (ex2 - mu * mu).clamp_min(0.0).sqrt()

    mu_r, sd_r = _joint(m1_r, m2_r)
    mu_f, sd_f = _joint(m1_f, m2_f)
    return mu_r, sd_r, mu_f, sd_f, inl.mean(dim=1)


def match_window(prev_tail: torch.Tensor, new_window: torch.Tensor, overlap_v: int, *, apply: bool, index: int) -> torch.Tensor:
    """Log the hand-over mismatch and, if ``apply``, return the affine-matched new window (same dtype/shape)."""
    regen = new_window[:, :, :overlap_v]
    m0, s0 = _stats(prev_tail)
    m1, s1 = _stats(regen)
    mfree, _ = _stats(new_window[:, :, overlap_v : overlap_v + max(1, overlap_v)])
    logger.info(
        "h3_tp2 hand-over window %d: guide tail mean %.4f std %.4f | regenerated overlap mean %.4f std %.4f | next frames mean %.4f | "
        "per-channel |dmean| max %.4f, std ratio %.3f..%.3f",
        index, m0.mean().item(), s0.mean().item(), m1.mean().item(), s1.mean().item(), mfree.mean().item(),
        (m1 - m0).abs().max().item(), (s0 / s1.clamp_min(1e-6)).min().item(), (s0 / s1.clamp_min(1e-6)).max().item(),
    )
    if not apply:
        return new_window
    gain = (s0 / s1.clamp_min(1e-6)).clamp(0.5, 2.0).view(1, -1, 1, 1, 1)
    offset = (m0 - m1 * gain.view(-1)).view(1, -1, 1, 1, 1)
    out = (new_window.float() * gain + offset).to(new_window.dtype)
    m2, s2 = _stats(out[:, :, :overlap_v])
    logger.info("h3_tp2 hand-over window %d: matched overlap mean %.4f std %.4f (gain %.3f..%.3f)", index,
                m2.mean().item(), s2.mean().item(), gain.min().item(), gain.max().item())
    return out


def seam_ratio(video: torch.Tensor, overlap_v: int) -> tuple[float, float, float]:
    """Latent-space seam check of one sampled window: mean |v[t+1]-v[t]| at the boundary between the last overlap
    frame and the first free frame, against the median of the same quantity over the window's free part.
    Returns (boundary, median, ratio).  A cut right after the prefix shows as a ratio far above 1."""
    v = video.float()
    d = (v[:, :, 1:] - v[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))   # per transition
    boundary = float(d[overlap_v - 1].item())
    free = d[overlap_v:]
    median = float(free.median().item()) if free.numel() else boundary
    return boundary, median, boundary / max(median, 1e-6)


def cut_ratio(video: torch.Tensor, overlap_v: int) -> tuple[float, float, float, int]:
    """Interior cut check of one sampled window: every free transition's mean |v[t+1]-v[t]| against the median of the
    free transitions of its own phase (t mod 5: the VAE's 17-frame blocks give the transitions into and out of a
    block-start token 5k their own, larger level), the largest such ratio wins.  Returns (peak, median of the peak's
    phase, ratio, token t of the peak's transition t -> t + 1).  Calibrated 2026-11-05 on the 118 s Polish scene
    (960x544, 277-frame windows, 18 saved windows): clean windows 1.07-1.30, the window that cut to a close-up 2 s
    after its hand-over 1.96 (the cut fell on a transition into a block start, where a pooled median hides it; the
    boundary seam check had passed that window at 1.0), a window that cut right after the prefix 2.72, the re-sampled
    window 11 that followed the cut-away 1.87.  Default limit 1.7 (H3_TP2_CONT_CUT_RATIO)."""
    v = video.float()
    d = (v[:, :, 1:] - v[:, :, :-1]).abs().mean(dim=(0, 1, 3, 4))   # per transition t: token t -> t + 1
    n = d.numel()
    lo = max(int(overlap_v), 0)
    if n - lo < 3 * BLOCK_TOKENS:
        return float("nan"), float("nan"), 0.0, -1
    t = torch.arange(n, device=d.device)
    best = (float("nan"), float("nan"), 0.0, -1)
    for phase in range(BLOCK_TOKENS):
        sel = (t >= lo) & (t % BLOCK_TOKENS == phase)
        if int(sel.sum()) < 3:
            continue
        free = d[sel]
        median = float(free.median().item())
        i = int(free.argmax().item())
        peak = float(free[i].item())
        ratio = peak / max(median, 1e-6)
        if ratio > best[2]:
            best = (peak, median, ratio, int(t[sel][i].item()))
    return best


def _log_post_correction_seam(video: torch.Tensor, overlap_v: int, index: int) -> None:
    """Observe the returned window, independently of correction and retry selection."""
    try:
        with torch.no_grad():
            boundary, median, ratio = seam_ratio(video.detach(), overlap_v)
        if not all(math.isfinite(value) for value in (boundary, median, ratio)):
            raise ValueError("nonfinite post-correction seam measurement")
        logger.info(
            "h3_tp2 seam postcheck window %d: returned window after match/look-lock stage, "
            "before downstream crossfade: boundary %.4f median %.4f ratio %.2f "
            "(diagnostic only; not the retry score)", index, boundary, median, ratio,
        )
    except Exception as exc:  # noqa: BLE001
        # Even a diagnostic logging failure must not turn an observation into
        # a render failure. The normal AV correction has already run.
        try:
            logger.warning("h3_tp2 seam postcheck window %d skipped (%s)", index, exc)
        except Exception:  # noqa: BLE001
            pass


def window_map(text: str) -> dict[int, int]:
    """``"3,5"`` -> {3: 1, 5: 1} and ``"3:7919,5:11"`` -> {3: 7919, 5: 11}: window indices (from 1) with an optional value."""
    out: dict[int, int] = {}
    for item in (text or "").replace(" ", "").split(","):
        if not item:
            continue
        key, _, value = item.partition(":")
        try:
            index, number = int(key), int(value) if value else 1
        except ValueError:
            raise ValueError(f"expected <window>[:<number>] items separated by commas, got {item!r}") from None
        if index < 1:
            raise ValueError(f"window indices start at 1, got {item!r}")
        out[index] = number
    return out


def wrap_diffuse(diffuse, *, apply: bool, overlap_v_hint: int | None = None, retries: int = 0, max_ratio: float = 2.0,
                 lock: str = "", lock_strength: float = 1.0, lock_ramp: float = 1.0, lock_audio: bool = False,
                 lock_std: float | None = None, lock_ref: str = "window", lock_classes: bool = False,
                 max_cut: float = 0.0, identity_fn=None, max_unknown: float = 0.0, lock_robust: bool = False,
                 speech_fn=None, max_cer: float = 0.0, early_fn=None, early_steps=()):  # noqa: ANN001
    """Return a ``diffuse`` replacement for one continuation request (state: the previous window's video latents).

    With ``retries`` > 0 a window whose latent seam ratio exceeds ``max_ratio``, or (``max_cut`` > 0) whose interior
    cut ratio exceeds ``max_cut``, is sampled again with another seed (up to ``retries`` times); the attempt with the
    lowest score (the larger of the two ratios relative to their limits) is kept.  ``identity_fn(video, skip_frames)``
    (with ``max_unknown`` > 0) adds the identity guard, which also judges the first window: it returns
    ``(score, faces, unknown_frac)`` for the window's decoded frames after ``skip_frames`` (``h3_tp2.identity_guard``);
    a score >= 1.0 means more faces than tolerated match no reference face, and the window is re-sampled.
    ``speech_fn(audio, index)`` (with ``max_cer`` > 0) adds the speech guard (``h3_tp2.speech_guard``): it returns
    ``(score, cer, chars)`` for the window's decoded and transcribed audio against the lines of its prompt; a score
    > 1.0 means the lines came out with more than the tolerated character error rate, and the window is re-sampled.
    ``early_fn(x0, index, attempt, step, args, may_stop)`` (with ``early_steps``, 1-based denoising steps) is handed
    the loop's clean estimate of the audio rows at those steps of every sampling (``h3_tp2.speech_early``).  It may
    raise ``speech_early.Stop`` when ``may_stop`` is true (another attempt can follow): the sampling ends there, the
    attempt counts as failed and the window is sampled again with the next seed.  A sampling that is not stopped is
    untouched.
    ``lock_robust`` estimates the look-lock's statistics on the positions that did not change between the reference
    tail and the new window (``robust_params``), so a character who returns from behind something is not recoloured.
    """
    state = {"prev": None, "index": 0, "ref": None, "ref_audio": None, "ref_maps": None}
    # Read once per request wrapper, covering both plugin continuation routes.
    # Disabled: no extra tensor operations, measurements or log records.
    postcheck = os.environ.get("H3_TP2_SEAM_POSTCHECK", "0").strip() == "1"
    # Measurement only (they change the output): H3_TP2_CONT_RETRY_FORCE=<k[,k]> samples those windows a second time
    # whatever the checks say and drops the first attempt; H3_TP2_CONT_RETRY_SEED_SHIFT=<k:n[,k:n]> adds n to the seed of
    # window k's attempts.  FORCE=3 and SEED_SHIFT=3:7919 must give the same clip: a retry is a first attempt with
    # another seed and nothing else.  Read from the environment only, so every rank decides alike.
    force = set(window_map(os.environ.get("H3_TP2_CONT_RETRY_FORCE", "")))
    shift = window_map(os.environ.get("H3_TP2_CONT_RETRY_SEED_SHIFT", ""))

    def _overlap_audio(args):  # noqa: ANN001
        lengths = args.get("audio_condition_lengths") or []
        if lengths:
            return int(lengths[-1])
        mask = args.get("audio_edit_mask_rows")
        if mask is not None:
            return int((mask == 0).sum().item() // 2)
        return None

    def _overlap(args):  # noqa: ANN001
        blocks = args.get("ref_blocks") or []
        guide = next((b for b in reversed(blocks) if isinstance(b, dict) and b.get("kind") == "latent_guide"), None)
        if guide is not None:
            return int(guide["latent_t"])
        mask = args.get("video_edit_mask_rows")
        if mask is not None and overlap_v_hint:
            return overlap_v_hint
        return None

    guard = (identity_fn is not None and max_unknown > 0) or (speech_fn is not None and max_cer > 0)

    def _diffuse(**args):  # noqa: ANN003
        state["index"] += 1
        ov = _overlap(args) if state["prev"] is not None else None
        best = None
        forced = state["index"] in force
        attempts = 1 + max(retries if (ov or guard) else 0, 1 if forced else 0)
        for attempt in range(attempts):
            a = dict(args)
            offset = 7919 * attempt + shift.get(state["index"], 0)
            if offset and "seed" in a and a["seed"] is not None:
                a["seed"] = int(a["seed"]) + offset
            # giving an attempt up early needs an attempt after it; a forced re-sample is a measurement and runs whole
            may_stop = attempt < attempts - 1 and not forced
            listen = None
            if early_fn is not None and early_steps:
                def listen(step, x0, a=a, n=attempt + 1, may_stop=may_stop):  # noqa: ANN001
                    early_fn(x0, state["index"], n, step, a, may_stop)
            started = _pace.clock()
            try:
                with _early.tapping(early_steps, listen):
                    video, audio = diffuse(**a)
            except _early.Stop as stop:
                if not may_stop:
                    raise RuntimeError(f"h3_tp2: window {state['index']} attempt {attempt + 1} was stopped early with no "
                                       "attempt left to follow it") from stop
                logger.warning("h3_tp2 speech check window %d attempt %d (seed %s): given up at step %d, the lines already came "
                               "out with a character error rate of %.4f (limit %.3f); re-sampling with another seed",
                               state["index"], attempt + 1, a.get("seed"), stop.step, stop.cer, stop.limit)
                continue
            _pace.report(state["index"], attempt + 1, a, _pace.clock() - started)   # wall time only, see h3_tp2/pace.py
            drop = forced and attempt == 0   # the checks still run on the dropped attempt, as they do before a real retry
            if drop:
                logger.warning("h3_tp2 MEASUREMENT window %d: forced re-sample (H3_TP2_CONT_RETRY_FORCE), the first attempt is dropped",
                               state["index"])
            if not (ov or guard) or not (isinstance(video, torch.Tensor) and video.dim() == 5):
                if drop:
                    continue
                best = (video, audio, 0.0)
                break
            r = cr = 0.0
            if ov:
                try:
                    b, m, r = seam_ratio(video, ov)
                    logger.info("h3_tp2 seam check window %d attempt %d (seed %s): boundary %.4f median %.4f ratio %.2f",
                                state["index"], attempt + 1, a.get("seed"), b, m, r)
                    if max_cut > 0:
                        cp, cm, cr, ct = cut_ratio(video, ov)
                        logger.info("h3_tp2 cut check window %d attempt %d (seed %s): peak %.4f at token %d median %.4f ratio %.2f",
                                    state["index"], attempt + 1, a.get("seed"), cp, ct, cm, cr)
                except Exception as e:  # noqa: BLE001
                    logger.warning("h3_tp2 seam check skipped (%s)", e)
                    if drop:
                        continue
                    best = (video, audio, 0.0)
                    break
            iscore = sscore = 0.0
            if identity_fn is not None and max_unknown > 0:
                try:
                    # the first window is judged whole; a continued one from its first free frame (4 frames per token)
                    iscore, faces, unknown = identity_fn(video, 4 * int(ov) if ov else 0)
                    logger.info("h3_tp2 identity check window %d attempt %d (seed %s): %d face(s), %.0f %% unknown, score %.2f",
                                state["index"], attempt + 1, a.get("seed"), int(faces), 100.0 * unknown, iscore)
                except Exception as e:  # noqa: BLE001  (the guard never breaks a render)
                    logger.warning("h3_tp2 identity check skipped (%s)", e)
                    iscore = 0.0
            if speech_fn is not None and max_cer > 0:
                try:
                    sscore, cer, chars = speech_fn(audio, state["index"])
                    logger.info("h3_tp2 speech check window %d attempt %d (seed %s): CER %.4f over %d script chars, score %.2f",
                                state["index"], attempt + 1, a.get("seed"), cer, int(chars), sscore)
                except Exception as e:  # noqa: BLE001  (the guard never breaks a render)
                    logger.warning("h3_tp2 speech check skipped (%s)", e)
                    sscore = 0.0
            if drop:
                continue
            score = max(r / max(max_ratio, 1e-9) if ov else 0.0, cr / max_cut if max_cut > 0 else 0.0, iscore, sscore)
            if best is None or score < best[2]:
                best = (video, audio, score)
            if (not ov or (r <= max_ratio and (max_cut <= 0 or cr <= max_cut))) and iscore <= 1.0 and sscore <= 1.0:
                break
            if ov and r > max_ratio:
                logger.warning("h3_tp2 seam check window %d: ratio %.2f > %.2f, re-sampling with another seed", state["index"], r, max_ratio)
            elif max_cut > 0 and cr > max_cut:
                logger.warning("h3_tp2 cut check window %d: ratio %.2f > %.2f (a cut inside the window), re-sampling with another seed",
                               state["index"], cr, max_cut)
            elif iscore > 1.0:
                logger.warning("h3_tp2 identity check window %d: more than %.0f %% of the faces match no reference face (score %.2f), re-sampling with another seed",
                               state["index"], 100.0 * max_unknown, iscore)
            else:
                logger.warning("h3_tp2 speech check window %d: the lines came out with a character error rate above %.3f (score %.2f), re-sampling with another seed",
                               state["index"], max_cer, sscore)
        video, audio, _ = best
        if state["prev"] is not None and ov:
            try:
                video = match_window(state["prev"][:, :, -ov:], video, ov, apply=apply, index=state["index"])
                if lock == "first" and state["ref"] is not None:
                    video = lock_to_reference(state["ref"], video, ov, lock_strength, index=state["index"], ramp=lock_ramp, std_strength=lock_std,
                                              ref_maps=state["ref_maps"])
                # The ambient look-lock must not alter latents explicitly
                # supplied by audio_mode=lock_source. Otherwise later windows
                # are re-colored toward window one's audio statistics even
                # though the caller requested the driving soundtrack.
                if lock == "first" and lock_audio and args.get("locked_audio_rows") is not None:
                    logger.info("h3_tp2 audio look-lock bypassed for locked source window %d", state["index"])
                if (lock == "first" and lock_audio and args.get("locked_audio_rows") is None
                        and state["ref_audio"] is not None and isinstance(audio, torch.Tensor) and audio.dim() == 3):
                    ov_a = _overlap_audio(args)
                    if ov_a:
                        audio = lock_audio_to_reference(state["ref_audio"], audio, ov_a, lock_strength, index=state["index"], ramp=lock_ramp)
            except Exception as e:  # noqa: BLE001  (diagnostics never break a render)
                logger.warning("h3_tp2 hand-over diagnostics skipped (%s)", e)
        elif state["ref"] is None and isinstance(video, torch.Tensor) and video.dim() == 5:
            _rv = ref_slice(video, lock_ref, dim=2)
            # class phase of the reference slice: the tail starts at token T - len(tail) of the window
            state["ref"] = class_stats(_rv, offset=video.shape[2] - _rv.shape[2]) if lock_classes else _stats(_rv)
            if lock_robust and lock == "first":
                state["ref_maps"] = class_pos_moments(_rv, offset=video.shape[2] - _rv.shape[2]) if lock_classes else pos_moments(_rv)
            if isinstance(audio, torch.Tensor) and audio.dim() == 3:
                state["ref_audio"] = audio_stats(ref_slice(audio, lock_ref, dim=2))
        if postcheck and ov:
            _log_post_correction_seam(video, ov, state["index"])
        state["prev"] = video
        return video, audio

    def _snapshot_state():
        """Copy match state to CPU after a completed window; experimental, not automatic checkpointing."""
        prev, ref, ref_audio = state["prev"], state["ref"], state["ref_audio"]
        if state["index"] < 1 or not isinstance(prev, torch.Tensor) or prev.dim() != 5 or ref is None:
            raise ValueError("no completed tensor window with a reference to snapshot")
        # the class-aware reference (lock_classes, default since 2026-11-02) is {"all": (m, s), "A": (m, s), "B": (m, s)}
        ref_all = ref["all"] if isinstance(ref, dict) else ref
        tensors = {"match.index": torch.tensor(state["index"], dtype=torch.int64),
                   "match.prev": prev.detach().to("cpu", copy=True),
                   "match.ref_mean": ref_all[0].detach().to("cpu", copy=True),
                   "match.ref_std": ref_all[1].detach().to("cpu", copy=True)}
        if isinstance(ref, dict):
            for cls in ("A", "B"):
                tensors[f"match.ref_{cls}_mean"] = ref[cls][0].detach().to("cpu", copy=True)
                tensors[f"match.ref_{cls}_std"] = ref[cls][1].detach().to("cpu", copy=True)
        if ref_audio is not None:
            tensors["match.audio_mean"] = ref_audio[0].detach().to("cpu", copy=True)
            tensors["match.audio_std"] = ref_audio[1].detach().to("cpu", copy=True)
        maps = state["ref_maps"]
        if maps is not None:
            # the robust lock's per-position moments of the reference tail; class maps only with a class reference
            for cls, pair in (maps.items() if isinstance(maps, dict) else (("all", maps),)):
                tag = "maps" if cls == "all" else f"maps_{cls}"
                tensors[f"match.{tag}_mean"] = pair[0].detach().to("cpu", copy=True)
                tensors[f"match.{tag}_sq"] = pair[1].detach().to("cpu", copy=True)
        return tensors

    def _restore_state(tensors, *, device):  # noqa: ANN001
        """Restore a fresh wrapper's match state; caller must verify request and cumulative AV state."""
        required = {"match.index", "match.prev", "match.ref_mean", "match.ref_std"}
        audio_keys = {"match.audio_mean", "match.audio_std"}
        class_keys = {"match.ref_A_mean", "match.ref_A_std", "match.ref_B_mean", "match.ref_B_std"}
        maps_keys = {"match.maps_mean", "match.maps_sq"}
        maps_class_keys = {"match.maps_A_mean", "match.maps_A_sq", "match.maps_B_mean", "match.maps_B_sq"}
        if state["index"] != 0:
            raise ValueError("match state can only be restored into a fresh wrapper")
        if not isinstance(tensors, dict) or not required <= set(tensors):
            raise ValueError("incomplete or unexpected match-state tensors")
        extra = set(tensors) - required
        groups = (audio_keys, class_keys, maps_keys, maps_class_keys)
        if (extra - set().union(*groups) or any(g & extra and not g <= extra for g in groups)
                or (maps_class_keys <= extra and not (class_keys <= extra and maps_keys <= extra))):
            raise ValueError("incomplete or unexpected match-state tensors")
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise ValueError("match state contains a non-tensor value")
        index = tensors["match.index"]
        prev = tensors["match.prev"]
        mean, std = tensors["match.ref_mean"], tensors["match.ref_std"]
        if (index.ndim != 0 or index.dtype != torch.int64 or index.item() < 1 or prev.dim() != 5
                or not prev.numel() or not prev.is_floating_point() or mean.dtype != torch.float32
                or std.dtype != torch.float32 or mean.dim() != 1 or mean.shape != std.shape
                or mean.numel() != prev.shape[1]):
            raise ValueError("invalid match-state index, video, or reference shape")
        if any(not torch.isfinite(value).all().item() for name, value in tensors.items() if name != "match.index"):
            raise ValueError("match state contains non-finite values")
        if (std < 0).any().item():
            raise ValueError("invalid match reference standard deviation")
        ref = (mean.to(device=device, copy=True), std.to(device=device, copy=True))
        if class_keys.issubset(tensors):
            ref = {"all": ref}
            for cls in ("A", "B"):
                m_c, s_c = tensors[f"match.ref_{cls}_mean"], tensors[f"match.ref_{cls}_std"]
                if (m_c.dtype != torch.float32 or s_c.dtype != torch.float32 or m_c.shape != mean.shape
                        or s_c.shape != std.shape or (s_c < 0).any().item()):
                    raise ValueError("invalid class match reference")
                ref[cls] = (m_c.to(device=device, copy=True), s_c.to(device=device, copy=True))
        ref_audio = None
        if audio_keys.issubset(tensors):
            audio_mean, audio_std = tensors["match.audio_mean"], tensors["match.audio_std"]
            if (audio_mean.dtype != torch.float32 or audio_std.dtype != torch.float32 or audio_mean.dim() != 3
                    or not audio_mean.numel() or audio_mean.shape != audio_std.shape or audio_mean.shape[-1] != 1
                    or (audio_std < 0).any().item()):
                raise ValueError("invalid audio match reference")
            ref_audio = (audio_mean.to(device=device, copy=True), audio_std.to(device=device, copy=True))
        ref_maps = None
        if maps_keys.issubset(tensors):
            def _maps(tag):
                m1, m2 = tensors[f"match.{tag}_mean"], tensors[f"match.{tag}_sq"]
                if (m1.dtype != torch.float32 or m2.dtype != torch.float32 or m1.dim() != 3 or m1.shape != m2.shape
                        or m1.shape[0] != prev.shape[1] or m1.shape[1:] != prev.shape[3:] or (m2 < m1 * m1 - 1e-4).any().item()):
                    raise ValueError("invalid robust look-lock maps")
                return m1.to(device=device, copy=True), m2.to(device=device, copy=True)
            ref_maps = _maps("maps")
            if maps_class_keys.issubset(tensors):
                ref_maps = {"all": ref_maps, "A": _maps("maps_A"), "B": _maps("maps_B")}
        restored = {"prev": prev.to(device=device, copy=True), "index": int(index.item()), "ref": ref, "ref_audio": ref_audio,
                    "ref_maps": ref_maps}
        state.update(restored)

    # Introspection is inert unless explicitly invoked; the serving loop does
    # not save or restore checkpoints, and this state alone is insufficient.
    _diffuse.snapshot_state = _snapshot_state
    _diffuse.restore_state = _restore_state
    return _diffuse


def crossfade_tail(video: torch.Tensor, new_window: torch.Tensor, overlap_v: int) -> torch.Tensor:
    """Replace the cumulative video's last ``overlap_v`` latent frames by a linear crossfade from them to the new
    window's regenerated overlap, then append the new window's free frames.  Upstream keeps the previous window's
    frames up to the hand-over and switches to the new window there, so any difference in fine texture between the
    two renderings of the same frames shows as a jump; the ramp spreads it over the overlap (0.9 s at 22 frames)."""
    prev_tail = video[:, :, -overlap_v:].float()
    regen = new_window[:, :, :overlap_v].float()
    w = torch.linspace(0.0, 1.0, steps=overlap_v + 2, device=video.device, dtype=torch.float32)[1:-1].view(1, 1, -1, 1, 1)
    blended = prev_tail * (1 - w) + regen * w
    # variance-preserving: a mix of two renderings with different fine texture has a lower std mid-ramp (the decoded
    # frame dips darker/flatter for a few frames: the owner's "blink" at the seam, 2026-10-29 20:10). Re-match every
    # blended frame's per-channel mean/std to the interpolation of the two sources' statistics.
    dims = (0, 3, 4)   # per channel and per latent frame
    m_prev, s_prev = prev_tail.mean(dim=dims, keepdim=True), prev_tail.std(dim=dims, keepdim=True)
    m_new, s_new = regen.mean(dim=dims, keepdim=True), regen.std(dim=dims, keepdim=True)
    m_target = m_prev * (1 - w) + m_new * w
    s_target = s_prev * (1 - w) + s_new * w
    m_b, s_b = blended.mean(dim=dims, keepdim=True), blended.std(dim=dims, keepdim=True)
    blended = (blended - m_b) * (s_target / s_b.clamp_min(1e-6)) + m_target
    return torch.cat((video[:, :, :-overlap_v], blended.to(video.dtype), new_window[:, :, overlap_v:]), dim=2)


def ref_slice(x: torch.Tensor, mode: str, *, dim: int) -> torch.Tensor:
    """The frames of the first window that define the look-lock reference.  ``window``: all of them (the first
    behaviour).  ``tail``: the last quarter (at least one frame): a first window that drifts (the painting brightened
    115 -> 135 luma over 345 frames on 2026-10-30) may be pulled toward an earlier look by a whole-window reference.
    The owner reported flicker at 15-22 s; the cause and this alternative remain to be tested."""
    if mode != "tail":
        return x
    n = x.shape[dim]
    return x.narrow(dim, n - max(1, n // 4), max(1, n // 4))


def lock_to_reference(ref_stats: tuple[torch.Tensor, torch.Tensor], new_window: torch.Tensor, overlap_v: int,
                      strength: float, *, index: int, ramp: float = 1.0, std_strength: float | None = None,
                      ref_maps=None) -> torch.Tensor:  # noqa: ANN001
    """Per-channel affine of the new window's FREE frames (after the pinned overlap) so that their mean/std move
    toward the reference window's statistics by ``strength`` (1.0 = equal).  Against the drift that compounds window
    after window (luma +41 and a 10x detail metric over 60 s on 2026-10-29): every window restarts exactly from the
    previous tail and then wanders toward the model's brighter, sharper prior; the lock pulls it back to the look of
    the first window.  Latents only; the pinned overlap frames are left untouched.  With ``ref_maps`` (the reference
    tail's per-position moments, see ``robust_params``) the statistics on both sides come from the positions that did
    not change, so a character who is back in the frame does not move the correction."""
    free = new_window[:, :, overlap_v:].float()
    robust = ref_maps is not None
    if robust:
        _m = ref_maps["all"][0] if isinstance(ref_maps, dict) else ref_maps[0]
        if tuple(_m.shape[1:]) != tuple(free.shape[3:]):
            logger.warning("h3_tp2 look-lock window %d: reference maps %s do not match the window %s, plain statistics used",
                           index, tuple(_m.shape), tuple(free.shape[1:]))
            robust = False
    inliers = []
    n_free = free.shape[2]
    # Ramped: the first free frame gets 1/n_ramp of the correction; the pinned overlap stays untouched. Full correction
    # begins at the n_ramp-th free frame. A whole-window correction previously stepped brightness at the hand-over
    # (-8.5 luma at seam 2, 03:27). ``ramp``: the fraction of the free frames over which the correction reaches full
    # strength (1.0 = the whole window; 0.25 = full after a quarter, then held).
    n_ramp = max(1, int(round(n_free * max(0.05, min(1.0, ramp)))))
    w = torch.clamp(torch.arange(1, n_free + 1, device=free.device, dtype=torch.float32) / n_ramp, max=1.0).view(1, 1, -1, 1, 1)
    k_std = strength if std_strength is None else std_strength   # std_strength 0 = mean only

    def _affine(x, m_ref, s_ref, wt, maps=None):
        if robust and maps is not None:
            m_ref, s_ref, m, sd, frac = robust_params(maps, x)
            inliers.append(frac.mean().item())
        else:
            m, sd = _stats(x)
        gain = 1.0 + k_std * ((s_ref / sd.clamp_min(1e-6)).clamp(0.5, 2.0) - 1.0)
        shift = strength * (m_ref - m)
        gain_t = 1.0 + wt * (gain.view(1, -1, 1, 1, 1) - 1.0)
        return (x - m.view(1, -1, 1, 1, 1)) * gain_t + m.view(1, -1, 1, 1, 1) + wt * shift.view(1, -1, 1, 1, 1), m, sd

    if isinstance(ref_stats, dict):
        # token-class-aware: the block-start tokens (every 5th, the VAE's padded clip-start token) and the other tokens
        # are matched to the reference's statistics of the SAME class, so their relation (what the decoder renders as
        # the brightness of a block's first frame against the rest) stays as in the reference instead of being moved by
        # one affine fitted on the mixture (the 17-frame flicker, 2026-11-02).
        a = block_start_mask(n_free, free.device, offset=overlap_v)
        out = free.clone()
        for cls, mask in (("A", a), ("B", ~a)):
            if int(mask.sum()) == 0:
                continue
            m_r, s_r = ref_stats[cls]
            maps = (ref_maps.get(cls, ref_maps["all"]) if isinstance(ref_maps, dict) else ref_maps) if robust else None
            out[:, :, mask], _, _ = _affine(free[:, :, mask], m_r, s_r, w[:, :, mask], maps)
        m, sd = _stats(free)
        m_ref, s_ref = ref_stats["all"]
        tag = " classes A/B"
    else:
        m_ref, s_ref = ref_stats
        out, m, sd = _affine(free, m_ref, s_ref, w, (ref_maps["all"] if isinstance(ref_maps, dict) else ref_maps) if robust else None)
        tag = ""
    if robust and inliers:
        tag += " robust, inliers %.2f" % (sum(inliers) / len(inliers))
    m2, s2 = _stats(out)
    logger.info("h3_tp2 look-lock window %d: free frames mean %.4f std %.4f -> %.4f / %.4f (reference %.4f / %.4f, strength %.2f, ramp %.2f%s)",
                index, m.mean().item(), sd.mean().item(), m2.mean().item(), s2.mean().item(), m_ref.mean().item(), s_ref.mean().item(), strength, ramp, tag)
    locked = new_window.clone()
    locked[:, :, overlap_v:] = out.to(new_window.dtype)
    return locked


def lock_audio_to_reference(ref_stats: tuple[torch.Tensor, torch.Tensor], new_audio: torch.Tensor, overlap_t: int,
                            strength: float, *, index: int, ramp: float = 1.0) -> torch.Tensor:
    """The audio analogue of ``lock_to_reference``: audio latents [C, D, T]; per-(channel, dim) mean/std over time of
    the free steps (after the pinned overlap) pulled toward the first window's, ramped along time.  The generated
    ambience climbed 23 dB over the last two windows of a 60 s clip (2026-10-30 05:30) while the video lock held."""
    m_ref, s_ref = ref_stats
    free = new_audio[..., overlap_t:].float()
    m, sd = free.mean(dim=-1, keepdim=True), free.std(dim=-1, keepdim=True)
    gain = (s_ref / sd.clamp_min(1e-6)).clamp(0.5, 2.0)
    gain = 1.0 + strength * (gain - 1.0)
    shift = strength * (m_ref - m)
    n_free = free.shape[-1]
    n_ramp = max(1, int(round(n_free * max(0.05, min(1.0, ramp)))))
    w = torch.clamp(torch.arange(1, n_free + 1, device=free.device, dtype=torch.float32) / n_ramp, max=1.0)
    gain_t = 1.0 + w * (gain - 1.0)
    out = (free - m) * gain_t + m + w * shift
    logger.info("h3_tp2 audio look-lock window %d: free steps std %.4f -> %.4f (reference %.4f), mean %.4f -> %.4f", index,
                sd.mean().item(), out.std(dim=-1).mean().item(), s_ref.mean().item(), m.mean().item(), out.mean().item())
    locked = new_audio.clone()
    locked[..., overlap_t:] = out.to(new_audio.dtype)
    return locked


def audio_stats(audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = audio.float()
    return a.mean(dim=-1, keepdim=True), a.std(dim=-1, keepdim=True)


def audio_pin_mask(spec: str | bool | float, rows: int, device=None) -> torch.Tensor | None:
    """Edit-mask values for the `rows` audio latent rows of a window's hand-over prefix (H3_TP2_CONT_EDIT_AUDIO).

    The upstream latent edit takes a soft mask: 0 presents the row as the clean previous tail at the condition
    timestep and writes it back unchanged, 1 samples it free, and a value in between blends both (model input,
    velocity and the restored row are all lerped by the mask).  Returns None when nothing is pinned.

      "1" / True   hard pin of the whole prefix (the default; windows inherit the previous tail)
      "0" / False  no audio pin (soundtrack cut at the hand-over; voice no longer inherits the chain)
      "ramp"       0 at the seam rising linearly to 1 at the end of the prefix: continuity at the cut, free after
      "rows:N"     hard pin of the first N rows only, the rest free
      "0.5"        a constant soft value in (0, 1) on the whole prefix
    """
    text = ("1" if spec is True else "0" if spec is False else str(spec)).strip().lower() or "1"
    if text == "0":
        return None
    if text == "1":
        return torch.zeros(rows, device=device, dtype=torch.float32)
    if text == "ramp":
        return torch.linspace(0.0, 1.0, rows + 1, device=device, dtype=torch.float32)[:-1]
    if text.startswith("rows:"):
        n = max(0, min(rows, int(text[5:])))
        if n == 0:
            return None
        mask = torch.ones(rows, device=device, dtype=torch.float32)
        mask[:n] = 0.0
        return mask
    value = float(text)
    if not 0.0 < value < 1.0:
        raise ValueError(f"H3_TP2_CONT_EDIT_AUDIO={spec!r}: use 0, 1, ramp, rows:N or a value in (0, 1)")
    return torch.full((rows,), value, device=device, dtype=torch.float32)
