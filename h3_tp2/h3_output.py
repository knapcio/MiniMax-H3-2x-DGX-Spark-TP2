"""Output-path patches for MiniMax H3: uint8 frames from the GPU onwards, a digest of the frames, and phase timers.

Why: the stock path keeps the decoded video as FP32 from the VAE to the HTTP encoder.  For a 15 s 1344x768 clip that
is 4.5 GB per copy, and the head holds several copies at once (GPU FP32, pinned host, /dev/shm, the API process copy,
the clamp copy, the float normalisation copy) on top of rank 0: that is what ran the head out of host memory.

What changes (all bit-identical to the stock bytes):
* ``MiniMaxH3Pipeline.decode`` returns the video as uint8 (B, F, H, W, C) computed on the GPU as
  ``round_half_even(clamp(x, 0, 1) * 255.0)``: the same FP32 operations, in the same order, that the stock path runs
  later in numpy (``clamp(0, 1)`` in post-processing, then ``clip(0, 1) * 255.0`` and ``np.rint`` in the encoder).
  Transfers and copies shrink four times.
* ``_minimax_h3_post_process`` passes uint8 video through instead of ``.float().clamp()``-ing it.
* ``_coerce_video_to_uint8_frames`` returns a (F, H, W, 3) uint8 array as is instead of normalising it to float and
  back, and logs a SHA-256 of the frames: the identity check for every lever in this recipe (x264 output can differ
  between encoder builds; the frames cannot).
* ``encode_prompt`` / ``diffuse`` / ``decode`` are timed and logged per request (worker log), so the fixed part of a
  render is visible instead of inferred.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

logger = logging.getLogger("vllm_omni.h3_tp2.output")

_AUDIO_SR = None
_FPS = None


def _sync() -> None:
    """Phase timers are host wall clock; synchronising at the boundary charges queued GPU work to its own phase."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_REPLAY = {"files": None, "next": 0}


def _replay_files() -> list[str] | None:
    """``H3_TP2_REPLAY_LATENTS=<path-or-glob>[,<path-or-glob>...]``: saved diffusion outputs (``H3_TP2_SAVE_LATENTS`` files,
    one per window and seam attempt, in order) that stand in for the denoising of the next request.  Recovery tool: the
    120 s 1344x768 X clip of 2026-11-04 diffused for 7.5 h and then died in the API server's 600 s async-output wait
    during the VAE decode (642 s); replaying its 14 window files through the same hand-over, stitch, decode and mux
    gave the clip in 15 minutes.  Every rank replays its own copies (the save hook writes on both hosts), the geometry
    of every file is checked against the diffuse call and a mismatch (a probe, another request) fails that request
    instead of consuming a file.  Set it for one boot only; the queue library strips it at the next restart.
    ``H3_TP2_REPLAY_PARTIAL=1`` lets the request continue with real denoising once the files are used up (repair of a
    late window: replay the good windows, re-sample the rest with another seed)."""
    spec = os.environ.get("H3_TP2_REPLAY_LATENTS", "").strip()
    if not spec:
        return None
    if _REPLAY["files"] is None:
        import glob

        files: list[str] = []
        for pat in spec.split(","):
            pat = os.path.expanduser(pat.strip())
            if not pat:
                continue
            hits = sorted(glob.glob(pat))
            files.extend(hits if hits else [pat])
        _REPLAY["files"] = files
        logger.warning("h3_tp2 REPLAY: %d saved latents files stand in for the diffusion, in order: %s", len(files),
                       ", ".join(os.path.basename(f) for f in files))
    return _REPLAY["files"]


def _replay_take(kw: dict):  # noqa: ANN001
    """Return the next saved (video, audio) latents for this diffuse call, or None when replay is off."""
    files = _replay_files()
    if not files:
        return None
    i = _REPLAY["next"]
    if i >= len(files):
        # H3_TP2_REPLAY_PARTIAL=1: the saved files cover only the first windows of the request; the rest is denoised
        # (and saved) as usual. Repairs a long continuation whose late window went wrong (a cut-away, babble) at the
        # cost of the windows after it: replay windows 1..k from the original request, give the request another seed
        # so that windows k+1.. sample differently, keep everything else identical. Added 2026-11-05 after the 118 s
        # Polish scene cut to a close-up in window 9 of 11.
        if os.environ.get("H3_TP2_REPLAY_PARTIAL", "").strip() == "1":
            if i == len(files):
                logger.warning("h3_tp2 REPLAY: the %d saved latents files are used up; this and the later windows are "
                               "denoised (H3_TP2_REPLAY_PARTIAL=1)", len(files))
            _REPLAY["next"] = i + 1
            return None
        raise RuntimeError(f"h3_tp2 REPLAY: diffuse call {i + 1} but only {len(files)} saved latents files were given")
    obj = torch.load(files[i], map_location="cpu")
    if not (isinstance(obj, (tuple, list)) and len(obj) == 2 and all(torch.is_tensor(t) for t in obj)):
        raise RuntimeError(f"h3_tp2 REPLAY: {files[i]} is not a saved (video, audio) latents pair")
    video, audio = obj
    want_v = (kw.get("latent_t"), kw.get("latent_h"), kw.get("latent_w"))
    if all(x is not None for x in want_v) and tuple(video.shape[-3:]) != tuple(int(x) for x in want_v):
        raise RuntimeError(f"h3_tp2 REPLAY: {os.path.basename(files[i])} video latents {tuple(video.shape)} do not match the "
                           f"request's (t, h, w) {want_v}; this request is not the one that saved them")
    if kw.get("audio_t") is not None and int(audio.shape[-1]) != int(kw["audio_t"]):
        raise RuntimeError(f"h3_tp2 REPLAY: {os.path.basename(files[i])} audio latents {tuple(audio.shape)} do not match the "
                           f"request's audio_t {kw['audio_t']}")
    _REPLAY["next"] = i + 1
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    logger.warning("h3_tp2 REPLAY %d/%d: %s -> video %s audio %s (seed %s not sampled)", i + 1, len(files),
                   os.path.basename(files[i]), tuple(video.shape), tuple(audio.shape), kw.get("seed"))
    return video.to(dev), audio.to(dev)


def to_uint8_frames(video: torch.Tensor) -> torch.Tensor:
    """(B, C, F, H, W) float -> (B, F, H, W, C) uint8, same arithmetic as the stock numpy path."""
    out = []
    for sample in video:  # one sample at a time: the FP32 temporaries stay at frame-batch size
        chunks = []
        for f0 in range(0, sample.shape[1], 16):
            x = torch.round(sample[:, f0 : f0 + 16].float().clamp(0.0, 1.0) * 255.0)  # half to even, like np.rint
            chunks.append(x.to(torch.uint8).permute(1, 2, 3, 0).contiguous())
        out.append(torch.cat(chunks, dim=0))
    return torch.stack(out, dim=0)


def post_process(output, output_type: str = "np"):
    """Module-level (pickled by name across the result queue), handles uint8 and the stock float video."""
    if not isinstance(output, tuple) or len(output) != 2:
        return output
    video, audio = output
    if output_type == "latent":
        return output
    if output_type == "np":
        if video.dtype == torch.uint8:
            video = video.detach().cpu().numpy()  # already (B, F, H, W, C)
        else:
            video = video.detach().float().cpu().permute(0, 2, 3, 4, 1).clamp(0, 1).numpy()
        audio = audio.detach().float().cpu().numpy()
        video = [sample for sample in video]
    return {
        "video": video,
        "audio": audio,
        "audio_sample_rate": _AUDIO_SR,
        "fps": _FPS,
    }


def frames_digest(frames_u8: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frames_u8).tobytes()).hexdigest()


def tensors_digest(obj) -> str:  # noqa: ANN001
    """sha256 over the raw bytes of every tensor reachable from ``obj`` (tuples, lists, dicts), in traversal order.

    Logged for the diffusion output before the VAE decode, so a changed clip can be attributed to the diffusion
    (different latents) or to the decoder (same latents, different frames).  bf16/fp8 tensors are hashed as bytes.
    """
    h = hashlib.sha256()
    n = 0

    def walk(x, depth=0):  # noqa: ANN001
        nonlocal n
        if depth > 3 or isinstance(x, torch.nn.Module):
            return
        if isinstance(x, torch.Tensor):
            t = x.detach()
            t = t.contiguous().view(torch.uint8) if t.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2) else t.contiguous()
            h.update(t.cpu().numpy().tobytes())
            n += 1
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                walk(x[k], depth + 1)
        elif isinstance(x, (tuple, list)):
            for y in x:
                walk(y, depth + 1)
        elif hasattr(x, "__dict__") and not isinstance(x, type):
            for k in sorted(vars(x)):
                walk(vars(x)[k], depth + 1)

    walk(obj)
    return f"{h.hexdigest()} tensors={n}"


def _blend_trace_context(pipeline):  # noqa: ANN001
    if os.environ.get("H3_TP2_VAE_BLEND_TRACE", "0").strip() != "1":
        return nullcontext()
    try:
        from h3_tp2.vae_trace import trace_temporal_blends

        return trace_temporal_blends(pipeline.video_vae.model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("h3_tp2 VAE temporal blend tracing unavailable (%s)", exc)
        return nullcontext()


def install(pipeline_module, api_utils_module, convert_u8: bool = True) -> None:
    """Patch the pipeline module and the API encoder module (idempotent).

    ``convert_u8=False`` keeps only the phase timers and the frame digest: vLLM-Omni >= 0.30 quantises the video to
    uint8 on the GPU itself (``_prepare_minimax_h3_video_output``, the same clamp/scale/round arithmetic) and its
    post-processing expects that layout, so the conversion and the post-process replacement must stay out of the way.
    """
    global _AUDIO_SR, _FPS
    _AUDIO_SR = pipeline_module.MINIMAX_H3_AUDIO_SAMPLE_RATE
    _FPS = pipeline_module.MINIMAX_H3_FPS
    cls = pipeline_module.MiniMaxH3Pipeline

    if not getattr(cls, "_h3_tp2_output_patched", False):
        orig_decode = cls.decode
        orig_encode = cls.encode_prompt
        orig_diffuse = cls.diffuse

        def decode(self, *a, **kw):  # noqa: ANN001
            t0 = time.monotonic()
            with _blend_trace_context(self):
                out = orig_decode(self, *a, **kw)
            _sync()
            t1 = time.monotonic()
            if not convert_u8:
                logger.info("h3_tp2 phase decode %.1f s", t1 - t0)
                return out
            video, audio = out
            u8 = to_uint8_frames(video)
            del video
            _sync()
            logger.info("h3_tp2 phase decode %.1f s (+%.1f s uint8) frames=%s", t1 - t0, time.monotonic() - t1, tuple(u8.shape))
            return u8, audio

        if hasattr(cls, "decode_to_mp4"):  # v0.30 chunked decode + mux on the worker (preencode_mp4)
            orig_decode_mp4 = cls.decode_to_mp4

            def decode_to_mp4(self, *a, **kw):  # noqa: ANN001
                t0 = time.monotonic()
                try:
                    with _blend_trace_context(self):
                        return orig_decode_mp4(self, *a, **kw)
                finally:
                    _sync()
                    logger.info("h3_tp2 phase decode_to_mp4 %.1f s", time.monotonic() - t0)

            cls.decode_to_mp4 = decode_to_mp4

        def encode_prompt(self, *a, **kw):  # noqa: ANN001
            t0 = time.monotonic()
            out = None
            try:
                out = orig_encode(self, *a, **kw)
                return out
            finally:
                _sync()
                logger.info("h3_tp2 phase encode_prompt %.1f s", time.monotonic() - t0)
                _log_digest("prompt", out)

        def _save_latents(obj):  # noqa: ANN001
            """H3_TP2_SAVE_LATENTS=<dir>: keep the diffusion output (video + audio latents) of every request as a
            .pt file, so the decoder can be run on the same latents offline (the 17-frame luma flicker, 2026-11-02).
            Diagnostics only; never breaks a render."""
            d = os.environ.get("H3_TP2_SAVE_LATENTS", "").strip()
            if not d:
                return
            try:
                import socket

                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"latents-{socket.gethostname()}-{time.strftime('%Y%m%d-%H%M%S')}.pt")
                torch.save(obj, path)
                logger.info("h3_tp2 latents saved to %s", path)
            except Exception as e:  # noqa: BLE001
                logger.info("h3_tp2 latents not saved (%s)", e)

        def _log_digest(what, obj):  # noqa: ANN001
            try:
                logger.info("h3_tp2 %s sha256 %s", what, tensors_digest(obj))
            except Exception as e:  # noqa: BLE001  (a digest must never break a render)
                logger.info("h3_tp2 %s sha256 unavailable (%s)", what, e)

        if hasattr(cls, "_encode_local_media"):  # v0.30: reference image / video / audio -> conditioning rows
            orig_media = cls._encode_local_media

            def _encode_local_media(self, *a, **kw):  # noqa: ANN001
                out = orig_media(self, *a, **kw)
                _sync()
                _log_digest("media", out)
                return out

            cls._encode_local_media = _encode_local_media

        def diffuse(self, *a, **kw):  # noqa: ANN001
            t0 = time.monotonic()
            out = None
            replayed = _replay_take(kw)  # H3_TP2_REPLAY_LATENTS: saved latents instead of the denoising (recovery)
            _step0["armed"] = replayed is None
            try:
                out = replayed if replayed is not None else orig_diffuse(self, *a, **kw)
                return out
            finally:
                _sync()
                logger.info("h3_tp2 phase diffuse %.1f s%s", time.monotonic() - t0, " (replayed)" if replayed is not None else "")
                _log_digest("latents", out)
                if replayed is None:
                    _save_latents(out)

        cls.decode = decode
        cls.encode_prompt = encode_prompt
        cls.diffuse = diffuse
        cls._h3_tp2_output_patched = True

        # the DiT's first forward of each diffusion (its kwargs: noisy latents, embeddings, positions, timesteps) hashed
        # once per request: equal here but different latents means the denoising loop itself varies
        _step0 = {"armed": False}
        try:
            from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as _tr

            _dit = _tr.MiniMaxH3DiTModel
            if not getattr(_dit, "_h3_tp2_step0_patched", False):
                _orig_dit_fwd = _dit.forward

                def _dit_fwd(self, **kw):  # noqa: ANN001
                    if _step0["armed"]:
                        _step0["armed"] = False
                        _sync()
                        _log_digest("step0", kw)
                    return _orig_dit_fwd(self, **kw)

                _dit.forward = _dit_fwd
                _dit._h3_tp2_step0_patched = True
        except Exception as e:  # noqa: BLE001
            logger.info("h3_tp2 step0 digest not installed (%s)", e)

    if convert_u8:
        pipeline_module._minimax_h3_post_process = post_process
        pipeline_module.get_minimax_h3_post_process_func = lambda od_config=None: post_process

    if api_utils_module is not None and not getattr(api_utils_module, "_h3_tp2_output_patched", False):
        if hasattr(api_utils_module, "_prepare_video_frames"):  # v0.30: the shared entry of both encoder paths
            orig_prepare = api_utils_module._prepare_video_frames

            def prepare(video):  # noqa: ANN001
                frames, shape, dtype = orig_prepare(video)
                try:
                    if dtype == np.uint8:
                        h = hashlib.sha256()
                        for f in frames:
                            h.update(np.ascontiguousarray(f).tobytes())
                        logger.info("h3_tp2 frames sha256 %s shape=%s", h.hexdigest(), (len(frames),) + tuple(shape))
                except Exception:
                    logger.exception("h3_tp2 frame digest failed")
                return frames, shape, dtype

            api_utils_module._prepare_video_frames = prepare
            api_utils_module._h3_tp2_output_patched = True
            return
        orig_coerce = api_utils_module._coerce_video_to_uint8_frames

        def coerce(video):  # noqa: ANN001
            if isinstance(video, np.ndarray) and video.dtype == np.uint8 and video.ndim == 4 and video.shape[-1] == 3:
                frames = np.ascontiguousarray(video)
            else:
                frames = orig_coerce(video)
            logger.info("h3_tp2 frames sha256 %s shape=%s", frames_digest(frames), tuple(frames.shape))
            return frames

        api_utils_module._coerce_video_to_uint8_frames = coerce
        api_utils_module._h3_tp2_output_patched = True
