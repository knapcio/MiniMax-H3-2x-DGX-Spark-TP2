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
import time

import numpy as np
import torch

logger = logging.getLogger("vllm_omni.h3_tp2.output")

_AUDIO_SR = None
_FPS = None


def _sync() -> None:
    """Phase timers are host wall clock; synchronising at the boundary charges queued GPU work to its own phase."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def install(pipeline_module, api_utils_module) -> None:
    """Patch the pipeline module and the API encoder module (idempotent)."""
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
            video, audio = orig_decode(self, *a, **kw)
            _sync()
            t1 = time.monotonic()
            u8 = to_uint8_frames(video)
            del video
            _sync()
            logger.info("h3_tp2 phase decode %.1f s (+%.1f s uint8) frames=%s", t1 - t0, time.monotonic() - t1, tuple(u8.shape))
            return u8, audio

        def encode_prompt(self, *a, **kw):  # noqa: ANN001
            t0 = time.monotonic()
            try:
                return orig_encode(self, *a, **kw)
            finally:
                _sync()
                logger.info("h3_tp2 phase encode_prompt %.1f s", time.monotonic() - t0)

        def diffuse(self, *a, **kw):  # noqa: ANN001
            t0 = time.monotonic()
            try:
                return orig_diffuse(self, *a, **kw)
            finally:
                _sync()
                logger.info("h3_tp2 phase diffuse %.1f s", time.monotonic() - t0)

        cls.decode = decode
        cls.encode_prompt = encode_prompt
        cls.diffuse = diffuse
        cls._h3_tp2_output_patched = True

    pipeline_module._minimax_h3_post_process = post_process
    pipeline_module.get_minimax_h3_post_process_func = lambda od_config=None: post_process

    if api_utils_module is not None and not getattr(api_utils_module, "_h3_tp2_output_patched", False):
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
