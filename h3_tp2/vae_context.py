"""Left-context temporal VAE decode (opt-in, ``H3_TP2_VAE_LEFT_CONTEXT=<tokens>``).

The released MiniMax H3 decoder decodes a video in temporal chunks of 5 latent tokens (17 frames) plus a 2-token
look-ahead (``token_drop`` 3 -> ``token_overlap`` 2, ``frame_overlap`` 5): block k's frames come from a chunk that
starts at token 5k with no past, and its first five frames are a linear blend between the previous chunk's look-ahead
(decoded with the past) and this fresh start.  Where the fresh start decodes brighter or darker than the contextual
decode, every block begins with a luma step: the periodic 17-frame flicker (-6 YAVG in the 5 s-history painting clip,
regional in the 960x544 renders, 2026-10-30 .. 11-02).

Mechanism: both chunk loops (the checkpoint's ``decode_temporal`` as patched by vLLM-Omni's streaming uint8 fork, and
vLLM-Omni's ``temporal_chunks.decode_temporal_chunks`` used by ``preencode_mp4`` requests) call
``model._adaptive_decode(clip)`` with ``clip = latent[:, :, a:b]``, a view of the (padded) latent.  We wrap
``_adaptive_decode``: when the clip is such a view and ``a > 0``, decode ``latent[:, :, a-ctx:b]`` instead and drop the
first ``ctx * vae_ratio_t`` frames (the decoder is a ViT with a pure reshape unpack: exactly 4 frames per token, the
3 pre-padding frames belong to the chunk's first token, so the remaining frames are positioned exactly as before).
Chunk starts, block boundaries, the 5-frame blend, padding and the emitted frame count are the released ones; a clip
that is not a view (e.g. with isolated head/tail frames concatenated; off in this checkpoint) is decoded unchanged.
Known caveat (review 2026-11-02): the decoder's temporal RoPE is normalised by the clip length, so a 9-token clip is a
geometry the released loop never uses; ``scripts/vae_decode_offline.py --geometry`` measures that effect alone.
``left_context=0`` leaves the model untouched.  The env value is forwarded head -> rank 1 (H3_TP2_ prefix), so both
ranks decode the same clip shapes.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm_omni.h3_tp2.vae_context")


def _temporal_view(clip: torch.Tensor):
    """If ``clip`` is ``base[:, :, a:b]`` of a contiguous rank-5 latent, return (base, a); else None."""
    base = getattr(clip, "_base", None)
    if base is None or clip.dim() != 5 or base.dim() != 5 or not base.is_contiguous():
        return None
    if clip.shape[0] != base.shape[0] or clip.shape[1] != base.shape[1] or clip.shape[3:] != base.shape[3:]:
        return None
    hw = base.shape[3] * base.shape[4]
    off = clip.storage_offset() - base.storage_offset()
    if off < 0 or off % hw:
        return None
    a = off // hw
    if a + clip.shape[2] > base.shape[2] or clip.stride() != base.stride():
        return None
    return base, a


def install_on_model(model, left_context: int):  # noqa: ANN001
    """Wrap ``model._adaptive_decode``; returns a restore callable.  ``left_context <= 0`` installs nothing."""
    left_context = int(left_context)
    if left_context <= 0:
        return lambda: None
    orig = model._adaptive_decode
    ratio_t = int(model.vae_ratio_t)
    stats = {"ctx": 0, "plain": 0}

    def _adaptive_decode_ctx(clip):  # noqa: ANN001
        tv = _temporal_view(clip)
        if tv is None or tv[1] <= 0:
            stats["plain"] += 1
            return orig(clip)
        base, a = tv
        c = min(left_context, a)
        dec = orig(base[:, :, a - c : a + clip.shape[2]])
        stats["ctx"] += 1
        return dec[:, :, c * ratio_t :]

    model._adaptive_decode = _adaptive_decode_ctx
    model._h3_tp2_ctx_stats = stats

    def restore():
        if "_adaptive_decode" in vars(model):
            del model._adaptive_decode
        model.__dict__.pop("_h3_tp2_ctx_stats", None)

    return restore


def install(left_context: int | None = None) -> int:
    """Serving: wrap vLLM-Omni's ``install_temporal_stream_patches`` so every loaded video VAE gets the wrapper."""
    if left_context is None:
        left_context = int(os.environ.get("H3_TP2_VAE_LEFT_CONTEXT", "0").strip() or 0)
    if left_context <= 0:
        return 0
    from vllm_omni.diffusion.models.minimax_h3 import vae as _vae

    if getattr(_vae, "_h3_tp2_ctx_patched", False):
        return left_context
    orig_install = _vae.install_temporal_stream_patches

    def _install(model):  # noqa: ANN001
        out = orig_install(model)
        if callable(getattr(model, "_adaptive_decode", None)) and hasattr(model, "vae_ratio_t"):
            install_on_model(model, left_context)
            logger.warning("h3_tp2: EXPERIMENT VAE temporal decode with %d token(s) of left context per chunk (changes outputs)", left_context)
        return out

    _vae.install_temporal_stream_patches = _install
    _vae._h3_tp2_ctx_patched = True
    return left_context
