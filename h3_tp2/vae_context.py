"""Left-context temporal VAE decode (opt-in, ``H3_TP2_VAE_LEFT_CONTEXT=<tokens>``).

The released MiniMax H3 decoder decodes a video in temporal chunks of 5 latent tokens (17 frames) plus a 2-token
look-ahead (``token_drop`` 3 -> ``token_overlap`` 2, ``frame_overlap`` 5): block k's frames come from a chunk that
starts at token 5k with no past, and its first five frames are a linear blend between the previous chunk's look-ahead
(decoded WITH the past) and this fresh start. Where the fresh start decodes brighter or darker than the contextual
decode, every block begins with a luma step: the periodic 17-frame flicker (-6 YAVG in the 5 s-history painting clip,
regional in the 960x544 renders, 2026-10-30 .. 11-02).

This loop is upstream vLLM-Omni's ``temporal_chunks.decode_temporal_chunks`` (v0.30.0) with one change: chunk k > 0 is
decoded from ``left_context`` extra tokens before its start, whose frames are dropped, so the chunk's own frames are
decoded with the same past the look-ahead had.  Nothing else moves: chunk starts, block boundaries, the 5-frame blend,
padding, isolated frames and the emitted frame count are the released ones.  Cost: (5 + 2 + ctx) / 7 of the decoder
compute.  ``left_context=0`` is bit-for-bit the upstream loop.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import torch
import torch.nn as nn

logger = logging.getLogger("vllm_omni.h3_tp2.vae_context")


def decode_temporal_chunks_ctx(
    model: nn.Module,
    latent: torch.Tensor,
    callback: Callable[[torch.Tensor], None] | None,
    *,
    left_context: int = 0,
) -> torch.Tensor:
    if latent.ndim != 5 or not bool(getattr(model, "use_3d_conv", False)):
        raise ValueError("MiniMax-H3 temporal chunk decode requires a rank-5 3D latent")
    left_context = max(0, int(left_context))

    token_drop = int(model.token_drop)
    chunk_size = int(model.tokens_chunk_size)
    overlap_tokens = int(model.token_overlap)
    ratio_t = int(model.vae_ratio_t)
    pre_padding = int(model.frame_pre_padding)

    isolated_first = bool(model.isolated_first_frame and pre_padding == 0)
    isolated_last = bool(model.isolated_last_frame)
    z_head = latent[:, :, :1] if isolated_first else None
    z_tail = latent[:, :, -1:] if isolated_last else None
    start = int(isolated_first)
    stop = int(latent.shape[2]) - int(isolated_last)
    latent = latent[:, :, start:stop]

    pseudo_tokens = int(latent.shape[2]) + token_drop
    pad_tokens = (-pseudo_tokens) % chunk_size
    if pad_tokens:
        latent = torch.cat((latent, latent[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)), dim=2)
    num_chunks = (pseudo_tokens + pad_tokens) // chunk_size - int(token_drop > 0)
    if num_chunks <= 0:
        raise ValueError("MiniMax-H3 temporal chunk plan is empty")

    total_frames, pad_frames, output_frames = model._decode_temporal_output_frame_plan(
        latent, z_head, z_tail, num_chunks, pad_tokens
    )
    output_frames = int(output_frames)
    if output_frames <= 0:
        raise ValueError("MiniMax-H3 temporal chunk plan has no output frames")

    collected: list[torch.Tensor] = []
    overlap: torch.Tensor | None = None
    written = 0

    def emit(part: torch.Tensor) -> None:
        nonlocal written
        frames = int(part.shape[2])
        if frames <= 0 or written >= output_frames:
            return
        frames = min(frames, output_frames - written)
        part = part[:, :, :frames]
        if callback is None:
            collected.append(part)
        else:
            callback(part)
        written += frames

    chunk_frames = chunk_size * ratio_t
    split_count = int(token_drop > 0) + 1
    for index in range(num_chunks):
        begin = index * chunk_size
        end = begin + chunk_size + overlap_tokens
        # --- the one change: real past for every chunk after the first -------------------------------------------
        ctx = min(left_context, begin) if index > 0 else 0
        clip = latent[:, :, begin - ctx : end]
        if index == 0 and z_head is not None:
            clip = torch.cat((z_head, clip), dim=2)
        if index == num_chunks - 1 and z_tail is not None:
            clip = torch.cat((clip, z_tail), dim=2)

        decoded = model._adaptive_decode(clip)
        if ctx:
            decoded = decoded[:, :, ctx * ratio_t :]   # drop the context tokens' frames; the rest is positioned as before
        # ----------------------------------------------------------------------------------------------------------

        decoded_tail = None
        if index == 0 and z_head is not None:
            emit(decoded[:, :, ratio_t - 1 : ratio_t])
            decoded = decoded[:, :, ratio_t:]
        if index == num_chunks - 1 and z_tail is not None:
            decoded_tail = decoded[:, :, -1:]
            decoded = decoded[:, :, :-ratio_t]

        for split in range(split_count):
            b = split * chunk_frames
            e = min(b + chunk_frames, int(decoded.shape[2]))
            part = decoded[:, :, b:e]
            part = part[:, :, pre_padding:]
            if split == 0:
                if overlap is not None:
                    part = model.blend(overlap, part, int(model.frame_overlap), dim=-3)
                    overlap = None
                emit(part)
            else:
                overlap = part.contiguous()

        if index == num_chunks - 1:
            if overlap is not None:
                emit(overlap)
                overlap = None
            if decoded_tail is not None:
                emit(decoded_tail)

    if written != output_frames:
        raise RuntimeError(f"MiniMax-H3 temporal decode emitted {written}/{output_frames} frames")
    if callback is not None:
        return latent.new_empty((0,))
    return torch.cat(collected, dim=2) if collected else latent.new_empty((0,))


def install(left_context: int | None = None) -> int:
    """Rebind upstream's chunk loop to the left-context version.  Returns the context in use (0 = untouched)."""
    if left_context is None:
        left_context = int(os.environ.get("H3_TP2_VAE_LEFT_CONTEXT", "0").strip() or 0)
    if left_context <= 0:
        return 0
    from vllm_omni.diffusion.models.minimax_h3 import chunked_decode as _cd

    def _loop(model, latent, callback):  # noqa: ANN001
        return decode_temporal_chunks_ctx(model, latent, callback, left_context=left_context)

    _cd.decode_temporal_chunks = _loop
    logger.warning("h3_tp2: EXPERIMENT VAE temporal decode with %d token(s) of left context per chunk (changes outputs)", left_context)
    return left_context
