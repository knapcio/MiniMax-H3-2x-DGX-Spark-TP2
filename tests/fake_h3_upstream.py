"""Fake vLLM-Omni MiniMax H3 modules for ``tests/test_step_cache.py`` (CPU, no vLLM, no GPU).

The stock reference is upstream code: the sections marked VERBATIM are copied unchanged from vLLM-Omni v0.30.0
(a8576ccb7, Apache-2.0, ``vllm_omni/diffusion/models/minimax_h3/``): the scheduler helpers
(``scheduling_minimax_h3_euler_ancestral.py:10-34, 49-103``), the latent edit (``latent_mask.py:69-175``), the
denoise loop (``denoise_loop.py:33-129, 166-316, 319-458``; only the video-span layout block of the branch's
``__init__``, ``:130-164``, is left out) and the RoPE (``minimax_h3_transformer.py:300-332``).
``test_step_cache`` checks that the copied functions carry the
fingerprints ``h3_tp2.step_cache.EXPECTED`` was reviewed against.  Imports of the upstream modules are replaced by
the stubs below; the DiT, the token refiner and the pipeline are small fakes with upstream's call structure.

Load a fresh, independent copy with ``load()`` (one stock, one patched per test).
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# ---- stubs for denoise_loop's imports ------------------------------------------------------------------------------
CANCEL_CALLS: list[bool] = []
PROGRESS: list[tuple] = []


def check_request_cancellation(*, synchronize: bool = False) -> None:
    CANCEL_CALLS.append(synchronize)


def set_forward_context_denoise_step_idx(step):  # noqa: ANN001
    PROGRESS.append(("step", step))


def set_forward_context_denoise_timestep(t):  # noqa: ANN001
    PROGRESS.append(("t", t))


def set_forward_context_denoise_total_steps(n):  # noqa: ANN001
    PROGRESS.append(("n", n))


class _Platform:
    @staticmethod
    def is_npu() -> bool:
        return False


current_omni_platform = _Platform()


# ---- VERBATIM scheduling_minimax_h3_euler_ancestral.py:10-34 -------------------------------------------------------
def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_unit_timestep(timestep: torch.Tensor, name: str) -> None:
    if not isinstance(timestep, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(timestep):
        raise ValueError(f"{name} must be a floating point tensor")
    _require_finite_tensor(timestep, name)
    out_of_range = (timestep < 0) | (timestep > 1)
    if bool(out_of_range.any().item()):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"{name} must be finite")
    if sigma < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return sigma




# ---- VERBATIM scheduling_minimax_h3_euler_ancestral.py:49-103 ---------------------------------------------------
def minimax_h3_rf_v_to_x0(
    xt: torch.Tensor,
    v: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if xt.shape != v.shape:
        raise ValueError(f"xt and v shapes must match, got {xt.shape} vs {v.shape}")
    if not torch.is_floating_point(xt):
        raise ValueError("xt must be a floating point tensor")
    if not torch.is_floating_point(v):
        raise ValueError("v must be a floating point tensor")
    _require_finite_tensor(xt, "xt")
    _require_finite_tensor(v, "v")
    _validate_unit_timestep(timestep, "timestep")
    cond_t = timestep.to(device=xt.device, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    sigma_t = 1 - cond_t
    x0 = xt + sigma_t * v
    _require_finite_tensor(x0, "x0")
    return x0


def minimax_h3_euler_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    if state.shape != denoised.shape:
        raise ValueError(f"state and denoised shapes must match, got {state.shape} vs {denoised.shape}")
    if not torch.is_floating_point(state):
        raise ValueError("state must be a floating point tensor")
    if not torch.is_floating_point(denoised):
        raise ValueError("denoised must be a floating point tensor")
    _require_finite_tensor(state, "state")
    _require_finite_tensor(denoised, "denoised")
    sigma_curr = _validate_sigma(sigma_curr, "sigma_curr")
    sigma_next = _validate_sigma(sigma_next, "sigma_next")
    if sigma_curr == 0.0:
        if sigma_next != 0.0:
            raise ValueError("sigma_next must be 0 when sigma_curr is 0")
        return state
    compute_dtype = torch.float32
    if state.dtype not in (torch.float16, torch.bfloat16):
        compute_dtype = state.dtype
    sigma_curr_t = state.new_tensor(sigma_curr, dtype=compute_dtype)
    sigma_next_t = state.new_tensor(sigma_next, dtype=compute_dtype)
    sigma_ratio = sigma_next_t / sigma_curr_t
    out = sigma_ratio * state.to(dtype=compute_dtype) + (1.0 - sigma_ratio) * denoised.to(dtype=compute_dtype)
    out = out.to(dtype=state.dtype)
    _require_finite_tensor(out, "euler_eta0_step output")
    return out


# ---- VERBATIM latent_mask.py:69-175 ------------------------------------------------------------------------------
@dataclass(frozen=True)
class MiniMaxH3LatentEdit:
    """Clean source, model anchor, and masks for one packed latent stream."""

    clean_rows: torch.Tensor
    anchor_rows: torch.Tensor
    mask_rows: torch.Tensor
    restore_mask_rows: torch.Tensor

    @classmethod
    def from_rows(
        cls,
        clean_rows: torch.Tensor,
        anchor_rows: torch.Tensor,
        mask_rows: torch.Tensor,
        restore_mask_rows: torch.Tensor,
    ) -> MiniMaxH3LatentEdit:
        """Create an edit from already-validated encoder outputs.

        Value validation and all-generate elision belong at the encoder input
        boundary. This constructor only normalizes devices/dtypes and checks
        the cheap row-shape invariants needed by the sampler math.
        """
        if clean_rows.ndim != 2 or anchor_rows.shape != clean_rows.shape:
            raise ValueError("clean_rows and anchor_rows must have matching [rows, width] shape")
        raw_mask = torch.as_tensor(mask_rows, dtype=torch.float32, device=clean_rows.device)
        if raw_mask.ndim != 1 or raw_mask.numel() != clean_rows.shape[0]:
            raise ValueError(f"mask_rows must have shape ({clean_rows.shape[0]},)")
        restore = torch.as_tensor(restore_mask_rows, dtype=torch.float32, device=clean_rows.device)
        valid_restore = restore.ndim == 1 and restore.shape == raw_mask.shape
        valid_restore |= restore.ndim == 2 and tuple(restore.shape) in {
            (clean_rows.shape[0], 1),
            tuple(clean_rows.shape),
        }
        if not valid_restore:
            raise ValueError("restore_mask_rows must have one scalar or one feature row per clean row")
        return cls(clean_rows, anchor_rows, raw_mask, restore)

    def to(self, *, device: torch.device, dtype: torch.dtype = torch.float32) -> MiniMaxH3LatentEdit:
        return MiniMaxH3LatentEdit(
            self.clean_rows.to(device=device, dtype=dtype),
            self.anchor_rows.to(device=device, dtype=dtype),
            self.mask_rows.to(device=device),
            self.restore_mask_rows.to(device=device),
        )

    def model_rows(self, state_rows: torch.Tensor) -> torch.Tensor:
        return torch.lerp(self.anchor_rows, state_rows, self.mask_rows.to(state_rows.dtype).unsqueeze(-1))

    def target_timesteps(
        self,
        timestep: float,
        condition_timestep: float,
        *,
        sigma: float | None = None,
    ) -> torch.Tensor:
        sigma = 1.0 - timestep if sigma is None else float(sigma)
        return torch.clamp(1.0 - self.mask_rows * sigma, max=condition_timestep)

    def prepare(
        self,
        state_rows: torch.Tensor,
        timestep: float,
        condition_timestep: float,
        *,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the model rows and per-row timesteps for one denoise step."""
        return (
            self.model_rows(state_rows),
            self.target_timesteps(timestep, condition_timestep, sigma=sigma),
        )

    def x0(self, model_rows: torch.Tensor, velocity: torch.Tensor, timestep: float) -> torch.Tensor:
        weight = self.mask_rows.to(model_rows.dtype).unsqueeze(-1)
        predicted = minimax_h3_rf_v_to_x0(
            model_rows,
            velocity.to(model_rows.dtype) * weight,
            torch.tensor(float(timestep), dtype=model_rows.dtype, device=model_rows.device),
        )
        restore = self.restore_mask_rows.to(model_rows.dtype)
        if restore.ndim == 1:
            restore = restore.unsqueeze(-1)
        return torch.lerp(self.clean_rows, predicted, restore)


def minimax_h3_prepare_edit_rows(
    rows: torch.Tensor,
    update_mask: torch.Tensor,
    edit: MiniMaxH3LatentEdit | None,
    timestep: float,
    condition_timestep: float,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build one model input view without mutating the persistent sampler rows."""
    if edit is None:
        return rows, None
    model_rows = rows.clone()
    model_rows[update_mask], target_timesteps = edit.prepare(
        rows[update_mask],
        timestep,
        condition_timestep,
        sigma=sigma,
    )
    return model_rows, target_timesteps


# ---- VERBATIM denoise_loop.py:33-129 (the branch's __init__ ends before the video-span block :130-164) ----------
MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
# ref2va audio reference anchor timestep
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
# Packed row widths: video rows are [1,2,2]-patchified 24-channel latents
# (24 * 1 * 2 * 2 = 96); audio rows carry the 32-dim audio latent.
MINIMAX_H3_VIDEO_ROW_WIDTH = 96
MINIMAX_H3_AUDIO_ROW_WIDTH = 32


def minimax_h3_publish_denoise_progress(
    step: int | None, sigma_video: float | None, total_steps: int | None = None
) -> None:
    """Publish denoise progress for step-gated attention features.

    Both execution modes must publish the same trio: the step index drives the
    dense warmup of RAINFUSION_ATTN, the normalized descending timestep
    drives the TRTLLM_ATTN skip gate (which stays dense while it is unset),
    and the total step count enables the ``end_step`` tail fallback.
    """
    set_forward_context_denoise_step_idx(step)
    set_forward_context_denoise_timestep(sigma_video)
    set_forward_context_denoise_total_steps(total_steps)


class MiniMaxH3DenoiseBranch:
    """Static per-branch state: packed layout + fixed forward kwargs.

    `packed` is a minimax_h3_packed_sequence(...) result (or equivalent layout
    dict); `text_embeddings` is the branch's [text_len, 5120] hidden states;
    `token_tags` must already carry any fl2va vision-span overrides.
    """

    def __init__(
        self,
        *,
        packed: dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        device: torch.device,
    ) -> None:
        seq_len = int(packed["seq_len"])
        self.seq_len = seq_len
        self.img_pos = packed["img_pos"].view(-1).to(torch.long)
        self.audio_pos = packed["audio_pos"].view(-1).to(torch.long)
        self.update_mask = packed["update_mask"].view(-1).to(torch.bool)
        # ref2va: audio_pos may include reference-audio anchor rows
        # (audio_update_mask False); absent means all rows are targets.
        if "audio_update_mask" in packed:
            self.audio_update_mask = packed["audio_update_mask"].view(-1).to(torch.bool)
        else:
            self.audio_update_mask = torch.ones(self.audio_pos.shape[0], dtype=torch.bool)
        text_len = int(packed["text_pos"].view(-1).shape[0])
        if list(text_embeddings.shape)[0] != text_len:
            raise ValueError(f"text_embeddings rows {list(text_embeddings.shape)} != packed text_len {text_len}")
        if int(token_tags.view(-1).shape[0]) != seq_len:
            raise ValueError(f"token_tags length {int(token_tags.view(-1).shape[0])} != seq_len {seq_len}")
        cu = packed["cu_seqlens"].to(torch.int32)
        self.device = device
        self.locked_audio_rows: torch.Tensor | None = None
        # ``used_len`` is the real (non-padding) document length; step-mode
        # batching concatenates layouts and needs both bounds as Python ints so
        # it can rebuild ``cu_seqlens`` without a device sync per step.
        self.used_len = int(cu[1])
        self.text_len = text_len
        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.update_mask_dev = self.update_mask.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        self.x_base = torch.zeros(1, seq_len, MINIMAX_H3_VIDEO_ROW_WIDTH, dtype=torch.float32, device=device)
        self.audio_x_base = torch.zeros(1, seq_len, MINIMAX_H3_AUDIO_ROW_WIDTH, dtype=torch.float32, device=device)
        self.text_pos_dev = packed["text_pos"].view(-1).to(torch.long).to(device)
        self.token_tags_dev = token_tags.view(-1).to(torch.long).to(device)
        self.img_position_ids_dev = packed["img_position_ids"].to(device)
        self.text_embeddings_dev = text_embeddings.to(device)
        self.static_kwargs: dict[str, Any] = {
            "img_position_ids": self.img_position_ids_dev[None],
            "update_mask": self.update_mask_dev,
            "token_tags": self.token_tags_dev,
            "skip_mask_out_condition": False,
            "prompt_embeds": self.text_embeddings_dev,
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": self.text_pos_dev},
            "img_pos_for_infer_output_info": {"position_ids": self.img_pos_dev},
            "packed_seq_params": {
                "cu_seqlens_q": cu.to(device),
                "max_seqlen_q": self.used_len,
                # One request: valid rows are a prefix, so attention may use a
                # KV prefix length or a 1-D pad mask. See batched_packing for the
                # co-batched layout, where neither can describe the valid rows.
                "num_requests": 1,
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device),
                "max_seqlen_q": text_len,
            },
        }

    # ---- VERBATIM denoise_loop.py:166-316 ------------------------------------------------------------------------
    def prepare_rope_table(self, model: Any) -> None:
        """Materialize the branch-local DiT RoPE table once per denoise run.

        ``img_position_ids`` is immutable for this branch while latents and
        timesteps change every scheduler step. Keeping the table in
        ``static_kwargs`` makes every model call reuse the exact same BF16
        tensor without extending its lifetime beyond this request branch.
        """
        prepare = getattr(model, "prepare_rope_table", None)
        if not callable(prepare):
            return
        self.static_kwargs["rope_table"] = prepare(
            self.static_kwargs["img_position_ids"],
            seq_len=self.seq_len,
        )

    def forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
        video_target_timesteps: torch.Tensor | None = None,
        audio_target_timesteps: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        x = self.x_base.clone()
        x[0].index_copy_(0, self.img_pos_dev, video_rows)
        audio_x = self.audio_x_base.clone()
        audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
        timesteps = torch.empty(self.seq_len, dtype=torch.float32, device=self.device)
        self.fill_timesteps(
            timesteps,
            t_video=t_video,
            t_audio=t_audio,
            imgvid_cond_timestep=imgvid_cond_timestep,
            audio_ref_cond_timestep=audio_ref_cond_timestep,
            video_target_timesteps=video_target_timesteps,
            audio_target_timesteps=audio_target_timesteps,
        )
        unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
        }

    def fill_timesteps(
        self,
        timesteps: torch.Tensor,
        *,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
        video_target_timesteps: torch.Tensor | None = None,
        audio_target_timesteps: torch.Tensor | None = None,
    ) -> None:
        """Fill one request's packed row timesteps in an existing tensor."""
        if timesteps.shape != (self.seq_len,):
            raise ValueError(f"timesteps must have shape ({self.seq_len},)")
        # Packed-sequence timestep semantics: non-media rows (text and
        # padding) inherit the current video timestep. Later steps must reuse
        # the previous step's updated rows; re-initializing from zeros is only
        # valid at step 0.
        timesteps.fill_(float(t_video))
        timestep_groups = (
            (
                "video_target_timesteps",
                self.img_pos_dev,
                self.update_mask_dev,
                video_target_timesteps,
                t_video,
                imgvid_cond_timestep,
            ),
            (
                "audio_target_timesteps",
                self.audio_pos_dev,
                self.audio_update_mask_dev,
                audio_target_timesteps,
                t_audio if self.locked_audio_rows is None else 1.0,
                audio_ref_cond_timestep,
            ),
        )
        for name, positions, update_mask, target_timesteps, current_timestep, condition_timestep in timestep_groups:
            target_positions = positions[update_mask]
            if target_timesteps is None:
                timesteps[target_positions] = current_timestep
            else:
                target_timesteps = target_timesteps.to(device=self.device, dtype=torch.float32).reshape(-1)
                if target_timesteps.shape[0] != target_positions.shape[0]:
                    raise ValueError(
                        f"{name} rows {target_timesteps.shape[0]} != target rows {target_positions.shape[0]}"
                    )
                timesteps[target_positions] = target_timesteps
            timesteps[positions[~update_mask]] = condition_timestep


def minimax_h3_prepare_denoise_rows(
    *,
    positive: MiniMaxH3DenoiseBranch,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    audio_ref_rows: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Validate the initial rows against the layout and pin the condition rows.

    Returns ``(video_rows, audio_rows, cond_anchor, audio_anchor)`` with the
    anchors already written into their rows, which is the state both the
    request-mode loop and step-mode ``prepare_encode()`` start from.
    """
    n_cond = int((~positive.update_mask).sum())
    if keyframe_cond_rows is None:
        if n_cond != 0:
            raise ValueError(f"layout has {n_cond} cond rows but keyframe_cond_rows is None")
    else:
        if int(keyframe_cond_rows.shape[0]) != n_cond:
            raise ValueError(f"keyframe_cond_rows {int(keyframe_cond_rows.shape[0])} != layout cond rows {n_cond}")
    video_rows = initial_video_rows.to(device=device, dtype=torch.float32).clone()
    audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32).clone()
    update = positive.update_mask_dev
    audio_update = positive.audio_update_mask_dev
    if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
        raise ValueError(
            f"initial video rows {int(video_rows.shape[0])} != positive layout rows {int(positive.img_pos.shape[0])}"
        )
    if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
        raise ValueError(
            f"initial audio rows {int(audio_rows.shape[0])} != positive layout rows {int(positive.audio_pos.shape[0])}"
        )
    n_audio_ref = int((~positive.audio_update_mask).sum())
    if audio_ref_rows is None:
        if n_audio_ref != 0:
            raise ValueError(f"layout has {n_audio_ref} audio ref rows but audio_ref_rows is None")
        audio_anchor = None
    else:
        if int(audio_ref_rows.shape[0]) != n_audio_ref:
            raise ValueError(f"audio_ref_rows {int(audio_ref_rows.shape[0])} != layout audio ref rows {n_audio_ref}")
        audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
    cond_anchor = keyframe_cond_rows.to(device=device, dtype=torch.float32) if keyframe_cond_rows is not None else None
    if cond_anchor is not None:
        video_rows[~update] = cond_anchor
    if audio_anchor is not None:
        audio_rows[~audio_update] = audio_anchor
    return video_rows, audio_rows, cond_anchor, audio_anchor


# ---- VERBATIM denoise_loop.py:319-458 ----------------------------------------------------------------------------
def minimax_h3_denoise_loop(
    *,
    model: Any,
    positive: MiniMaxH3DenoiseBranch,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    audio_ref_rows: torch.Tensor | None = None,
    video_edit: MiniMaxH3LatentEdit | None = None,
    audio_edit: MiniMaxH3LatentEdit | None = None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    imgvid_cond_noise_aug_for_inference: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
    audio_cond_noise_aug_for_inference: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full denoise loop; returns final (video_rows, audio_rows).

    ``initial_video_rows`` covers all image rows of the positive layout. For a
    conditional task, pass ``keyframe_cond_rows`` and/or ``audio_ref_rows`` to
    pin those rows across every step. The model's raw positive velocity is the
    update signal; MiniMax H3 only supports cfg-distilled checkpoints.
    """
    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("video/audio sigma schedules must have equal length")
    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    # Keep CUDA/CPU on the reference path: its accuracy CI compares the
    # generated video against the official artifact. The table reuse is an
    # Ascend-specific performance optimization and must not alter that path.
    if current_omni_platform.is_npu():
        positive.prepare_rope_table(model)
    video_rows, audio_rows, cond_anchor, audio_anchor = minimax_h3_prepare_denoise_rows(
        positive=positive,
        initial_video_rows=initial_video_rows,
        initial_audio_rows=initial_audio_rows,
        keyframe_cond_rows=keyframe_cond_rows,
        audio_ref_rows=audio_ref_rows,
        device=device,
    )
    update = positive.update_mask_dev
    audio_update = positive.audio_update_mask_dev
    if video_edit is not None:
        video_edit = video_edit.to(device=device, dtype=torch.float32)
    if audio_edit is not None:
        audio_edit = audio_edit.to(device=device, dtype=torch.float32)

    num_steps = len(sigmas_video) - 1
    for step in range(num_steps):
        check_request_cancellation()
        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
        with step_cm:
            s_v, s_v_next = sigmas_video[step], sigmas_video[step + 1]
            s_a, s_a_next = sigmas_audio[step], sigmas_audio[step + 1]
            # Publish where we are so step-gated attention features (the dense
            # warmup of RAINFUSION_ATTN, the timestep gate of TRTLLM_ATTN) can
            # see it. Gates use the scheduler-style descending timestep, which
            # for a rectified-flow schedule is the video sigma.
            minimax_h3_publish_denoise_progress(step, s_v, num_steps)
            t_v, t_a = 1.0 - s_v, 1.0 - s_a
            imgvid_cond_t = max(t_v, float(imgvid_cond_noise_aug_for_inference))
            audio_ref_cond_t = max(t_a, float(audio_cond_noise_aug_for_inference))

            model_video_rows, video_target_timesteps = minimax_h3_prepare_edit_rows(
                video_rows,
                update,
                video_edit,
                t_v,
                imgvid_cond_t,
                sigma=s_v,
            )
            model_audio_rows, audio_target_timesteps = minimax_h3_prepare_edit_rows(
                audio_rows,
                audio_update,
                audio_edit,
                t_a,
                audio_ref_cond_t,
                sigma=s_a,
            )

            fk = positive.forward_kwargs(
                video_rows=model_video_rows,
                audio_rows=model_audio_rows,
                t_video=t_v,
                t_audio=t_a,
                imgvid_cond_timestep=imgvid_cond_t,
                audio_ref_cond_timestep=audio_ref_cond_t,
                video_target_timesteps=video_target_timesteps,
                audio_target_timesteps=audio_target_timesteps,
            )
            with torch.inference_mode():
                v_video, v_audio = model(**fk)
            mv_video_t = v_video.float()[update]
            mv_audio_t = v_audio.float()[audio_update]

            if video_edit is None:
                x0_video = minimax_h3_rf_v_to_x0(
                    video_rows[update],
                    mv_video_t,
                    torch.tensor(t_v, dtype=torch.float32, device=device),
                )
            else:
                x0_video = video_edit.x0(
                    model_video_rows[update],
                    mv_video_t,
                    t_v,
                )
            new_target = minimax_h3_euler_eta0_step(video_rows[update], x0_video, sigma_curr=s_v, sigma_next=s_v_next)
            video_rows = video_rows.clone()
            video_rows[update] = new_target
            if cond_anchor is not None:
                video_rows[~update] = cond_anchor  # per-step imgvid cond reset

            if audio_edit is None:
                x0_audio = minimax_h3_rf_v_to_x0(
                    audio_rows[audio_update],
                    mv_audio_t,
                    torch.tensor(t_a, dtype=torch.float32, device=device),
                )
            else:
                x0_audio = audio_edit.x0(
                    model_audio_rows[audio_update],
                    mv_audio_t,
                    t_a,
                )
            new_audio = minimax_h3_euler_eta0_step(
                audio_rows[audio_update], x0_audio, sigma_curr=s_a, sigma_next=s_a_next
            )
            audio_rows = audio_rows.clone()
            audio_rows[audio_update] = new_audio if positive.locked_audio_rows is None else positive.locked_audio_rows
            if audio_anchor is not None:
                audio_rows[~audio_update] = audio_anchor  # per-step audio ref reset
            if on_step is not None:
                on_step(step, video_rows, audio_rows)
            check_request_cancellation(synchronize=True)

    minimax_h3_publish_denoise_progress(None, None, None)
    return video_rows, audio_rows


# ---- END OF VERBATIM CODE ---------------------------------------------------------------------------------------

_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32


# ---- VERBATIM minimax_h3_transformer.py:300-332 (RoPE) ---------------------------------------------------------
class MiniMaxH3Rope(nn.Module):
    """3D rope over (t, h, w); rotates 96 of 128 head dims (rotary_percent 0.75).

    Frequency layout concatenates temporal, height, and width embeddings twice,
    with 16 frequencies per axis (inv_freq = base^-(arange(0,32,2)/32)).
    """

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.empty(inv_freq_len, dtype=_FP32_DTYPE),
            persistent=True,
        )

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """img_position_ids: [1, S, 3] (t, h, w) -> freqs [S, rot_dim=96]."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(f"img_position_ids must be [1, S, 3], got {list(img_position_ids.shape)}")
        pos = img_position_ids[0].to(_FP32_DTYPE)  # [S, 3]
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)  # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)  # each [S, 16]
        half = torch.cat((t_f, h_f, w_f), dim=-1)  # [S, 48]
        return torch.cat((half, half), dim=-1)  # [S, 96]


def _build_rope_table(freqs: torch.Tensor) -> torch.Tensor:
    """Materialize H3's packed ``[cos(freqs[:48]), sin(freqs[:48])]`` table."""
    half = freqs.shape[-1] // 2
    return torch.cat(
        (torch.cos(freqs[..., :half]), torch.sin(freqs[..., :half])),
        dim=-1,
    ).to(_BF16_DTYPE)
# ---- END OF VERBATIM CODE ---------------------------------------------------------------------------------------


# ---- fakes with upstream's call structure ------------------------------------------------------------------------
class _Backend:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class _AttentionLayer:
    """Stands for vllm-omni's ``Attention`` (``attn_backend.get_name()``, ``_kv_cache_dtype``)."""

    def __init__(self, backend: str) -> None:
        self.attn_backend = _Backend(backend)
        self._kv_cache_dtype = None


class _RefinerAttn(nn.Module):
    def __init__(self, hidden: int, backend: str) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)
        self.attention = _AttentionLayer(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(x @ x.T / math.sqrt(x.shape[-1]), dim=-1)
        return self.proj(weights @ x)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, hidden: int, backend: str) -> None:
        super().__init__()
        self.attn = _RefinerAttn(hidden, backend)
        self.mlp = nn.Linear(hidden, hidden)

    def forward(self, x, *, cu_seqlens, max_seqlen, num_requests=1):  # noqa: ANN001
        x = x + self.attn(x) * (1.0 + 0.01 * float(max_seqlen) + 0.1 * num_requests) + 0.001 * cu_seqlens[1].float()
        return x + torch.tanh(self.mlp(x))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, hidden: int, backend: str = "CUDNN_ATTN") -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3TokenRefinerBlock(hidden, backend) for _ in range(2)])
        self.calls = 0

    def forward(self, x, *, cu_seqlens, max_seqlen, num_requests=1):  # noqa: ANN001
        self.calls += 1
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, num_requests=num_requests)
        return x


class MiniMaxH3DiTModel(nn.Module):
    """Upstream's forward/_embed structure (rope_table kwarg or the inline CUDA/CPU build; condition_proj ->
    token_refiner -> index_add) on a few small layers.  ``world``/``rank`` emulate the sequence-parallel span."""

    def __init__(self, hidden: int = 16, text_dim: int = 8, backend: str = "CUDNN_ATTN", seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.hidden_size = hidden
        self.world, self.rank = 1, 0
        self.rope = MiniMaxH3Rope(16)
        with torch.no_grad():
            self.rope.inv_freq.copy_(10000.0 ** -(torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        self.condition_proj = nn.Linear(text_dim, hidden)
        self.token_refiner = MiniMaxH3TokenRefiner(hidden, backend)
        self.video_patch_proj = nn.Linear(96, hidden)
        self.audio_patch_proj = nn.Linear(32, hidden)
        self.time_w = nn.Parameter(torch.randn(hidden))
        self.rope_w = nn.Parameter(torch.randn(96, hidden) * 0.1)
        self.video_out = nn.Linear(hidden, 96)
        self.audio_out = nn.Linear(hidden, 32)
        self.inline_rope_builds = 0
        self.outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _rope_local_span(self, seq_len: int) -> tuple[int, int]:
        if self.world <= 1 or seq_len % self.world:
            return 0, seq_len
        chunk = seq_len // self.world
        return self.rank * chunk, chunk

    def _validate_prepared_rope_table(self, rope_table, *, local_len, device) -> None:  # noqa: ANN001
        if rope_table.dim() != 2 or tuple(rope_table.shape) != (local_len, 96):
            raise ValueError(f"rope_table shape {list(rope_table.shape)}")
        if rope_table.device != device:
            raise ValueError("rope_table device")
        if rope_table.dtype != _BF16_DTYPE:
            raise ValueError("rope_table dtype")

    def _embed(self, *, x, audio_x, text_embeddings_selected, unique_timesteps, img_pos, audio_pos, text_pos,  # noqa: ANN001
               refiner_cu_seqlens, refiner_max_seqlen, seq_len, device, local_span, num_requests=1):
        local_start, local_len = local_span
        text_rows = text_embeddings_selected.to(device=device, dtype=_FP32_DTYPE)
        text_embed = self.condition_proj(text_rows)
        text_embed = self.token_refiner(
            text_embed,
            cu_seqlens=refiner_cu_seqlens,
            max_seqlen=refiner_max_seqlen,
            num_requests=num_requests,
        )
        embeddings = torch.zeros((seq_len, self.hidden_size), device=device, dtype=_FP32_DTYPE)
        embeddings.index_add_(0, text_pos, text_embed)
        embeddings.index_add_(0, img_pos, self.video_patch_proj(x.view(-1, x.shape[-1]).index_select(0, img_pos)))
        audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_pos)
        embeddings.index_add_(0, audio_pos, self.audio_patch_proj(audio_rows))
        t_emb = unique_timesteps.view(-1, 1) * self.time_w
        return embeddings.narrow(0, local_start, local_len), t_emb

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x = kwargs["x"]
        img_position_ids = kwargs["img_position_ids"]
        inverse_indices = kwargs["inverse_indices"].view(-1).to(torch.long)
        img_pos = kwargs["img_pos_info"]["position_ids"].view(-1).to(torch.long)
        audio_pos = kwargs["audio_pos_info"]["position_ids"].view(-1).to(torch.long)
        text_pos = kwargs["text_pos_info"]["position_ids"].view(-1).to(torch.long)
        refiner_psp = kwargs["refiner_packed_seq_params"]
        num_requests = int(kwargs["packed_seq_params"].get("num_requests", 1))
        seq_len = int(x.shape[1])
        device = x.device
        local_span = self._rope_local_span(seq_len)
        local_start, local_len = local_span
        rope_table = kwargs.get("rope_table")
        if rope_table is None:
            self.inline_rope_builds += 1
            rope_position_ids = img_position_ids.narrow(1, local_start, local_len)
            rope_table = _build_rope_table(self.rope(rope_position_ids).to(device))
        else:
            self._validate_prepared_rope_table(rope_table, local_len=local_len, device=device)
        hidden, t_emb = self._embed(
            x=x,
            audio_x=kwargs["audio_x"],
            text_embeddings_selected=kwargs["prompt_embeds"],
            unique_timesteps=kwargs["unique_timesteps"].view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_psp["cu_seqlens_q"].to(torch.int32).to(device),
            refiner_max_seqlen=int(refiner_psp["max_seqlen_q"]),
            num_requests=num_requests,
            seq_len=seq_len,
            device=device,
            local_span=local_span,
        )
        hidden = hidden + rope_table.float() @ self.rope_w + t_emb.index_select(0, inverse_indices.narrow(0, local_start, local_len))
        full = torch.zeros((seq_len, self.hidden_size), device=device, dtype=_FP32_DTYPE)
        full[local_start : local_start + local_len] = torch.tanh(hidden)
        video = torch.tanh(self.video_out(full.index_select(0, kwargs["img_pos_for_infer_output_info"]["position_ids"])))
        audio = torch.tanh(self.audio_out(full.index_select(0, audio_pos)))
        video = video * kwargs["update_mask"].view(-1).unsqueeze(-1)
        self.outputs.append((video.clone(), audio.clone()))
        return video, audio


class MiniMaxH3Pipeline:
    def __init__(self, transformer: MiniMaxH3DiTModel) -> None:
        self.transformer = transformer
        self.device = torch.device("cpu")

    def diffuse(self, *, branch, **loop_kwargs):  # noqa: ANN001
        # the real pipeline imports the loop by name and calls it here (pipeline_minimax_h3.py:1973)
        return minimax_h3_denoise_loop(model=self.transformer, positive=branch, device=self.device, **loop_kwargs)


def load(name: str):  # noqa: ANN201
    """A fresh copy of this module (own classes and globals) so stock and patched runs never share state."""
    spec = importlib.util.spec_from_file_location(name, __file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
