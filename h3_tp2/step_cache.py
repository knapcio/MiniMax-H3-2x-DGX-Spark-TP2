"""Per-request step caches for the MiniMax H3 denoise loop (opt-in, ``H3_TP2_STEP_CACHE=1``; speed plan item 7).

Upstream (vLLM-Omni v0.30.0, a8576ccb7, ``vllm_omni/diffusion/models/minimax_h3/``) recomputes per denoising step
things that are constant for the whole request:

(a) RoPE table.  ``minimax_h3_transformer.py:1609-1620``: on CUDA/CPU ``forward`` rebuilds
    ``_build_rope_table(self.rope(img_position_ids.narrow(1, local_start, local_len)).to(device))`` every step (only
    the NPU path prepares it once, ``denoise_loop.py:351``).  We build it with the same expression on the first step
    of a ``diffuse()`` call and pass it as the upstream-supported ``rope_table`` kwarg (validated by
    ``_validate_prepared_rope_table``) on every step.  Key: model, ``img_position_ids`` (same object or same storage,
    shape, strides, dtype, device, version), ``seq_len`` and the sequence-parallel ``local_span``.
(b) Text token refiner.  ``minimax_h3_transformer.py:1515``: the two refiner blocks rerun every step on the constant
    text rows.  ``_embed`` is wrapped to publish the call's key (``text_embeddings_selected``, ``refiner_cu_seqlens``,
    ``refiner_max_seqlen``, ``num_requests``, device); ``MiniMaxH3TokenRefiner.forward`` returns the first step's
    output when the key matches.  ``condition_proj`` (one 602x5120x5376 GEMM) still runs; its output is ignored on a
    hit.  Refused (logged once) when a refiner attention layer is not a step-independent dense backend, has a
    quantised KV dtype, or any refiner module carries hooks (layerwise offload, cache hooks), because then the output
    could depend on the step.
(c) Constant boolean masks.  ``denoise_loop.py:255/265`` (``positions[update_mask]``, ``positions[~update_mask]``),
    ``:413-452`` (``v_video.float()[update]``, ``video_rows[update]``, ``video_rows[update] = ...``, ``[~update]``,
    the audio twins) each run a ``nonzero`` and a device->host sync per step.  The loop is replaced by a copy that
    computes ``nonzero`` once per request and uses ``index_select`` / ``index_copy_`` (same rows, same ascending
    order, pure copies: bit-identical).  ``torch.unique(timesteps)`` (``:208``, a sort and a sync per step) is
    replaced, when no latent edit supplies per-row target timesteps, by the unique of the <= 5 group values on the
    host (rows are grouped once per request by the same write order as ``fill_timesteps``) and a gather; values,
    order and dtypes equal ``torch.unique(sorted=True, return_inverse=True)``.
(d) ``check_request_cancellation(synchronize=True)`` (``denoise_loop.py:455``) is left alone: it synchronises only
    when a cancel is pending locally; its per-step cost is a gloo all_reduce over the two hosts (one at the start and
    one at the end of each step).  Skipping one would change the number of collectives per rank, and a rank that
    disagrees (patch on one host only) deadlocks; the saving would be ~0.2 ms of 21 s.

Scope and invalidation: (a) and (b) are active only inside ``MiniMaxH3Pipeline.diffuse`` (the method h3_output also
wraps); both caches are cleared when a ``diffuse`` call starts and when it ends, so step mode (``prepare_encode`` /
``denoise_step``) and anything outside a request is untouched.  (c) is active for one call of the loop (index state on
the branch, removed in ``finally``).

Guards: each patch is installed only if the upstream function it depends on has the reviewed source (fingerprint of
the stripped lines, see ``EXPECTED``); on a mismatch it is skipped with a warning.  At run time every assumption
(tensor kwargs, ranks, 1-D bool masks of the right length, shapes/dtypes for ``index_copy_``) falls back to the stock
code path, logged once.  The patches never catch exceptions from upstream code.  ``H3_TP2_STEP_CACHE=verify`` also
recomputes the stock value on every cache hit (and the stock ``torch.unique``) and compares with ``torch.equal``
(device syncs; qualification only); a mismatch is logged as an error and the stock value is used.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from typing import Any

import torch

logger = logging.getLogger("vllm_omni.h3_tp2.step_cache")

# sha256 (first 16 hex) of the stripped, non-empty, non-comment source lines of the upstream functions each patch
# relies on, vLLM-Omni v0.30.0 a8576ccb7 (~/h3-long/vllm-omni on Spark_06, 2026-11-02).
EXPECTED = {
    "denoise_loop": "4f6405957f5362ca",  # denoise_loop.py:319-458
    "forward_kwargs": "90a662124387494d",  # denoise_loop.py:182-215
    "fill_timesteps": "1394e2bec6dcf71d",  # denoise_loop.py:217-265
    "dit_forward": "8154c8e90a4cd8b8",  # minimax_h3_transformer.py:1548-1719
    "dit_embed": "b4ce1fa37b2ac35d",  # minimax_h3_transformer.py:1458-1546
}

# Attention backends whose output does not depend on the denoise step (no step/timestep gate, no sparsity schedule).
STEP_FREE_BACKENDS = frozenset(
    {"CUDNN_ATTN", "FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB", "FLASHINFER_ATTN", "SDPA"}
)


def fingerprint_source(src: str) -> str:
    lines = []
    for line in src.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("@"):
            continue
        lines.append(s)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def fingerprint(fn) -> str | None:  # noqa: ANN001
    try:
        return fingerprint_source(inspect.getsource(fn))
    except (OSError, TypeError):
        return None


class _State:
    def __init__(self) -> None:
        self.depth = 0
        self.verify = False
        self.rope: dict[int, dict] = {}
        self.refiner: dict[int, dict] = {}
        self.refiner_ctx: dict | None = None
        self.warned: set[str] = set()
        self.stats: dict[str, int] = {}

    def clear(self) -> None:
        self.rope.clear()
        self.refiner.clear()
        self.refiner_ctx = None

    def bump(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1


_STATE = _State()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key in _STATE.warned:
        return
    _STATE.warned.add(key)
    logger.warning("h3_tp2 step cache: " + msg, *args)


def _version(t: torch.Tensor):  # noqa: ANN202
    try:
        return t._version
    except RuntimeError:  # inference tensors do not track a version counter
        return None


def _tsig(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype, str(t.device), _version(t))


def _sample(t: torch.Tensor) -> torch.Tensor:
    flat = t.detach().reshape(-1)
    step = max(1, flat.numel() // 64)
    return flat[::step][:64].clone()


def _same_tensor(entry_t: torch.Tensor, entry_sig: tuple, entry_sample: torch.Tensor | None, t: torch.Tensor) -> bool:
    """Same storage, geometry and version counter (where tracked) as when cached: the entry holds a reference to the
    old tensor, so equal data pointers mean the same memory.  The strided-sample content check runs where it is free
    (CPU) and in verify mode; on CUDA it would be a device sync per step."""
    del entry_t  # kept alive by the entry; the signature is the comparison
    if _tsig(t) != entry_sig:
        return False
    if entry_sample is not None and (t.device.type == "cpu" or _STATE.verify):
        return torch.equal(_sample(t), entry_sample)
    return True


def _key_tensor(t: torch.Tensor) -> tuple:
    return (t, _tsig(t), _sample(t) if (t.device.type == "cpu" or _STATE.verify) else None)


def _same_key_tensor(k: tuple, t: torch.Tensor) -> bool:
    return _same_tensor(k[0], k[1], k[2], t)


# --------------------------------------------------------------------------------------------------------------------
# diffuse scope


def _install_scope(pp) -> None:  # noqa: ANN001
    cls = pp.MiniMaxH3Pipeline
    if getattr(cls, "_h3_tp2_step_cache_scope", False):
        return
    orig = cls.diffuse

    def diffuse(self, *a, **kw):  # noqa: ANN001
        _STATE.depth += 1
        _STATE.clear()
        _STATE.stats = {}
        try:
            return orig(self, *a, **kw)
        finally:
            _STATE.depth -= 1
            _STATE.clear()
            if _STATE.stats:
                logger.info("h3_tp2 step cache: %s", " ".join(f"{k}={v}" for k, v in sorted(_STATE.stats.items())))

    cls.diffuse = diffuse
    cls._h3_tp2_step_cache_scope = True


# --------------------------------------------------------------------------------------------------------------------
# (a) RoPE table


def _rope_for(model, kw: dict, build) -> torch.Tensor | None:  # noqa: ANN001
    x = kw.get("x")
    ipi = kw.get("img_position_ids")
    if not isinstance(x, torch.Tensor) or not isinstance(ipi, torch.Tensor):
        _warn_once("rope-args", "rope: x / img_position_ids not tensors; stock path")
        return None
    if x.dim() != 3 or ipi.dim() != 3 or ipi.shape[0] != 1:
        _warn_once("rope-shape", "rope: unexpected shapes x=%s img_position_ids=%s; stock path",
                   tuple(x.shape), tuple(ipi.shape))
        return None
    seq_len = int(x.shape[1])
    device = x.device
    local_start, local_len = (int(v) for v in model._rope_local_span(seq_len))
    if local_start < 0 or local_len <= 0 or local_start + local_len > ipi.shape[1]:
        _warn_once("rope-span", "rope: span %s outside img_position_ids %s; stock path",
                   (local_start, local_len), tuple(ipi.shape))
        return None
    extra = (seq_len, local_start, local_len, str(device))
    ent = _STATE.rope.get(id(model))
    if ent is not None and ent["model"] is model and ent["extra"] == extra and _same_key_tensor(ent["src"], ipi):
        table = ent["table"]
        if _STATE.verify:
            fresh = build(model.rope(ipi.narrow(1, local_start, local_len)).to(device))
            if not torch.equal(fresh, table):
                logger.error("h3_tp2 step cache VERIFY: cached RoPE table differs from the stock one; using stock")
                _STATE.bump("rope_verify_mismatch")
                ent["table"] = fresh
                return fresh
        _STATE.bump("rope_reused")
        return table
    # the exact CUDA/CPU expression of MiniMaxH3DiTModel.forward
    table = build(model.rope(ipi.narrow(1, local_start, local_len)).to(device))
    model._validate_prepared_rope_table(table, local_len=local_len, device=device)
    _STATE.rope[id(model)] = {"model": model, "extra": extra, "src": _key_tensor(ipi), "table": table}
    _STATE.bump("rope_built")
    return table


def _install_rope(tr) -> None:  # noqa: ANN001
    cls = tr.MiniMaxH3DiTModel
    if getattr(cls, "_h3_tp2_step_cache_rope", False):
        return
    orig = cls.forward

    def forward(self, **kw):  # noqa: ANN001
        if _STATE.depth <= 0 or kw.get("rope_table") is not None:
            return orig(self, **kw)
        try:
            table = _rope_for(self, kw, tr._build_rope_table)
        except Exception as e:  # noqa: BLE001  (stock forward raises its own error for bad inputs)
            _warn_once("rope-err", "rope: precompute failed (%s: %s); stock path", type(e).__name__, e)
            table = None
        if table is None:
            return orig(self, **kw)
        kw2 = dict(kw)
        kw2["rope_table"] = table
        return orig(self, **kw2)

    cls.forward = forward
    cls._h3_tp2_step_cache_rope = True


# --------------------------------------------------------------------------------------------------------------------
# (b) text token refiner


def _refiner_step_independent(model) -> tuple[bool, str]:  # noqa: ANN001
    refiner = getattr(model, "token_refiner", None)
    blocks = getattr(refiner, "blocks", None)
    if refiner is None or blocks is None:
        return False, "no token_refiner.blocks"
    for name, mod in refiner.named_modules():
        if mod._forward_hooks or mod._forward_pre_hooks or getattr(mod, "_hook_registry", None) is not None:
            return False, f"hooks on token_refiner.{name or '<root>'}"
    for i, block in enumerate(blocks):
        layer = getattr(getattr(block, "attn", None), "attention", None)
        backend = getattr(layer, "attn_backend", None)
        bname = backend.get_name() if backend is not None and hasattr(backend, "get_name") else None
        if bname not in STEP_FREE_BACKENDS:
            return False, f"block {i} attention backend {bname}"
        if getattr(layer, "_kv_cache_dtype", None) not in (None, "float"):
            return False, f"block {i} quantised KV ({getattr(layer, '_kv_cache_dtype', None)})"
    return True, ""


def _refiner_ctx(model, kw: dict) -> dict | None:  # noqa: ANN001
    ok = model.__dict__.get("_h3_tp2_refiner_ok")
    if ok is None:  # decided once per model, on its first request (after loading)
        ok = _refiner_step_independent(model)
        model.__dict__["_h3_tp2_refiner_ok"] = ok
        if not ok[0]:
            logger.warning("h3_tp2 step cache: refiner cache refused (%s); stock refiner", ok[1])
    if not ok[0]:
        return None
    text = kw.get("text_embeddings_selected")
    cu = kw.get("refiner_cu_seqlens")
    max_seqlen = kw.get("refiner_max_seqlen")
    if not isinstance(text, torch.Tensor) or not isinstance(cu, torch.Tensor) or max_seqlen is None:
        _warn_once("refiner-args", "refiner: unexpected _embed arguments; stock refiner")
        return None
    extra = (int(max_seqlen), int(kw.get("num_requests", 1)), str(kw.get("device")), tuple(text.shape))
    ent = _STATE.refiner.get(id(model))
    hit = (
        ent is not None
        and ent["model"] is model
        and ent["extra"] == extra
        and _same_key_tensor(ent["text"], text)
        and _same_key_tensor(ent["cu"], cu)
    )
    return {
        "model": model,
        "refiner": model.token_refiner,
        "extra": extra,
        "text": text,
        "cu": cu,
        "cached": ent["out"] if hit else None,
        "consumed": False,
    }


def _install_refiner(tr) -> None:  # noqa: ANN001
    dit = tr.MiniMaxH3DiTModel
    ref = tr.MiniMaxH3TokenRefiner
    if getattr(dit, "_h3_tp2_step_cache_refiner", False):
        return
    orig_embed = dit._embed
    orig_ref = ref.forward

    def _embed(self, *a, **kw):  # noqa: ANN001
        if _STATE.depth <= 0 or a:
            return orig_embed(self, *a, **kw)
        try:
            ctx = _refiner_ctx(self, kw)
        except Exception as e:  # noqa: BLE001
            _warn_once("refiner-err", "refiner: key failed (%s: %s); stock refiner", type(e).__name__, e)
            ctx = None
        if ctx is None:
            return orig_embed(self, **kw)
        prev = _STATE.refiner_ctx
        _STATE.refiner_ctx = ctx
        try:
            return orig_embed(self, **kw)
        finally:
            _STATE.refiner_ctx = prev

    def forward(self, x, *a, **kw):  # noqa: ANN001
        ctx = _STATE.refiner_ctx
        if ctx is None or ctx["refiner"] is not self or ctx["consumed"] or a:
            return orig_ref(self, x, *a, **kw)
        ctx["consumed"] = True
        cu = kw.get("cu_seqlens")
        if (
            cu is not ctx["cu"]
            or kw.get("max_seqlen") != ctx["extra"][0]
            or int(kw.get("num_requests", 1)) != ctx["extra"][1]
            or not isinstance(x, torch.Tensor)
            or x.shape[0] != ctx["text"].shape[0]
        ):
            _warn_once("refiner-call", "refiner: call does not match the _embed arguments; stock refiner")
            return orig_ref(self, x, *a, **kw)
        cached = ctx["cached"]
        if cached is not None:
            if _STATE.verify:
                fresh = orig_ref(self, x, **kw)
                if not torch.equal(fresh, cached):
                    logger.error("h3_tp2 step cache VERIFY: cached refiner output differs from the stock one; using stock")
                    _STATE.bump("refiner_verify_mismatch")
                    _STATE.refiner.pop(id(ctx["model"]), None)
                    return fresh
            _STATE.bump("refiner_reused")
            return cached
        out = orig_ref(self, x, **kw)
        if isinstance(out, torch.Tensor):
            _STATE.refiner[id(ctx["model"])] = {
                "model": ctx["model"],
                "extra": ctx["extra"],
                "text": _key_tensor(ctx["text"]),
                "cu": _key_tensor(ctx["cu"]),
                "out": out,
            }
            _STATE.bump("refiner_run")
        return out

    dit._embed = _embed
    ref.forward = forward
    dit._h3_tp2_step_cache_refiner = True


# --------------------------------------------------------------------------------------------------------------------
# (c) index tensors for the constant masks + torch.unique from row groups


class _Idx:
    """Per-request indices of the branch's constant masks (computed once: one nonzero + sync per mask)."""

    def __init__(self, branch, verify: bool) -> None:  # noqa: ANN001
        self.update_mask = branch.update_mask_dev
        self.audio_update_mask = branch.audio_update_mask_dev
        self.img_pos = branch.img_pos_dev
        self.audio_pos = branch.audio_pos_dev
        self.seq_len = int(branch.seq_len)
        self.upd = torch.nonzero(self.update_mask, as_tuple=False).view(-1)
        self.cond = torch.nonzero(~self.update_mask, as_tuple=False).view(-1)
        self.aupd = torch.nonzero(self.audio_update_mask, as_tuple=False).view(-1)
        self.acond = torch.nonzero(~self.audio_update_mask, as_tuple=False).view(-1)
        self.vid_target_pos = self.img_pos.index_select(0, self.upd)
        self.vid_cond_pos = self.img_pos.index_select(0, self.cond)
        self.aud_target_pos = self.audio_pos.index_select(0, self.aupd)
        self.aud_cond_pos = self.audio_pos.index_select(0, self.acond)
        # Row groups in fill_timesteps' write order: 0 = fill (text, padding), 1 video target, 2 video condition,
        # 3 audio target, 4 audio condition.  The last write wins, exactly as for the timesteps themselves.
        device = self.img_pos.device
        g = torch.zeros(self.seq_len, dtype=torch.long, device=device)
        for gid, pos in ((1, self.vid_target_pos), (2, self.vid_cond_pos), (3, self.aud_target_pos), (4, self.aud_cond_pos)):
            g[pos] = gid
        self.groups = g
        self.present = [int(v) for v in torch.unique(g).tolist()]  # once per request
        if verify:
            ok = (
                torch.equal(self.vid_target_pos, self.img_pos[self.update_mask])
                and torch.equal(self.vid_cond_pos, self.img_pos[~self.update_mask])
                and torch.equal(self.aud_target_pos, self.audio_pos[self.audio_update_mask])
                and torch.equal(self.aud_cond_pos, self.audio_pos[~self.audio_update_mask])
            )
            if not ok:
                raise RuntimeError("index tensors differ from the boolean-mask selection")

    def valid_for(self, branch) -> bool:  # noqa: ANN001
        return (
            self.update_mask is branch.update_mask_dev
            and self.audio_update_mask is branch.audio_update_mask_dev
            and self.img_pos is branch.img_pos_dev
            and self.audio_pos is branch.audio_pos_dev
            and self.seq_len == int(branch.seq_len)
        )


def _make_idx(branch) -> _Idx | None:  # noqa: ANN001
    try:
        um, am = branch.update_mask_dev, branch.audio_update_mask_dev
        ip, ap = branch.img_pos_dev, branch.audio_pos_dev
        ok = (
            all(isinstance(t, torch.Tensor) for t in (um, am, ip, ap))
            and um.dtype == torch.bool and am.dtype == torch.bool and um.dim() == 1 and am.dim() == 1
            and ip.dim() == 1 and ap.dim() == 1 and ip.dtype == torch.long and ap.dtype == torch.long
            and um.shape[0] == ip.shape[0] and am.shape[0] == ap.shape[0]
            and um.device == ip.device and am.device == ap.device == ip.device
        )
        if not ok:
            _warn_once("idx-shape", "index loop: masks/positions not 1-D bool/long of matching length; stock loop")
            return None
        return _Idx(branch, _STATE.verify)
    except Exception as e:  # noqa: BLE001
        _warn_once("idx-err", "index loop: setup failed (%s: %s); stock loop", type(e).__name__, e)
        return None


def _branch_idx(branch) -> _Idx | None:  # noqa: ANN001
    idx = branch.__dict__.get("_h3_tp2_idx")
    if idx is None or not idx.valid_for(branch):
        return None
    return idx


def _put(dst: torch.Tensor, index: torch.Tensor, mask_fn, value) -> None:  # noqa: ANN001
    """``dst[mask] = value`` as ``index_copy_`` when the value is exactly the selected rows; else the stock indexing
    (broadcasting / dtype casting stay upstream's)."""
    if (
        isinstance(value, torch.Tensor)
        and value.dim() == dst.dim()
        and value.shape[0] == index.shape[0]
        and value.shape[1:] == dst.shape[1:]
        and value.dtype == dst.dtype
        and value.device == dst.device
    ):
        dst.index_copy_(0, index, value)
    else:
        _STATE.bump("idx_put_stock")
        dst[mask_fn()] = value


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if torch.device(device).type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)  # no stream sync (a pageable H2D copy syncs)
    return t.to(device)


def _unique_from_groups(idx: _Idx, values: list[float], device) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    # float32 conversion exactly as fill_() / index_put_ convert the Python scalars
    vals = torch.tensor(values, dtype=torch.float32)
    present = torch.tensor(idx.present, dtype=torch.long)
    uniq, inv = torch.unique(vals.index_select(0, present), sorted=True, return_inverse=True)
    gmap = torch.full((5,), -1, dtype=torch.long)
    gmap[present] = inv
    gmap_dev = _to_device(gmap, device)
    return _to_device(uniq, device), gmap_dev.index_select(0, idx.groups)


def _install_index_loop(dl, pp) -> None:  # noqa: ANN001
    branch_cls = dl.MiniMaxH3DenoiseBranch
    if getattr(branch_cls, "_h3_tp2_step_cache_idx", False):
        return
    orig_fill = branch_cls.fill_timesteps
    orig_fk = branch_cls.forward_kwargs
    orig_loop = dl.minimax_h3_denoise_loop

    def fill_timesteps(self, timesteps, *, t_video, t_audio, imgvid_cond_timestep, audio_ref_cond_timestep,  # noqa: ANN001
                       video_target_timesteps=None, audio_target_timesteps=None):
        idx = _branch_idx(self)
        if idx is None:
            return orig_fill(self, timesteps, t_video=t_video, t_audio=t_audio,
                             imgvid_cond_timestep=imgvid_cond_timestep,
                             audio_ref_cond_timestep=audio_ref_cond_timestep,
                             video_target_timesteps=video_target_timesteps,
                             audio_target_timesteps=audio_target_timesteps)
        # upstream fill_timesteps with positions[update_mask] / positions[~update_mask] precomputed
        if timesteps.shape != (self.seq_len,):
            raise ValueError(f"timesteps must have shape ({self.seq_len},)")
        timesteps.fill_(float(t_video))
        timestep_groups = (
            ("video_target_timesteps", idx.vid_target_pos, idx.vid_cond_pos, video_target_timesteps, t_video,
             imgvid_cond_timestep),
            ("audio_target_timesteps", idx.aud_target_pos, idx.aud_cond_pos, audio_target_timesteps,
             t_audio if self.locked_audio_rows is None else 1.0, audio_ref_cond_timestep),
        )
        for name, target_positions, cond_positions, target_timesteps, current_timestep, condition_timestep in timestep_groups:
            if target_timesteps is None:
                timesteps[target_positions] = current_timestep
            else:
                target_timesteps = target_timesteps.to(device=self.device, dtype=torch.float32).reshape(-1)
                if target_timesteps.shape[0] != target_positions.shape[0]:
                    raise ValueError(
                        f"{name} rows {target_timesteps.shape[0]} != target rows {target_positions.shape[0]}"
                    )
                timesteps[target_positions] = target_timesteps
            timesteps[cond_positions] = condition_timestep

    def forward_kwargs(self, *, video_rows, audio_rows, t_video, t_audio, imgvid_cond_timestep,  # noqa: ANN001
                       audio_ref_cond_timestep, video_target_timesteps=None, audio_target_timesteps=None):
        kw = dict(video_rows=video_rows, audio_rows=audio_rows, t_video=t_video, t_audio=t_audio,
                  imgvid_cond_timestep=imgvid_cond_timestep, audio_ref_cond_timestep=audio_ref_cond_timestep,
                  video_target_timesteps=video_target_timesteps, audio_target_timesteps=audio_target_timesteps)
        idx = _branch_idx(self)
        if idx is None:
            return orig_fk(self, **kw)
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
        if video_target_timesteps is None and audio_target_timesteps is None:
            cur_audio = t_audio if self.locked_audio_rows is None else 1.0
            values = [float(t_video), float(t_video), float(imgvid_cond_timestep), float(cur_audio),
                      float(audio_ref_cond_timestep)]
            unique_timesteps, inverse_indices = _unique_from_groups(idx, values, self.device)
            _STATE.bump("unique_host")
            if _STATE.verify:
                ref_u, ref_i = torch.unique(timesteps, sorted=True, return_inverse=True)
                if not (torch.equal(ref_u, unique_timesteps) and torch.equal(ref_i, inverse_indices)):
                    logger.error("h3_tp2 step cache VERIFY: host unique differs from torch.unique; using stock")
                    _STATE.bump("unique_verify_mismatch")
                    unique_timesteps, inverse_indices = ref_u, ref_i
        else:
            unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)
            _STATE.bump("unique_stock")
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
        }

    def minimax_h3_denoise_loop(
        *,
        model,  # noqa: ANN001
        positive,  # noqa: ANN001
        initial_video_rows,  # noqa: ANN001
        initial_audio_rows,  # noqa: ANN001
        keyframe_cond_rows,  # noqa: ANN001
        audio_ref_rows=None,  # noqa: ANN001
        video_edit=None,  # noqa: ANN001
        audio_edit=None,  # noqa: ANN001
        sigmas_video,  # noqa: ANN001
        sigmas_audio,  # noqa: ANN001
        device,  # noqa: ANN001
        imgvid_cond_noise_aug_for_inference=dl.MINIMAX_H3_IMGVID_COND_TIMESTEP,  # noqa: ANN001
        audio_cond_noise_aug_for_inference=dl.MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,  # noqa: ANN001
        on_step=None,  # noqa: ANN001
        step_profiler=None,  # noqa: ANN001
    ):
        args = dict(model=model, positive=positive, initial_video_rows=initial_video_rows,
                    initial_audio_rows=initial_audio_rows, keyframe_cond_rows=keyframe_cond_rows,
                    audio_ref_rows=audio_ref_rows, video_edit=video_edit, audio_edit=audio_edit,
                    sigmas_video=sigmas_video, sigmas_audio=sigmas_audio, device=device,
                    imgvid_cond_noise_aug_for_inference=imgvid_cond_noise_aug_for_inference,
                    audio_cond_noise_aug_for_inference=audio_cond_noise_aug_for_inference,
                    on_step=on_step, step_profiler=step_profiler)
        if len(sigmas_video) != len(sigmas_audio):
            raise ValueError("video/audio sigma schedules must have equal length")
        if len(sigmas_video) < 2:
            raise ValueError("sigma schedules need at least 2 entries")
        idx = _make_idx(positive)
        if idx is None:
            return orig_loop(**args)
        _STATE.bump("index_loop")
        positive._h3_tp2_idx = idx
        try:
            return _indexed_loop(dl, idx, **args)
        finally:
            positive.__dict__.pop("_h3_tp2_idx", None)

    minimax_h3_denoise_loop.__wrapped_stock__ = orig_loop
    branch_cls.fill_timesteps = fill_timesteps
    branch_cls.forward_kwargs = forward_kwargs
    dl.minimax_h3_denoise_loop = minimax_h3_denoise_loop
    if pp is not None and getattr(pp, "minimax_h3_denoise_loop", None) is orig_loop:
        pp.minimax_h3_denoise_loop = minimax_h3_denoise_loop  # the pipeline imports the loop by name
    branch_cls._h3_tp2_step_cache_idx = True


def _indexed_loop(dl, idx: _Idx, *, model, positive, initial_video_rows, initial_audio_rows,  # noqa: ANN001, PLR0913
                  keyframe_cond_rows, audio_ref_rows, video_edit, audio_edit, sigmas_video, sigmas_audio, device,
                  imgvid_cond_noise_aug_for_inference, audio_cond_noise_aug_for_inference, on_step, step_profiler):
    """upstream minimax_h3_denoise_loop (denoise_loop.py:319-458) with the boolean-mask selections and assignments of
    the constant ``update`` / ``audio_update`` masks done through precomputed indices.  The helpers are looked up on
    the upstream module, so their validation (and its syncs) is unchanged."""
    from contextlib import nullcontext

    if dl.current_omni_platform.is_npu():
        positive.prepare_rope_table(model)
    video_rows, audio_rows, cond_anchor, audio_anchor = dl.minimax_h3_prepare_denoise_rows(
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
        dl.check_request_cancellation()
        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
        with step_cm:
            s_v, s_v_next = sigmas_video[step], sigmas_video[step + 1]
            s_a, s_a_next = sigmas_audio[step], sigmas_audio[step + 1]
            dl.minimax_h3_publish_denoise_progress(step, s_v, num_steps)
            t_v, t_a = 1.0 - s_v, 1.0 - s_a
            imgvid_cond_t = max(t_v, float(imgvid_cond_noise_aug_for_inference))
            audio_ref_cond_t = max(t_a, float(audio_cond_noise_aug_for_inference))

            model_video_rows, video_target_timesteps = dl.minimax_h3_prepare_edit_rows(
                video_rows, update, video_edit, t_v, imgvid_cond_t, sigma=s_v,
            )
            model_audio_rows, audio_target_timesteps = dl.minimax_h3_prepare_edit_rows(
                audio_rows, audio_update, audio_edit, t_a, audio_ref_cond_t, sigma=s_a,
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
            mv_video_t = v_video.float().index_select(0, idx.upd)  # == v_video.float()[update]
            mv_audio_t = v_audio.float().index_select(0, idx.aupd)

            video_target = video_rows.index_select(0, idx.upd)  # == video_rows[update] (read twice upstream)
            if video_edit is None:
                x0_video = dl.minimax_h3_rf_v_to_x0(
                    video_target,
                    mv_video_t,
                    torch.tensor(t_v, dtype=torch.float32, device=device),
                )
            else:
                x0_video = video_edit.x0(
                    model_video_rows.index_select(0, idx.upd),
                    mv_video_t,
                    t_v,
                )
            new_target = dl.minimax_h3_euler_eta0_step(video_target, x0_video, sigma_curr=s_v, sigma_next=s_v_next)
            video_rows = video_rows.clone()
            _put(video_rows, idx.upd, lambda: update, new_target)
            if cond_anchor is not None:
                _put(video_rows, idx.cond, lambda: ~update, cond_anchor)  # per-step imgvid cond reset

            audio_target = audio_rows.index_select(0, idx.aupd)
            if audio_edit is None:
                x0_audio = dl.minimax_h3_rf_v_to_x0(
                    audio_target,
                    mv_audio_t,
                    torch.tensor(t_a, dtype=torch.float32, device=device),
                )
            else:
                x0_audio = audio_edit.x0(
                    model_audio_rows.index_select(0, idx.aupd),
                    mv_audio_t,
                    t_a,
                )
            new_audio = dl.minimax_h3_euler_eta0_step(audio_target, x0_audio, sigma_curr=s_a, sigma_next=s_a_next)
            audio_rows = audio_rows.clone()
            _put(audio_rows, idx.aupd, lambda: audio_update,
                 new_audio if positive.locked_audio_rows is None else positive.locked_audio_rows)
            if audio_anchor is not None:
                _put(audio_rows, idx.acond, lambda: ~audio_update, audio_anchor)  # per-step audio ref reset
            if on_step is not None:
                on_step(step, video_rows, audio_rows)
            dl.check_request_cancellation(synchronize=True)

    dl.minimax_h3_publish_denoise_progress(None, None, None)
    return video_rows, audio_rows


# --------------------------------------------------------------------------------------------------------------------


def install(dl, tr, pp, *, verify: bool = False, check_source: bool = True) -> list[str]:  # noqa: ANN001
    """Install the patches on the upstream modules ``denoise_loop`` (dl), ``minimax_h3_transformer`` (tr) and
    ``pipeline_minimax_h3`` (pp).  Returns the names of the installed patches; a patch whose upstream source differs
    from the reviewed one, or whose install raises, is skipped with a warning."""
    _STATE.verify = bool(verify)
    done: list[str] = []

    def source_ok(name: str, fn) -> bool:  # noqa: ANN001
        if not check_source:
            return True
        fp = fingerprint(fn)
        if fp != EXPECTED[name]:
            logger.warning("h3_tp2 step cache: upstream %s changed (fingerprint %s, reviewed %s); patch skipped",
                           name, fp, EXPECTED[name])
            return False
        return True

    try:
        _install_scope(pp)
    except Exception:  # noqa: BLE001
        logger.warning("h3_tp2 step cache: diffuse scope not installed; RoPE/refiner caches stay off", exc_info=True)
    else:
        try:
            if source_ok("dit_forward", tr.MiniMaxH3DiTModel.forward):
                _install_rope(tr)
                done.append("rope")
        except Exception:  # noqa: BLE001
            logger.warning("h3_tp2 step cache: RoPE patch not installed", exc_info=True)
        try:
            if source_ok("dit_embed", tr.MiniMaxH3DiTModel._embed):
                _install_refiner(tr)
                done.append("refiner")
        except Exception:  # noqa: BLE001
            logger.warning("h3_tp2 step cache: refiner patch not installed", exc_info=True)
    try:
        if (source_ok("denoise_loop", dl.minimax_h3_denoise_loop)
                and source_ok("forward_kwargs", dl.MiniMaxH3DenoiseBranch.forward_kwargs)
                and source_ok("fill_timesteps", dl.MiniMaxH3DenoiseBranch.fill_timesteps)):
            _install_index_loop(dl, pp)
            done.append("index")
    except Exception:  # noqa: BLE001
        logger.warning("h3_tp2 step cache: index-loop patch not installed", exc_info=True)
    return done
