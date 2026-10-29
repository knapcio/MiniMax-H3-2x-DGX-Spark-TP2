"""Process-level patches for the two-Spark layout.

Loaded through the ``vllm.general_plugins`` entry point, so it runs in the API
server, the stage engine process and every diffusion worker (vLLM-Omni calls
``load_omni_general_plugins()`` in all of them).  Everything is gated by
``H3_TP2_ROLE`` (``head`` or ``worker``); without it the plugin is a no-op and
the stock single-host behaviour is untouched.

Patches (all numerics-neutral):
1. head: ``get_device_count()`` reports the world size so the stage-init
   device check (``world_size <= device_count``) passes on a one-GPU host.
   NOT applied on the worker host, where ``rank % device_count`` must stay 0.
2. all roles: ``get_torch_device()`` always returns ``cuda:0`` (one GPU per host).
3. all roles: ``init_distributed_environment`` is wrapped so ``MASTER_ADDR`` /
   ``MASTER_PORT`` come from ``H3_TP2_MASTER_ADDR`` / ``H3_TP2_MASTER_PORT``
   instead of the hard-coded ``localhost`` in ``DiffusionWorker.init_device``.
4. worker: the per-worker result ``MessageQueue`` is created as a remote-reader
   queue bound to ``H3_TP2_RESULT_CONNECT_IP`` (the fabric address), so the
   head can subscribe to it over ZMQ instead of shared memory.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm_omni.h3_tp2.plugin")

_APPLIED = False



def insert_anchor_block(args: dict, anchor_rows, anchor_v: int) -> dict:
    """Put ``anchor_rows`` (packed [n, 96] video-latent rows of ``anchor_v`` latent frames) into a continuation
    window's conditioning right before upstream's latent guide, which is always the last visual block: the rows, the
    shape list and the ``ref_blocks`` entry (kind ``video``, no audio rows).  Mutates and returns ``args``."""
    import torch

    rows, shapes, blocks = args["visual_condition"], list(args["visual_condition_shapes"]), list(args["ref_blocks"])
    if not blocks or blocks[-1].get("kind") != "latent_guide" or tuple(shapes[-1])[0] != blocks[-1]["latent_t"]:
        raise RuntimeError("h3_tp2 anchor: expected upstream's latent guide as the last visual block")
    gt, gh, gw = (int(v) for v in shapes[-1])
    guide_rows = gt * (gh // 2) * (gw // 2)
    if rows.shape[0] < guide_rows or anchor_rows.shape[0] != anchor_v * (gh // 2) * (gw // 2):
        raise RuntimeError("h3_tp2 anchor: condition rows do not match the window canvas")
    anc = anchor_rows.to(device=rows.device, dtype=rows.dtype)
    cut = rows.shape[0] - guide_rows
    args["visual_condition"] = torch.cat((rows[:cut], anc, rows[cut:]), dim=0)
    shapes.insert(len(shapes) - 1, (anchor_v, gh, gw))
    blocks.insert(len(blocks) - 1, {"kind": "video", "ref_audio_t": 0, "latent_t": anchor_v, "latent_h": gh, "latent_w": gw})
    args["visual_condition_shapes"], args["ref_blocks"] = shapes, blocks
    return args

def register() -> None:
    """Entry point.  vLLM's plugin loader swallows exceptions, so a required patch that fails ends the process
    (exit 70) instead of leaving a half-patched rank that looks healthy.  ``H3_TP2_PLUGIN_SOFT_FAIL=1`` turns
    that into a logged error (debugging only)."""
    global _APPLIED
    if _APPLIED:
        return
    role = os.environ.get("H3_TP2_ROLE", "").strip().lower()
    if role not in ("head", "worker"):
        return
    try:
        _apply(role)
    except Exception:
        logger.critical("h3_tp2: required patch failed for role=%s", role, exc_info=True)
        if os.environ.get("H3_TP2_PLUGIN_SOFT_FAIL") == "1":
            return
        os._exit(70)
    _APPLIED = True


def _apply(role: str) -> None:
    world = int(os.environ.get("H3_TP2_WORLD_SIZE", "2"))

    from vllm_omni.platforms import current_omni_platform

    plat = current_omni_platform if isinstance(current_omni_platform, type) else type(current_omni_platform)

    # 1) device count (head only)
    if role == "head":
        _orig_count = plat.get_device_count

        def _device_count(cls=None, _orig=_orig_count, _world=world):  # noqa: ANN001
            real = int(_orig())
            return max(real, _world) if real >= 1 else real

        plat.get_device_count = classmethod(lambda cls: _device_count(cls))  # type: ignore[assignment]

    # 2) one GPU per host
    _orig_dev = plat.get_torch_device
    plat.get_torch_device = classmethod(lambda cls, local_rank=None: _orig_dev(0))  # type: ignore[assignment]

    # 3) MASTER_ADDR / MASTER_PORT from env
    from vllm_omni.diffusion.worker import diffusion_worker as _dw

    _orig_init = _dw.init_distributed_environment

    def _init_env(*args, **kwargs):  # noqa: ANN001
        addr = os.environ.get("H3_TP2_MASTER_ADDR")
        port = os.environ.get("H3_TP2_MASTER_PORT")
        if addr:
            os.environ["MASTER_ADDR"] = addr
        if port:
            os.environ["MASTER_PORT"] = port
        os.environ["LOCAL_RANK"] = "0"
        logger.info(
            "h3_tp2: init_distributed_environment rank=%s world=%s master=%s:%s",
            os.environ.get("RANK"),
            os.environ.get("WORLD_SIZE"),
            os.environ.get("MASTER_ADDR"),
            os.environ.get("MASTER_PORT"),
        )
        return _orig_init(*args, **kwargs)

    _dw.init_distributed_environment = _init_env


    # 4b) worker host: the diffusion worker is rank 1 but local GPU 0 (NVML handle, device bookkeeping)
    if role == "worker":
        _Worker = _dw.DiffusionWorker
        _orig_worker_init = _Worker.__init__

        def _worker_init(self, *a, **kw):  # noqa: ANN001
            # e1aa6eae passes (local_rank, rank, od_config) positionally, v0.30 passes them as keywords
            if "local_rank" in kw:
                kw["local_rank"] = 0
            elif a:
                a = (0,) + tuple(a[1:])
            return _orig_worker_init(self, *a, **kw)

        _Worker.__init__ = _worker_init

        # 4c) the async D2H thread builds its side stream on device gpu_id (= 1 here); give it a view with gpu_id 0.
        _orig_async = _dw.WorkerProc._async_output_loop

        class _LocalDeviceView:
            gpu_id = 0

            def __init__(self, inner):  # noqa: ANN001
                object.__setattr__(self, "_inner", inner)

            def __getattr__(self, name):  # noqa: ANN001
                return getattr(self._inner, name)

            def __setattr__(self, name, value):  # noqa: ANN001
                setattr(self._inner, name, value)

        def _async_loop(self):  # noqa: ANN001
            return _orig_async(_LocalDeviceView(self))

        _dw.WorkerProc._async_output_loop = _async_loop

    # 4) remote-reader result queue (worker host only)
    connect_ip = os.environ.get("H3_TP2_RESULT_CONNECT_IP")
    if role == "worker" and connect_ip:
        _MQ = _dw.MessageQueue

        class _RemoteResultQueue(_MQ):  # type: ignore[misc,valid-type]
            def __init__(self, n_reader, n_local_reader, local_reader_ranks=None, **kw):  # noqa: ANN001
                # The worker creates exactly one result queue for one reader (the head).
                kw.setdefault("connect_ip", connect_ip)
                super().__init__(n_reader=n_reader, n_local_reader=0, local_reader_ranks=[], **kw)

        _dw.MessageQueue = _RemoteResultQueue
    elif role == "worker":
        raise RuntimeError("H3_TP2_RESULT_CONNECT_IP is not set on the worker")

    # 5) optional: return the allocator's cached pages before VAE decode (after vllm-omni #7241).  No tensor
    #    arithmetic is touched; the same-seed framemd5 check in the README is the evidence that outputs are unchanged.
    #    vLLM-Omni >= 0.30 does this itself (``_release_stage_cache`` inside decode), so the patch is skipped there.
    if os.environ.get("H3_TP2_EMPTY_CACHE_BEFORE_DECODE", "0") == "1":
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pp
        import gc
        import torch

        _cls = _pp.MiniMaxH3Pipeline
        if hasattr(_cls, "_release_stage_cache"):
            logger.info("h3_tp2: upstream releases the stage cache before decode itself; empty_cache patch skipped")
        else:
            _orig_decode = _cls.decode

            def _decode(self, *a, **kw):  # noqa: ANN001
                gc.collect()
                torch.cuda.empty_cache()
                return _orig_decode(self, *a, **kw)

            _cls.decode = _decode
            logger.info("h3_tp2: empty_cache before decode enabled")

    # 6) uint8 output path + frame digest + phase timers (default on; H3_TP2_U8_OUTPUT=0 restores the stock FP32 path)
    if os.environ.get("H3_TP2_U8_OUTPUT", "1") == "1":
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pp
        from h3_tp2 import h3_output

        try:
            from vllm_omni.entrypoints.openai import video_api_utils as _vau
        except Exception:  # not every process serves HTTP; the encoder patch is only needed where it does
            _vau = None
        convert = not hasattr(_pp, "_prepare_minimax_h3_video_output")  # v0.30 quantises to uint8 on the GPU itself
        h3_output.install(_pp, _vau, convert_u8=convert)
        logger.info("h3_tp2: output path installed (uint8 conversion=%s, api_utils=%s)", convert, _vau is not None)

    # 6b) measurement only: time the Ulysses all-to-alls per diffuse phase
    if os.environ.get("H3_TP2_A2A_TIMING", "0") == "1":
        from h3_tp2 import a2a_timing

        a2a_timing.install()

    # 6c) opt-in: overlap the Ulysses all-to-alls with attention compute (identity verified by frame digest only)
    ov = os.environ.get("H3_TP2_A2A_OVERLAP", "").strip()
    if ov and int(ov) > 1:
        from h3_tp2 import ulysses_overlap

        ulysses_overlap.install(int(ov))

    # 8) worker (vLLM-Omni >= 0.30): request cancellation signals are POSIX shared-memory bytes created by the engine
    #    on the head; a rank on another host cannot open them.  Unreadable names become a stand-in that always votes
    #    "cancelled" in the MIN all-reduce of ``check_request_cancellation``, so the pair follows rank 0's real flag.
    if role == "worker":
        try:
            from vllm_omni.diffusion import cancellation as _canc
        except ImportError:
            _canc = None
        if _canc is not None and hasattr(_canc, "request_cancellation_scope"):
            from contextlib import contextmanager
            from multiprocessing.shared_memory import SharedMemory

            class _DeferToRank0:
                buf = bytearray(b"\x01")

                def close(self) -> None:
                    pass

            @contextmanager
            def _remote_scope(signal_names, *, enabled=True):  # noqa: ANN001
                signals = []
                try:
                    for name in signal_names if enabled else ():
                        if name is None:
                            signals.append(None)
                            continue
                        try:
                            signals.append(SharedMemory(name=name))
                        except FileNotFoundError:
                            signals.append(_DeferToRank0())
                    token = _canc._current_signals.set(tuple(signals) if enabled else None)
                    try:
                        yield
                    finally:
                        _canc._current_signals.reset(token)
                finally:
                    for sig in signals:
                        if sig is not None:
                            sig.close()

            _canc.request_cancellation_scope = _remote_scope
            try:
                from vllm_omni.diffusion.worker import diffusion_model_runner as _dmr

                if hasattr(_dmr, "request_cancellation_scope"):
                    _dmr.request_cancellation_scope = _remote_scope
            except ImportError:
                pass
            logger.info("h3_tp2: remote rank defers request cancellation to rank 0")

    # 9) experiments for the continuation mode (vLLM-Omni >= 0.30), both off by default and NOT output-preserving:
    #    H3_TP2_WINDOW_SEED_STEP=<n>: window k of a continuation request samples with seed + k*n instead of the same
    #    seed in every window (upstream re-creates the generators from the request seed per window, so equal-shaped
    #    windows restart identical noise streams).  H3_TP2_REF_SHORT_EDGE=<px>: reference images are resized to this
    #    short edge instead of upstream's 2048 (7296 VAE tokens for a 16:9 image join the sequence at every step).
    #    H3_TP2_WINDOW_TEMPORAL_OFFSET=<float|cap:<float>>: upstream shifts every media row of window k by
    #    start_frame*40/24 RoPE units while text and the reference image stay put (window 6 of a 60 s clip sits at
    #    +2125 units; a 15 s training clip spans 600).  "0" keeps every window on local time, "cap:425" clamps.
    #    H3_TP2_ANCHOR_FRAMES=<frames on the 17n+5 grid, e.g. 22>: every window after the first also gets the first
    #    window's opening frames as a Ref2VA video reference (kind "video", no audio rows, placed before the latent
    #    guide), so the detail level of the clip's start conditions every hand-over, not only the previous tail.
    #    H3_TP2_ANCHOR_SOURCE=tail is the same-shaped control: the previous window's last frames instead (what the
    #    guide already carries), to separate "extra reference rows + RoPE shift" from "opening texture".
    step = os.environ.get("H3_TP2_WINDOW_SEED_STEP", "").strip()
    toff = os.environ.get("H3_TP2_WINDOW_TEMPORAL_OFFSET", "").strip()
    anchor = os.environ.get("H3_TP2_ANCHOR_FRAMES", "").strip()
    if (step and int(step) != 0) or toff or anchor:
        import torch
        from vllm_omni.diffusion.models.minimax_h3 import continuation as _cont
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pp

        _orig_dc = _cont.diffuse_continuation
        _step = int(step) if step else 0
        _cap = float(toff[4:]) if toff.startswith("cap:") else None
        _fixed = None if (not toff or toff.startswith("cap:")) else float(toff)
        _anchor_v = _cont._video_t(int(anchor)) if anchor else 0
        if anchor and (int(anchor) < 5 or int(anchor) % 17 != 5):
            raise RuntimeError("H3_TP2_ANCHOR_FRAMES must be on the 17n+5 frame grid")
        _anchor_src = os.environ.get("H3_TP2_ANCHOR_SOURCE", "first").strip().lower()
        if _anchor_src not in ("first", "tail"):
            raise RuntimeError("H3_TP2_ANCHOR_SOURCE must be first (opening frames of window 1) or tail (last frames of the previous window)")

        def _dc(diffuse, kwargs, **kw):  # noqa: ANN001
            counter = {"i": 0, "anchor": None}
            if _anchor_v:
                first = min(int(kwargs["num_frames"]), int(kw.get("window_frames", 277)))
                if _anchor_v > _cont._video_t(first):
                    raise RuntimeError(f"H3_TP2_ANCHOR_FRAMES={anchor} exceeds the first window ({first} frames)")

            def _diffuse(**a):  # noqa: ANN001
                i = counter["i"]
                counter["i"] += 1
                a = dict(a)
                if _step and i and a.get("seed") is not None:
                    a["seed"] = int(a["seed"]) + i * _step
                if _fixed is not None and "temporal_offset" in a:
                    a["temporal_offset"] = _fixed
                elif _cap is not None and "temporal_offset" in a:
                    a["temporal_offset"] = min(float(a["temporal_offset"]), _cap)
                if _anchor_v and i and counter["anchor"] is not None:
                    insert_anchor_block(a, counter["anchor"], _anchor_v)
                logger.info("h3_tp2: continuation window %d seed=%s temporal_offset=%s anchor_rows=%s", i + 1, a.get("seed"),
                            a.get("temporal_offset"), 0 if not (_anchor_v and i) else int(counter["anchor"].shape[0]))
                video, audio = diffuse(**a)
                if _anchor_v and (i == 0 if _anchor_src == "first" else True):
                    src = video[:, :, :_anchor_v] if _anchor_src == "first" else video[:, :, -_anchor_v:]
                    counter["anchor"] = _cont.minimax_h3_patchify_video_latent(src, patch_size=(1, 2, 2))
                return video, audio

            return _orig_dc(_diffuse, kwargs, **kw)

        _cont.diffuse_continuation = _dc
        if getattr(_pp, "diffuse_continuation", None) is _orig_dc:
            _pp.diffuse_continuation = _dc
        logger.warning("h3_tp2: EXPERIMENT continuation seed step=%s temporal offset=%s anchor frames=%s source=%s (changes outputs)",
                       _step, toff or "upstream", anchor or "none", _anchor_src)

    # 9b) H3_TP2_FP8_GEMM=tensor32: scalar-scale FP8 GEMM with fp32 output + one exact Triton scale/bias/round pass
    #     instead of the per-row-scale CUTLASS epilogue (3.2x slower GEMM on sm121).  Same arithmetic, accumulation
    #     order differs (not bit-identical).  See h3_tp2/fp8_gemm.py and results/20261028-fp8-gemm-bench/.
    #     DEFAULT ON since 2026-10-29 (owner's call: a clean speedup with no quality loss): -19..-24 % per render,
    #     byte-identical output at the serving cell (FL2VA 5 s 1344x768), same-class samples elsewhere.
    #     H3_TP2_FP8_GEMM=stock restores vLLM's CUTLASS kernel.
    _gemm = os.environ.get("H3_TP2_FP8_GEMM", "tensor32").strip().lower()
    if _gemm == "tensor32":
        from . import fp8_gemm as _fp8

        _fp8.install()
        # 9b') H3_TP2_FP8_FUSE_SWIGLU=1: the MLP's fc1 keeps its fp32 product and one Triton kernel applies the
        #      epilogue and silu_and_mul (the bf16 [M, 2*ffn] intermediate is neither written nor read; a placeholder of
        #      that size is still allocated).  Matches two-pass + silu_and_mul on the tested inputs and build.
        #      "2" additionally quantises the SwiGLU output per token inside the kernel for fc2 (vLLM's compiled quant
        #      reproduced bit for bit), so the bf16 [M, ffn] activation is never written either.
        #      Level 1 is the default since 2026-10-29 (bit-identical to the two-pass path in nine renders, -5..-7 %
        #      denoising); "0" disables, "2" stays experimental (one of two renders differed from the plain digest).
        _fuse = os.environ.get("H3_TP2_FP8_FUSE_SWIGLU", "1").strip()
        if _fuse in ("1", "2"):
            from . import fp8_swiglu as _fsw

            _fsw.install(level=int(_fuse))

    # 9c) H3_TP2_VAE_STACK_TILES=1: the checkpoint's batched tile path for the video VAE (``stack_tiling``), DECODE ONLY
    #     (the flag is raised around decode_latent / decode_with_chunks and lowered again, so the reference-image encode
    #     keeps the stock per-tile path and the diffusion stays bit-identical), with at most H3_TP2_VAE_TILE_BATCH tiles
    #     per decoder call (default 4; the first try batched every local tile and peaked 32 GB higher).  Same arithmetic
    #     per tile; batching can change cuDNN/cuBLAS algorithm choice inside the decoder, so the frames may differ.
    if os.environ.get("H3_TP2_VAE_STACK_TILES", "0") == "1":
        import torch
        from vllm_omni.diffusion.models.minimax_h3 import vae as _vae

        _tile_batch = max(1, int(os.environ.get("H3_TP2_VAE_TILE_BATCH", "4")))
        _patched_models: set = set()

        def _chunked_run_tile_tasks(self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None):  # noqa: ANN001
            if not stack_tiling or not tile_indices or cls_agg is not None:
                return self._h3_tp2_orig_run_tile_tasks(tiles, tile_indices, forward_fn, stack_tiling, cls_agg)
            out = []
            for i in range(0, len(tile_indices), _tile_batch):
                out.extend(self._h3_tp2_orig_run_tile_tasks(tiles, tile_indices[i:i + _tile_batch], forward_fn, True, None))
            return out

        def _with_stack(method_name):  # noqa: ANN001
            orig = getattr(_vae.MiniMaxH3VideoVAE, method_name)

            def wrapped(self, *a, **kw):  # noqa: ANN001
                model = getattr(self, "model", None)
                if model is None or not hasattr(model, "stack_tiling"):
                    return orig(self, *a, **kw)
                cls = type(model)
                if cls not in _patched_models and hasattr(cls, "_run_tile_tasks"):
                    cls._h3_tp2_orig_run_tile_tasks = cls._run_tile_tasks
                    cls._run_tile_tasks = _chunked_run_tile_tasks
                    _patched_models.add(cls)
                    logger.warning("h3_tp2: video VAE batched tile DECODE on (<= %d tiles per decoder call); encode unchanged", _tile_batch)
                prev = model.stack_tiling
                model.stack_tiling = True
                try:
                    return orig(self, *a, **kw)
                finally:
                    model.stack_tiling = prev

            return wrapped

        for _m in ("decode_latent", "decode_with_chunks"):
            if hasattr(_vae.MiniMaxH3VideoVAE, _m):
                setattr(_vae.MiniMaxH3VideoVAE, _m, _with_stack(_m))

    # 9d) H3_TP2_CONT_PROMPTS=auto (default; "off" disables): a long_video continuation request without its own
    #     continuation_prompts gets one prompt per window (first: the prompt + "one single continuous take"; the rest:
    #     "the same continuous shot goes on ..." + the prompt), the wording measured on 2026-10-27/28 that keeps the shot
    #     and the subject continuous across the seams. Runs identically on every rank (a pure function of the request).
    if os.environ.get("H3_TP2_CONT_PROMPTS", "auto").strip().lower() != "off":
        from h3_tp2 import cont_prompts as _cp
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pp9d

        _cls9d = _pp9d.MiniMaxH3Pipeline
        if not getattr(_cls9d, "_h3_tp2_cont_prompts_patched", False):
            _orig_plc = _cls9d._prepare_local_conditioning
            _extra_words = os.environ.get("H3_TP2_CONT_PROMPTS_EXTRA", "")
            _rest_template = os.environ.get("H3_TP2_CONT_PROMPTS_REST", "")   # experiments: later windows' wording

            def _plc(self, raw_prompt, sampling, *a, **kw):  # noqa: ANN001
                try:
                    ea = getattr(sampling, "extra_args", None)
                    if (isinstance(ea, dict) and ea.get("long_video") and ea.get("long_video_mode", "continuation") == "continuation"
                            and not ea.get("continuation_prompts")):
                        dur = _cp.request_duration(ea)
                        base, _ = self._extract_prompt(raw_prompt)
                        if dur is not None and isinstance(base, str) and base.strip():
                            n = _cp.window_count(dur, int(ea.get("continuation_window_frames", 277)), int(ea.get("continuation_overlap_frames", 22)))
                            if n > 1:
                                ea["continuation_prompts"] = _cp.continuation_prompts(base, n, _extra_words, _rest_template)
                                logger.info("h3_tp2: continuation prompts generated for %d windows (%.1f s); window 2: %s", n, dur, ea["continuation_prompts"][1][:160])
                except Exception as e:  # noqa: BLE001  (never break a request over the prompt helper)
                    logger.warning("h3_tp2: continuation prompt helper skipped (%s)", e)
                return _orig_plc(self, raw_prompt, sampling, *a, **kw)

            _cls9d._prepare_local_conditioning = _plc
            _cls9d._h3_tp2_cont_prompts_patched = True
            logger.warning("h3_tp2: server-side per-window continuation prompts ON (H3_TP2_CONT_PROMPTS=off disables)")

    # 9e) continuation hand-over diagnostics (always on, one log line per window) and H3_TP2_CONT_MATCH=1: per-channel
    #     affine match of each new window's video latents to the previous window's tail over the hidden overlap
    #     (experiment against the exposure/texture step at the seams; audio untouched).
    try:
        import torch

        from h3_tp2 import cont_match as _cm
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pp9e

        if not getattr(_pp9e, "_h3_tp2_cont_match_patched", False):
            _orig_dc = _pp9e.diffuse_continuation
            _apply_match = os.environ.get("H3_TP2_CONT_MATCH", "0").strip() == "1"
            _xfade = os.environ.get("H3_TP2_CONT_XFADE", "0").strip() == "1"
            # H3_TP2_CONT_EDIT=1|2: the previous window's tail becomes pinned TARGET rows of the new window through
            # upstream's latent-edit path (mask 0 = keep the clean rows as the noise-augmented anchor at the condition
            # timestep, like FL2VA keyframes; restore 0 = restore them at the end), video and audio. "2" also drops
            # upstream's latent-guide condition block (the same frames would otherwise be in the sequence twice).
            _edit = os.environ.get("H3_TP2_CONT_EDIT", "0").strip()
            _edit = int(_edit) if _edit in ("1", "2") else 0
            # H3_TP2_CONT_RETRY=n: re-sample a window (another seed, up to n times) when its latent seam ratio exceeds
            # H3_TP2_CONT_SEAM_RATIO (default 2.0): the model may still cut right after the prefix (a sampling event).
            _retries = int(os.environ.get("H3_TP2_CONT_RETRY", "0") or 0)
            _max_ratio = float(os.environ.get("H3_TP2_CONT_SEAM_RATIO", "2.0") or 2.0)
            # H3_TP2_CONT_LOCK=first (+ H3_TP2_CONT_LOCK_STRENGTH, default 1.0): per-channel latent look-lock of every
            # later window's free frames to the first window's statistics, against the compounding drift.
            _lock = os.environ.get("H3_TP2_CONT_LOCK", "").strip().lower()
            _lock_strength = float(os.environ.get("H3_TP2_CONT_LOCK_STRENGTH", "1.0") or 1.0)

            if _xfade or _edit:
                # own copy of upstream's loop (continuation.py, v0.30.0) with the hand-over crossfaded over the overlap
                from vllm_omni.diffusion.models.minimax_h3 import continuation as _ct

                def _dc(diffuse, kwargs, *, window_frames, overlap_frames, text_conditioning=None):  # noqa: ANN001
                    diffuse = _cm.wrap_diffuse(diffuse, apply=_apply_match, overlap_v_hint=_ct._video_t(overlap_frames),
                                               retries=_retries, max_ratio=_max_ratio, lock=_lock, lock_strength=_lock_strength)
                    windows = _ct.plan_continuation_windows(kwargs["num_frames"], window_frames, overlap_frames)
                    if text_conditioning is not None and len(text_conditioning) != len(windows):
                        raise _ct.OmniClientError("MiniMax H3 requires one text conditioning per continuation window")
                    media_time_origin = max(item[0].shape[0] for item in text_conditioning) if text_conditioning else None
                    source_rows = kwargs.get("locked_audio_rows")
                    source = None if source_rows is None else source_rows.reshape(2, kwargs["audio_t"], 32)
                    video = audio = None
                    for index, window in enumerate(windows):
                        frames = window.end - window.start
                        args = {**kwargs, "num_frames": frames, "latent_t": _ct._video_t(frames),
                                "audio_t": window.audio_end - window.audio_start, "temporal_offset": window.start * (40.0 / 24.0)}
                        if text_conditioning is not None:
                            args["text_embeddings"], args["text_tags"] = text_conditioning[index]
                            args["media_time_origin"] = media_time_origin
                        if source is not None:
                            args["locked_audio_rows"] = source[:, window.audio_start:window.audio_end].reshape(-1, 32)
                        overlap_v = _ct._video_t(window.overlap) if window.overlap else 0
                        if video is not None and audio is not None:
                            guide_video = _ct.minimax_h3_patchify_video_latent(video[:, :, -overlap_v:], patch_size=(1, 2, 2))
                            guide_audio = _ct.minimax_h3_pack_audio_latent(audio[..., -window.overlap_audio_t:])
                            if _edit != 2:
                                for name, guide in (("visual_condition", guide_video), ("audio_condition", guide_audio)):
                                    existing = kwargs.get(name)
                                    args[name] = guide if existing is None else torch.cat((existing, guide), dim=0)
                                args["visual_condition_shapes"] = [*(kwargs.get("visual_condition_shapes") or []), (overlap_v, kwargs["latent_h"], kwargs["latent_w"])]
                                args["audio_condition_lengths"] = [*(kwargs.get("audio_condition_lengths") or []), window.overlap_audio_t]
                                args["ref_blocks"] = [*(kwargs.get("ref_blocks") or []), {"kind": "latent_guide", "latent_t": overlap_v,
                                                      "latent_h": kwargs["latent_h"], "latent_w": kwargs["latent_w"], "ref_audio_t": window.overlap_audio_t}]
                            if _edit:
                                dev = video.device
                                rows_per_frame = (video.shape[3] // 2) * (video.shape[4] // 2)
                                n_rows = args["latent_t"] * rows_per_frame
                                n_guide = guide_video.shape[0]
                                clean = torch.zeros((n_rows, guide_video.shape[1]), device=dev, dtype=torch.float32)
                                clean[:n_guide] = guide_video.float()
                                mask = torch.ones(n_rows, device=dev, dtype=torch.float32); mask[:n_guide] = 0.0
                                args["video_edit_clean_rows"], args["video_edit_mask_rows"], args["video_edit_restore_mask_rows"] = clean, mask, mask.clone()
                                ov_a, a_t = window.overlap_audio_t, args["audio_t"]
                                clean_a = torch.zeros((2 * a_t, guide_audio.shape[1]), device=dev, dtype=torch.float32)
                                mask_a = torch.ones(2 * a_t, device=dev, dtype=torch.float32)
                                for c in range(2):
                                    clean_a[c * a_t:c * a_t + ov_a] = guide_audio[c * ov_a:(c + 1) * ov_a].float()
                                    mask_a[c * a_t:c * a_t + ov_a] = 0.0
                                args["audio_edit_clean_rows"], args["audio_edit_mask_rows"], args["audio_edit_restore_mask_rows"] = clean_a, mask_a, mask_a.clone()
                                _ct.logger.info("h3_tp2 continuation edit prefix: %d of %d video rows and %d of %d audio rows pinned to the previous tail (guide block %s)",
                                                n_guide, n_rows, 2 * ov_a, 2 * a_t, "kept" if _edit == 1 else "dropped")
                        _ct.logger.info("MiniMax H3 continuation window %d/%d: frames [%d, %d), hidden overlap %d, temporal offset %.6f (h3_tp2 crossfade)",
                                        index + 1, len(windows), window.start, window.end, window.overlap, args["temporal_offset"])
                        sampled_video, sampled_audio = diffuse(**args)
                        if video is None:
                            video, audio = sampled_video, sampled_audio
                        else:
                            video = (_cm.crossfade_tail(video, sampled_video, overlap_v) if _xfade
                                     else torch.cat((video, sampled_video[:, :, overlap_v:]), dim=2))
                            audio = torch.cat((audio, sampled_audio[..., window.overlap_audio_t:]), dim=-1)
                    assert video is not None and audio is not None
                    if video.shape[2] != kwargs["latent_t"] or audio.shape[-1] != kwargs["audio_t"]:
                        raise RuntimeError("MiniMax H3 continuation produced inconsistent cumulative AV lengths")
                    return video, audio
            else:
                def _dc(diffuse, kwargs, **kw):  # noqa: ANN001
                    return _orig_dc(_cm.wrap_diffuse(diffuse, apply=_apply_match), kwargs, **kw)

            _pp9e.diffuse_continuation = _dc
            _pp9e._h3_tp2_cont_match_patched = True
            logger.warning("h3_tp2: continuation hand-over diagnostics ON; latent match %s; crossfade %s; edit prefix %s; seam retries %d (ratio > %.1f); look-lock %s (%.2f)",
                           "ON" if _apply_match else "off", "ON" if _xfade else "off", _edit or "off", _retries, _max_ratio, _lock or "off", _lock_strength)
    except Exception as e:  # noqa: BLE001
        logger.warning("h3_tp2: continuation hand-over hook not installed (%s)", e)

    edge = os.environ.get("H3_TP2_REF_SHORT_EDGE", "").strip()
    if edge:
        from vllm_omni.model_executor.models.minimax_h3 import preprocessing as _pre

        _pre.MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = int(edge)
        logger.warning("h3_tp2: EXPERIMENT reference image short edge %s px instead of 2048 (changes conditioning)", edge)

    # 10) measurement only: H3_TP2_PROFILE_STEP=<n> profiles the n-th DiT forward of this process (one denoising step
    #     on this rank) with torch.profiler and logs the top CUDA kernels; CUPTI on GB10 can be fragile, so off by default.
    pstep = os.environ.get("H3_TP2_PROFILE_STEP", "").strip()
    if pstep:
        import torch
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as _tr

        _cls = _tr.MiniMaxH3DiTModel
        _orig_fwd = _cls.forward
        _counter = {"n": 0, "done": False}
        _target = int(pstep)

        def _fwd(self, **kw):  # noqa: ANN001
            _counter["n"] += 1
            if _counter["done"] or _counter["n"] != _target:
                return _orig_fwd(self, **kw)
            _counter["done"] = True
            from torch.profiler import ProfilerActivity, profile

            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                out = _orig_fwd(self, **kw)
                torch.cuda.synchronize()
            try:
                table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=45, max_name_column_width=70)
                logger.info("h3_tp2 profile of DiT forward #%d (rank %s):\n%s", _target, os.environ.get("H3_TP2_ROLE"), table)
                path = os.path.expanduser(f"~/h3-tp2-logs/profile-{os.environ.get('H3_TP2_ROLE','x')}-{os.getpid()}.txt")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=200, max_name_column_width=90))
            except Exception:
                logger.exception("h3_tp2 profile report failed")
            return out

        _cls.forward = _fwd
        logger.warning("h3_tp2: PROFILING DiT forward #%d of this process (measurement only)", _target)

    # 7) experimental, off by default: lift the pipeline's 15 s duration cap.  The model is specified for 4-15 s;
    #    longer clips extrapolate the temporal RoPE beyond training.  Memory and time are the operator's problem.
    cap = os.environ.get("H3_TP2_MAX_OUTPUT_SECONDS", "").strip()
    if cap:
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pp

        _pp.MINIMAX_H3_MAX_OUTPUT_SECONDS = float(cap)  # e1aa6eae: both validation sites read this global
        try:  # v0.30: the limit comes from long_video.max_output_seconds (15 s unless long_video=true)
            from vllm_omni.model_executor.models.minimax_h3 import long_video as _lv
            from vllm_omni.model_executor.models.minimax_h3 import encoder_processing as _ep

            _orig_max = _lv.max_output_seconds

            def _max_output_seconds(extra, task):  # noqa: ANN001
                return max(_orig_max(extra, task), float(cap)) if not extra.get("long_video", False) else _orig_max(extra, task)

            _lv.max_output_seconds = _max_output_seconds
            if getattr(_ep, "max_output_seconds", None) is _orig_max:
                _ep.max_output_seconds = _max_output_seconds
        except ImportError:
            pass
        logger.warning("h3_tp2: EXPERIMENTAL duration cap raised to %s s (model spec is 15 s)", cap)

    logger.info("h3_tp2 plugin active: role=%s world=%s", role, world)
