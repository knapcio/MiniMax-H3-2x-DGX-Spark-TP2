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
    if os.environ.get("H3_TP2_FP8_GEMM", "").strip().lower() == "tensor32":
        from . import fp8_gemm as _fp8

        _fp8.install()

    # 9c) H3_TP2_VAE_STACK_TILES=1: the checkpoint's own batched tile path for the video VAE (``stack_tiling``, off by
    #     default and not exposed by vLLM-Omni): each rank decodes its tiles as one batch instead of one call per tile.
    #     Same arithmetic per tile; batching can change cuDNN/cuBLAS algorithm choice, so the frame digest is checked.
    #     (Pointed at by NVlabs' sol-engine GB10 port, which measures 1.47x on the decode with the same change.)
    if os.environ.get("H3_TP2_VAE_STACK_TILES", "0") == "1":
        from vllm_omni.diffusion.models.minimax_h3 import vae as _vae

        _orig_vae_init = _vae.MiniMaxH3VideoVAE.__init__

        def _vae_init(self, *a, **kw):  # noqa: ANN001
            _orig_vae_init(self, *a, **kw)
            model = getattr(self, "model", None)
            if model is not None and hasattr(model, "stack_tiling"):
                model.stack_tiling = True
                logger.warning("h3_tp2: video VAE stack_tiling ON (batched tile decode; frame digest to be checked)")

        _vae.MiniMaxH3VideoVAE.__init__ = _vae_init

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
