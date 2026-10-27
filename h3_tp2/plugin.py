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
