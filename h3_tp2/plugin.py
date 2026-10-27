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

        def _worker_init(self, local_rank, rank, od_config, *a, **kw):  # noqa: ANN001
            return _orig_worker_init(self, 0, rank, od_config, *a, **kw)

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
    if os.environ.get("H3_TP2_EMPTY_CACHE_BEFORE_DECODE", "0") == "1":
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pp
        import gc
        import torch

        _cls = _pp.MiniMaxH3Pipeline
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
        h3_output.install(_pp, _vau)
        logger.info("h3_tp2: uint8 output path installed (api_utils=%s)", _vau is not None)

    # 6b) measurement only: time the Ulysses all-to-alls per diffuse phase
    if os.environ.get("H3_TP2_A2A_TIMING", "0") == "1":
        from h3_tp2 import a2a_timing

        a2a_timing.install()

    # 6c) opt-in: overlap the Ulysses all-to-alls with attention compute (bit-identical by construction, measured)
    ov = os.environ.get("H3_TP2_A2A_OVERLAP", "").strip()
    if ov and int(ov) > 1:
        from h3_tp2 import ulysses_overlap

        ulysses_overlap.install(int(ov))

    # 7) experimental, off by default: lift the pipeline's 15 s duration cap.  The model is specified for 4-15 s;
    #    longer clips extrapolate the temporal RoPE beyond training.  Memory and time are the operator's problem.
    cap = os.environ.get("H3_TP2_MAX_OUTPUT_SECONDS", "").strip()
    if cap:
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as _pp

        _pp.MINIMAX_H3_MAX_OUTPUT_SECONDS = float(cap)
        logger.warning("h3_tp2: EXPERIMENTAL duration cap raised to %s s (model spec is 15 s)", cap)

    logger.info("h3_tp2 plugin active: role=%s world=%s", role, world)
