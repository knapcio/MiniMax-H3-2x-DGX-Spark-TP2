"""Ulysses exact exchange without its last copy: the out-projection's per-token FP8 quant reads the reverse all-to-all's
receive buffer (``H3_TP2_ULYSSES_EXACT=2``; ``1`` keeps the copy).

With ``H3_TP2_ULYSSES_EXACT=1`` (``h3_tp2/ulysses_exact.py``) one copy of the Ulysses exchange is left per attention
layer and rank: the reverse unpack (``comm.py:93-96``, copy 5 of that module's table), receive [P, M, hl, d] ->
[1, M, H, d], 14336*M bytes written and read again (270 MB each way at M = 18880).  It exists only so that the
out-projection gets a row-major [M, H*d] input; the first and only thing that reads that tensor is the out-projection's
dynamic per-token FP8 quant (``MiniMaxH3Attention.forward`` -> ``out.reshape(total, H*d)`` -> ``self.out_proj`` ->
``RowParallelLinear.forward`` -> ``Fp8PerTensorOnlineLinearMethod.apply`` -> ``FP8ScaledMMLinearKernel.apply_weights``
-> ``x.view(-1, K)`` -> ``self.quant_fp8(x_2d, None, None)``, vLLM ``QuantFP8``, per token, dynamic).  Row m of that
input is the concatenation over the sources p of recv[p, m] (hl*d contiguous elements each), so the quant can read the
receive buffer directly: one program per row loads the P segments, takes the row's abs-max, and writes the FP8 row
and its scale exactly where the stock quant writes them.  The copy and the [1, M, H, d] tensor are never made.

Exactness.  In serving ``QuantFP8`` runs its ``forward_native`` compiled by Inductor (custom ops "none", the diffusion
VllmConfig: ``CustomOp.maybe_compile`` with ``dynamic=True``; the generated kernel is kept in
``results/20261104-ulysses-copy5/``).  Its arithmetic per row is: amax = max |x| over the bf16 row (in fp32, a max,
so independent of order and of where the elements live), scale = max(amax / 448, 1/(448*512)) in fp32 (the divisor is
a runtime fp32 448), r = 1 / scale, q = fp8_e4m3(clamp(x * r, -448, 448)).  ``_quant_rows_kernel`` performs these
operations with the same Triton expressions as the Inductor kernel (``/`` for both divisions, inductor's NaN-propagating
``triton_helpers.maximum`` / ``minimum`` / ``max2``, the same constants, the default fp8 cast), so the same compiler
emits the same instructions; only the addresses differ.  ``tests/test_ulysses_outq_gpu.py`` checks it bit for bit
against the real compiled ``QuantFP8`` on random, attention-output-like and directed rows and on every bf16 value as
the row maximum (all scales that can occur), and the first eligible call of every process compares the two on the full
real tensor (``_selfcheck``): a mismatch, a compile error or a launch error disables the path for the process and that
call returns the stock result.

Safety.  The copy is skipped only for a ``MiniMaxH3Attention`` whose ``out_proj`` was checked (``prepare``): plain
``RowParallelLinear`` (no LoRA wrapper), tp 1, input parallel, the online per-tensor FP8 method with a CUTLASS-family
``FP8ScaledMMLinearKernel``, a dynamic per-token ``QuantFP8`` without padding, no static input scale, no batch-invariant
mode, and the reviewed source of every function on the path (``EXPECTED``).  The attention returns a stride-0
placeholder of the stock shape; that layer's ``quant_fp8`` is replaced (on the kernel instance) by ``_OutQuant``,
which recognises the placeholder by pointer and shape and quantises from the receive buffer.  A placeholder that
reaches any other quant, or a deferred receive that the forward never consumed, raises instead of computing on garbage.

With ``H3_TP2_FP8_QUANT_FUSE`` (``h3_tp2/fp8_quant_fuse.py``) its qkv / fc1 / fc2 sites combine with this mode; its out
site (the fused unpack + quant of mode 1) is skipped, since there is no unpack copy here.  The two placeholders are
kept apart: one of this module reaching the fuse's ``QuantFP8.forward`` wrapper raises there (``is_placeholder``), and
a pending stash placeholder of the fuse reaching ``_OutQuant`` raises here.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Any

import torch

logger = logging.getLogger("vllm_omni.h3_tp2.ulysses_outq")

FP8_MAX = 448.0                     # vLLM get_fp8_min_max() for float8_e4m3fn on CUDA
FP8_MIN = -448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)  # QuantFP8 _FP8_MIN_SCALING_FACTOR
_WARPS = 8

# sha256 (first 16 hex) of the stripped source lines (step_cache.fingerprint) of every function between the attention
# output and the quant, vLLM 0.30.0 / vLLM-Omni v0.30.0 a8576ccb7 as installed in ~/h3-long/.venv on Spark_06 (2026-11-04)
EXPECTED = {
    "attn_forward": "b2a25e0b45bea674",   # minimax_h3_transformer.py:630 MiniMaxH3Attention.forward
    "row_forward": "693d179dcadc13ff",    # linear.py:1757 RowParallelLinear.forward
    "online_apply": "5cbe04cd0e39db60",   # quantization/online/fp8.py:228 Fp8PerTensorOnlineLinearMethod.apply
    "apply_weights": "54ad02f4f844454d",  # kernels/linear/scaled_mm/ScaledMMLinearKernel.py:135 (FP8 apply_weights)
    "quant_native": "7a11e52df04d77be",   # input_quant_fp8.py:181 QuantFP8.forward_native (the arithmetic mirrored)
}

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers as _th
    from torch._inductor.runtime.triton_helpers import math as tl_math   # the module Inductor's kernels use for abs

    @triton.jit
    def _quant_rows_kernel(x_ptr, q_ptr, s_ptr, fp8_max, stride_p, stride_m,
                           P: tl.constexpr, SEG: tl.constexpr, P_BLOCK: tl.constexpr, BLOCK: tl.constexpr,
                           MIN_SCALE_C: tl.constexpr, FP8_MIN_C: tl.constexpr, FP8_MAX_C: tl.constexpr):
        # One row of the out-projection input = P segments of SEG bf16 at x + p*stride_p + row*stride_m (the receive
        # buffer [P, M, SEG]: stride_p = M*SEG, stride_m = SEG; a contiguous [M, P*SEG] row: SEG and P*SEG).
        # Arithmetic = the Inductor kernel of QuantFP8.forward_native (results/20261104-ulysses-copy5/quant_inductor.py)
        row = tl.program_id(0).to(tl.int64)
        ps = tl.arange(0, P_BLOCK)
        cols = tl.arange(0, BLOCK)
        mask = (ps[:, None] < P) & (cols[None, :] < SEG)
        x = tl.load(x_ptr + row * stride_m + ps[:, None].to(tl.int64) * stride_p + cols[None, :], mask=mask,
                    other=0.0).to(tl.float32)
        amax = _th.max2(_th.max2(tl_math.abs(x), 1), 0)          # aten.max over |x|: exact, NaN-propagating
        amax = tl.broadcast_to(amax, [1])
        scale = _th.maximum(amax / fp8_max, tl.full([1], MIN_SCALE_C, tl.float32))
        inv = tl.full([1], 1.0, tl.float32) / scale
        y = x * inv[:, None]
        y = _th.minimum(_th.maximum(y, tl.full([1, 1], FP8_MIN_C, tl.float32)), tl.full([1, 1], FP8_MAX_C, tl.float32))
        tl.store(q_ptr + row * (P * SEG) + ps[:, None] * SEG + cols[None, :], y.to(tl.float8e4nv), mask=mask)
        tl.store(s_ptr + row + tl.arange(0, 1), scale)

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    _HAVE_TRITON = False


# ---------------------------------------------------------------------------------------------------------------
# the quant (GPU kernel + eager mirror with the kernel's addressing)
# ---------------------------------------------------------------------------------------------------------------


def _pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def quant_rows(base: torch.Tensor, m: int, p: int, seg: int, stride_p: int, stride_m: int, warps: int = _WARPS):
    """Per-token FP8 quant of M rows whose row m is the P segments ``base[p*stride_p + m*stride_m : +seg]`` (element
    offsets into ``base``'s storage view, ``base`` bf16 and contiguous from its data pointer).  Returns (q [M, P*seg]
    fp8 e4m3fn contiguous, scale [M, 1] fp32): the shapes, strides and dtypes of ``QuantFP8``'s per-token result."""
    q = torch.empty((m, p * seg), dtype=torch.float8_e4m3fn, device=base.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=base.device)
    if m == 0:
        return q, s
    if base.is_cuda and _HAVE_TRITON:
        _quant_rows_kernel[(m,)](base, q, s, FP8_MAX, stride_p, stride_m, P=p, SEG=seg, P_BLOCK=_pow2(p),
                                 BLOCK=_pow2(seg), MIN_SCALE_C=MIN_SCALE, FP8_MIN_C=FP8_MIN, FP8_MAX_C=FP8_MAX,
                                 num_warps=warps)
        return q, s
    x = base.reshape(-1).as_strided((m, p, seg), (stride_m, stride_p, 1)).reshape(m, p * seg)
    return quant_reference_eager(x, out=(q, s))


def quant_reference_eager(x: torch.Tensor, out=None):  # noqa: ANN001
    """``QuantFP8.forward_native`` (dynamic, per token, no padding) in eager torch, for the CPU tests and the mirror."""
    x_max = x.abs().max(dim=-1)[0].unsqueeze(-1).to(torch.float32)
    scale = (x_max / FP8_MAX).clamp(min=MIN_SCALE)
    q = (x.to(torch.float32) * scale.to(torch.float32).reciprocal()).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)
    if out is not None:
        out[0].copy_(q)
        out[1].copy_(scale)
        return out
    return q, scale


def quant_from_recv(recv: torch.Tensor, warps: int = _WARPS):
    """recv [P, M, hl, d] contiguous (the reverse all-to-all's receive buffer) -> the quant of the [M, P*hl*d] tensor
    that stock's unpack would have produced (row m = recv[0, m], recv[1, m], ...)."""
    p, m, hl, d = recv.shape
    seg = hl * d
    return quant_rows(recv, m, p, seg, m * seg, seg, warps)


def unpack_rows(recv: torch.Tensor) -> torch.Tensor:
    """Stock's (and ulysses_exact's) copy 5: [P, M, hl, d] -> contiguous [M, P*hl*d]."""
    p, m, hl, d = recv.shape
    out = torch.empty((1, m, p * hl, d), dtype=recv.dtype, device=recv.device)
    out[0].view(m, p, hl, d).copy_(recv.transpose(0, 1))
    return out.view(m, p * hl * d)


# ---------------------------------------------------------------------------------------------------------------
# deferral: placeholder out of the exchange, quant from the receive buffer in the out-projection
# ---------------------------------------------------------------------------------------------------------------

_tls = threading.local()   # owner: (MiniMaxH3Attention, rows) armed for one deferral; pending: _Deferred
_state: dict[str, Any] = {"disabled": None, "checked": False, "sources": None}
_stats = {"deferred": 0, "quant_recv": 0, "selfcheck_pass": 0, "selfcheck_fail": 0, "not_prepared": 0}
_warned: set[str] = set()


def _count(kind: str) -> None:
    _stats[kind] += 1
    if _stats[kind] == 1:
        (logger.warning if kind in ("selfcheck_fail", "not_prepared") else logger.info)(
            "h3_tp2 ulysses exact copy5: first %s", kind)
    if kind == "quant_recv" and _stats[kind] % 2000 == 0:
        logger.info("h3_tp2 ulysses exact copy5: calls so far %s", " ".join(f"{k}={v}" for k, v in _stats.items() if v))


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("h3_tp2 ulysses exact copy5: " + msg, *args)


def stats() -> dict[str, int]:
    return dict(_stats)


def _placeholder(shape, like: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    return like.new_empty(1).as_strided(tuple(shape), (0,) * len(shape))


def _is_placeholder(t: torch.Tensor) -> bool:
    return t.numel() > 1 and all(s == 0 for s in t.stride())


def is_placeholder(t: torch.Tensor) -> bool:
    """True for a placeholder made by ``defer`` or a view of it: stride 0 everywhere and either the pending deferral's
    pointer or a one-element storage (for fp8_quant_fuse's guard: it must never reach another quant)."""
    if not _is_placeholder(t):
        return False
    pend = getattr(_tls, "pending", None)
    if pend is not None and t.data_ptr() == pend.ph.data_ptr():
        return True
    return t.untyped_storage().nbytes() <= t.element_size()


def _fuse_placeholder(x: torch.Tensor) -> bool:
    """True when ``x`` is H3_TP2_FP8_QUANT_FUSE's pending stash placeholder (an uninitialised bf16 tensor)."""
    fq = sys.modules.get(__package__ + ".fp8_quant_fuse")
    return fq is not None and fq.pending_placeholder(x)


class _Deferred:
    __slots__ = ("recv", "ph", "m", "k")

    def __init__(self, recv: torch.Tensor, ph: torch.Tensor):
        self.recv, self.ph = recv, ph
        p, m, hl, d = recv.shape
        self.m, self.k = m, p * hl * d

    def matches(self, x: torch.Tensor) -> bool:
        return x.data_ptr() == self.ph.data_ptr() and tuple(x.shape) == (self.m, self.k) and _is_placeholder(x)

    def quantize(self, orig):  # noqa: ANN001,ANN201
        """(q, scale) for the out-projection: from the receive buffer, after the once-per-process self-check."""
        if _state["disabled"] is None and not _state["checked"]:
            return _selfcheck(self, orig)
        if _state["disabled"] is not None:      # disabled after this deferral was made: the stock path, exactly
            return orig(unpack_rows(self.recv), None, None)
        _count("quant_recv")
        return quant_from_recv(self.recv)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def _selfcheck(pend: _Deferred, orig):  # noqa: ANN001,ANN202
    """The first deferral of the process: stock (copy 5 + the layer's own QuantFP8) and the receive-buffer quant on the
    full real tensor, compared bit for bit (FP8 values, scales, shapes, strides).  Returns the stock result."""
    _state["checked"] = True
    stock = orig(unpack_rows(pend.recv), None, None)
    try:
        got = quant_from_recv(pend.recv)
        ok = all(a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))
                 for a, b in zip(stock, got))
        why = "" if ok else "FP8 values or scales differ"
    except Exception as exc:  # noqa: BLE001
        ok, why = False, f"raised {exc!r}"
    if ok:
        _count("selfcheck_pass")
        logger.warning("h3_tp2 ulysses exact copy5: self-check M=%d K=%d bit-identical to copy 5 + QuantFP8; "
                       "the out-projection quant reads the receive buffer", pend.m, pend.k)
    else:
        _count("selfcheck_fail")
        _state["disabled"] = "self-check: " + why
        logger.warning("h3_tp2 ulysses exact copy5: self-check M=%d K=%d NOT bit-identical (%s); copy 5 restored for "
                       "this process", pend.m, pend.k, why)
    return stock


def take(x: torch.Tensor) -> "_Deferred | None":
    pend = getattr(_tls, "pending", None)
    if pend is not None and pend.matches(x):
        _tls.pending = None
        return pend
    if (pend is not None and x.data_ptr() == pend.ph.data_ptr()) or (pend is None and _is_placeholder(x)):
        raise RuntimeError("h3_tp2 ulysses exact copy5: a placeholder reached an out-projection quant without its "
                           "receive buffer (or in another shape)")
    return None


class _OutQuant:
    """Stands in for one out-projection kernel's ``quant_fp8``: the deferred placeholder is quantised from the receive
    buffer, every other input goes to the original ``QuantFP8`` unchanged."""

    _h3_tp2_outq = True

    def __init__(self, orig):  # noqa: ANN001
        self._orig = orig

    def __getattr__(self, name: str):  # noqa: ANN204
        if name == "_orig":
            raise AttributeError(name)
        return getattr(self._orig, name)

    def __call__(self, x, scale=None, scale_ub=None, *args, **kwargs):  # noqa: ANN001,ANN204
        if isinstance(x, torch.Tensor) and _fuse_placeholder(x):
            raise RuntimeError("h3_tp2 ulysses exact copy5: an H3_TP2_FP8_QUANT_FUSE placeholder (uninitialised) "
                               "reached the out-projection quant")
        pend = take(x) if isinstance(x, torch.Tensor) else None
        if pend is None:
            return self._orig(x, scale, scale_ub, *args, **kwargs)
        if scale is not None or scale_ub is not None or args or kwargs:
            raise RuntimeError("h3_tp2 ulysses exact copy5: unexpected static scale / bound for a deferred quant")
        return pend.quantize(self._orig)


# ---------------------------------------------------------------------------------------------------------------
# eligibility of an attention's out-projection
# ---------------------------------------------------------------------------------------------------------------


def source_fingerprints(attn_forward) -> dict[str, str | None]:  # noqa: ANN001
    """The reviewed upstream functions' fingerprints (``attn_forward`` = MiniMaxH3Attention.forward before wrapping)."""
    from .step_cache import fingerprint

    fps: dict[str, str | None] = {"attn_forward": fingerprint(attn_forward)}
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearKernel
        from vllm.model_executor.layers.linear import RowParallelLinear
        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
        from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
    except ImportError:
        return fps
    fps.update(row_forward=fingerprint(RowParallelLinear.forward),
               online_apply=fingerprint(Fp8PerTensorOnlineLinearMethod.apply),
               apply_weights=fingerprint(FP8ScaledMMLinearKernel.apply_weights),
               quant_native=fingerprint(QuantFP8.forward_native))
    return fps


def check_sources(attn_forward) -> str | None:  # noqa: ANN001
    """None when every function on the path has the reviewed source, else the reason (stored for ``prepare``)."""
    fps = source_fingerprints(attn_forward)
    bad = [f"{n} {fps.get(n)} (reviewed {v})" for n, v in EXPECTED.items() if fps.get(n) != v]
    _state["sources"] = ("upstream source changed: " + ", ".join(bad)) if bad else None
    if bad:
        _warn_once("sources", "%s; copy 5 kept", _state["sources"])
    return _state["sources"]


def layer_mismatch(owner) -> str | None:  # noqa: ANN001
    """None when ``owner.out_proj`` reaches ``kernel.quant_fp8(x.view(-1, K), None, None)`` with nothing in between
    that reads x (the reviewed path), else the reason."""
    try:
        import vllm.envs as envs
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearKernel
        from vllm.model_executor.layers.linear import RowParallelLinear
        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
        from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
    except ImportError as exc:
        return f"vLLM import: {exc}"
    layer = getattr(owner, "out_proj", None)
    if type(layer) is not RowParallelLinear:
        return f"out_proj is {type(layer).__name__}, not a plain RowParallelLinear"
    if getattr(layer, "tp_size", 1) != 1 or not getattr(layer, "input_is_parallel", False):
        return "out_proj tp_size != 1 or input not parallel"
    k = int(owner.num_heads) * int(owner.head_dim)
    if int(getattr(layer, "input_size_per_partition", -1)) != k:
        return "out_proj input width != heads * head_dim"
    qm = getattr(layer, "quant_method", None)
    if type(qm) is not Fp8PerTensorOnlineLinearMethod:
        return f"quant method {type(qm).__name__}"
    if envs.VLLM_BATCH_INVARIANT:
        return "VLLM_BATCH_INVARIANT"
    kernel = getattr(qm, "fp8_linear", None)
    if not isinstance(kernel, FP8ScaledMMLinearKernel) or type(kernel).apply_weights is not FP8ScaledMMLinearKernel.apply_weights:
        return f"linear kernel {type(kernel).__name__}"
    quant = kernel.quant_fp8
    if isinstance(quant, _OutQuant):
        quant = quant._orig
    if not isinstance(quant, QuantFP8) or quant.static or quant.is_group_quant or not quant.use_per_token_if_dynamic \
            or quant.num_token_padding is not None:
        return "activation quant is not dynamic per-token QuantFP8 without padding"
    if getattr(kernel, "fp8_dtype", None) is not torch.float8_e4m3fn:
        return "fp8 dtype"
    _, _, x_s, x_s_ub = kernel._get_layer_params(layer)
    if x_s is not None or x_s_ub is not None:
        return "static input scale or scale bound"
    return None


def prepare(owner) -> bool:  # noqa: ANN001
    """Once per attention: True when its out-projection's quant can take a deferred receive buffer (and then wraps
    that layer's ``quant_fp8``)."""
    layer = getattr(owner, "out_proj", None)
    if layer is None:
        return False
    ok = getattr(layer, "_h3_tp2_outq", None)
    if ok is None:
        reason = _state["sources"] or layer_mismatch(owner)
        ok = reason is None
        if ok:
            kernel = layer.quant_method.fp8_linear
            if not isinstance(kernel.quant_fp8, _OutQuant):
                kernel.quant_fp8 = _OutQuant(kernel.quant_fp8)
        else:
            _warn_once("layer-" + reason, "out-projection not eligible (%s); copy 5 kept there", reason)
        layer._h3_tp2_outq = ok
    return bool(ok)


def arm(owner, rows: int) -> tuple:  # noqa: ANN001
    """Called by the armed ``MiniMaxH3Attention.forward``: this forward's reverse exchange may defer its unpack."""
    prev = (getattr(_tls, "owner", None), getattr(_tls, "pending", None))
    _tls.owner = (owner, rows) if _state["disabled"] is None and prepare(owner) else None
    _tls.pending = None
    return prev


def disarm(prev: tuple, check: bool) -> None:
    """End of the armed forward.  ``check``: the forward returned normally, so a deferral must have been consumed."""
    pend = getattr(_tls, "pending", None)
    _tls.owner, _tls.pending = prev
    if check and pend is not None:
        raise RuntimeError("h3_tp2 ulysses exact copy5: the attention returned without its out-projection quantising "
                           "the deferred receive buffer")


def defer(recv: torch.Tensor, heads: int) -> torch.Tensor | None:
    """From the reverse exchange: a stride-0 placeholder [1, M, H, d] standing for unpack(recv) when this forward was
    armed for it (rows match, path enabled), else None (the caller makes copy 5)."""
    arm_ = getattr(_tls, "owner", None)
    p, m, hl, d = recv.shape
    if arm_ is None or arm_[1] != m or p * hl != heads or _state["disabled"] is not None or not recv.is_contiguous() \
            or recv.dtype is not torch.bfloat16:
        if arm_ is not None:
            _count("not_prepared")
        return None
    _tls.owner = None
    ph = _placeholder((1, m, heads, d), recv)
    _tls.pending = _Deferred(recv, ph)
    _count("deferred")
    return ph
