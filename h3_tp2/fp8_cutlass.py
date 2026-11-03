"""CuTe-DSL sm120 FP8 GEMM with the scale epilogue in registers for qkv, out and fc1 (``H3_TP2_FP8_GEMM=cutlass``, opt-in).

The kernel (``h3_tp2/fp8_cutlass_kernel.py``, derived from CUTLASS's CuTe-DSL Blackwell-GeForce dense GEMM example)
computes y = bf16_rn(sa[m] * (sb[n] * acc)) with the fp32 accumulator in registers: the same operations, in the same
order, as ``fp8_gemm.scaled_mm_tensor32`` (scalar-scale cuBLASLt GEMM to fp32, then the Triton scale pass), without
the fp32 [M, N] product.  On GB10 its accumulator is bit-identical to cuBLASLt's on the four DiT shapes (same K order
and MMA rounding, no split-K), so the output is the two-pass output bit for bit; 157-168 TFLOPS with the epilogue
(``results/20261103-cutlass-sm120/``).

Routing (speed plan item 4, wiring plan steps 1-5 of that README): while a DiT block's forward runs, the dispatcher
sends the GEMMs whose weight is that block's ``attn.qkv_proj``, ``attn.out_proj`` or ``mlp.fc1`` weight to this kernel,
ahead of the stash hooks.  Their consumers then get an ordinary bf16 tensor: qkv and out go to the stock
``fused_qk_norm_rope`` / ``indexed_gate_rms_norm_scale_shift`` (``fp8_consumers`` drops those two sites at this level;
its fused consumers round at the same point, so either way is the same bits), fc1 goes to the MLP's own
``silu_and_mul`` (``fp8_swiglu``'s stash hook does not fire because the GEMM never reaches it).  fc2 stays on
cuBLASLt with the consumer-fused scale pass (``fp8_consumers`` site fc2): the DSL kernel is slower there.  Every other
FP8 GEMM keeps the two-pass path.

Safety: ``probe()`` imports ``cutlass`` (lazily: the package is only needed at this level), compiles the kernel once
(dynamic layouts: one compile serves every M, N, K) and checks it bit for bit against ``scaled_mm_tensor32`` on
integer inputs (exact accumulators in any order) and on random inputs with partial row and column tiles.  Any import,
compile or check failure logs one line and leaves the dispatcher on the two-pass path for the process.  A call the
kernel cannot take (fp16 output, a bias, misaligned pointers, M = 0) falls through to the ordinary dispatch.
"""
from __future__ import annotations

import functools
import logging
import os
import threading
import time
from typing import Any

import torch

from . import fp8_gemm

logger = logging.getLogger("vllm_omni.h3_tp2.fp8_cutlass")

SITES = ("qkv", "out", "fc1")
# tile M, N, K; MMA warps (M, N, K); L2 swizzle; raster along M; epilogue smem stages (unused by the direct store, kept
# so the stage count is the measured one); see results/20261103-cutlass-sm120/README.md and sweep4.jsonl
CFG = (128, 128, 64, (2, 2, 1), 8, False, 1)
_ALIGN = 16

_tls = threading.local()   # want: {(weight data_ptr, weight shape): site} of the DiT block whose forward is running
_state: dict[str, Any] = {"ok": False, "reason": "not probed", "compile_s": None, "gemm": None, "cpu_mirror": False}
_stats = {"dsl": 0, "fallthrough": 0}
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("h3_tp2 fp8 cutlass: " + msg, *args)


class DslGemm:
    """The compiled kernel and its launcher.  ``run`` binds five torch tensors per call (``from_dlpack``, dynamic
    layouts) and launches on the current stream.  Host cost on Spark_06: ~24 us for the five ``from_dlpack`` plus
    ~5 us for the compiled call, ~42 us per dispatcher call in all (torch._scaled_mm: ~8 us); below the ~50 us at which
    caching the cute tensors would pay (150 routed GEMMs per step: ~6 ms of host time against ~20 s of GPU time), and
    nothing is cached, so no tensor is kept alive here (``scripts/fp8_cutlass_check.py --parts host``)."""

    def __init__(self, cfg=CFG):  # noqa: ANN001
        import cuda.bindings.driver as cuda  # noqa: PLC0415  (lazy: cutlass is only needed at this level)
        import cutlass  # noqa: PLC0415
        import cutlass.cute as cute  # noqa: PLC0415
        from cutlass.cute.runtime import from_dlpack  # noqa: PLC0415

        from .fp8_cutlass_kernel import Sm120GemmKernel  # noqa: PLC0415

        bm, bn, bk, atom, swizzle, raster_m, epi = cfg
        self._cuda, self._cutlass, self._cute, self._from_dlpack = cuda, cutlass, cute, from_dlpack
        self.kernel = Sm120GemmKernel(cutlass.Float32, (bm, bn, bk), atom_layout=atom, swizzle_size=swizzle,
                                      raster_along_m=raster_m, epi_stage=epi, scaled=True)
        self.max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
        self.compiled = None
        self.compile_s = None
        self._streams: dict[int, Any] = {}

    def _t2(self, t: torch.Tensor, dtype):  # noqa: ANN001
        # [R, C] row-major -> cute (R, C, 1), C-major, dynamic layout (64-bit strides)
        raw = t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
        ct = self._from_dlpack(raw.unsqueeze(0).permute(1, 2, 0), assumed_align=_ALIGN)
        ct.element_type = dtype
        return ct.mark_layout_dynamic(leading_dim=1)

    def _t1(self, t: torch.Tensor):  # noqa: ANN001
        return self._from_dlpack(t, assumed_align=_ALIGN).mark_layout_dynamic()

    def run(self, a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """out[M, N] (bf16) = epilogue(a[M, K] @ w[N, K]^T); a, w row-major fp8; sa [M], sb [N] fp32 contiguous."""
        ca = self._t2(a, self._cutlass.Float8E4M3FN)
        cb = self._t2(w, self._cutlass.Float8E4M3FN)
        csb = self._t1(sb)
        cc = self._t2(out, self._cutlass.BFloat16)
        csa = self._t1(sa)
        sid = torch.cuda.current_stream(a.device).cuda_stream
        stream = self._streams.get(sid)
        if stream is None:
            stream = self._streams[sid] = self._cuda.CUstream(sid)
        if self.compiled is None:
            t0 = time.time()
            self.compiled = self._cute.compile(self.kernel, ca, cb, cc, csa, csb, self.max_clusters, stream)
            self.compile_s = time.time() - t0
        self.compiled(ca, cb, cc, csa, csb, stream)
        return out


def _aligned(*ts: torch.Tensor) -> bool:
    return all(t.data_ptr() % _ALIGN == 0 for t in ts)


def scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype,
              bias: torch.Tensor | None = None) -> torch.Tensor | None:
    """The DSL kernel for one dispatcher-eligible call (``b`` [K, N] with ``b.t()`` contiguous), or None when it cannot
    take the call (the caller continues with the ordinary dispatch)."""
    if out_dtype is not torch.bfloat16 or bias is not None or a.shape[0] == 0:
        return None
    M, N = a.shape[0], b.shape[1]
    sa, sb = fp8_gemm._vec_scales(scale_a, scale_b, M, N)
    if a.is_cuda and _state["ok"]:
        w = b.t()
        if not _aligned(a, w, sa, sb):
            return None
        y = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        return _state["gemm"].run(a, w, sa, sb, y)
    if not a.is_cuda and _state["cpu_mirror"]:   # tests: the kernel's arithmetic on an fp32 matmul
        return fp8_gemm.epilogue_torch(a.float() @ b.float(), sa, sb, None, torch.bfloat16)
    return None


def route(a, b, scale_a, scale_b, out_dtype, bias):  # noqa: ANN001,ANN201
    """fp8_gemm router (runs before the stash hooks): the DSL kernel for the running block's qkv / out / fc1 GEMMs."""
    want = getattr(_tls, "want", None)
    if not want:
        return None
    site = want.get((b.data_ptr(), tuple(b.shape)))
    if site is None:
        return None
    try:
        y = scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    except Exception as exc:  # noqa: BLE001  (never take the rank down: this call and all later ones go two-pass)
        logger.warning("h3_tp2 fp8 cutlass: launch failed (%s: %s); two-pass path for the rest of this process",
                       type(exc).__name__, exc)
        _state["ok"] = False
        y = None
    if y is None:
        _stats["fallthrough"] += 1
        _warn_once("ft-" + site, "%s GEMM M=%d N=%d not taken by the DSL kernel (dtype, bias or alignment); two-pass",
                   site, a.shape[0], b.shape[1])
        return None
    _stats["dsl"] += 1
    if _stats["dsl"] == 1:
        logger.info("h3_tp2 fp8 cutlass: first DSL GEMM (%s, M=%d N=%d K=%d)", site, a.shape[0], b.shape[1], a.shape[1])
    return y


# --------------------------------------------------------------------------------------------------------------------
# install: compile + bitwise self-check, then the block context that names the routed weights
# --------------------------------------------------------------------------------------------------------------------


def _check_case(gemm: DslGemm, a: torch.Tensor, w: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> bool:
    ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa, sb, torch.bfloat16)
    y = torch.full(ref.shape, float("nan"), device=a.device, dtype=torch.bfloat16)   # unwritten elements cannot pass
    s_a, s_b = fp8_gemm._vec_scales(sa, sb, a.shape[0], w.shape[0])
    gemm.run(a, w, s_a, s_b, y)
    return torch.equal(y.view(torch.int16), ref.view(torch.int16))


def selfcheck(gemm: DslGemm, device) -> list[str]:  # noqa: ANN001
    """Bitwise against ``scaled_mm_tensor32``: integer inputs (exact fp32 accumulators in any order: checks the
    epilogue, the edge predicates and the tile map) and random inputs on DiT-like N/K with partial row tiles (M = 200,
    4100: also checks that the accumulator matches cuBLASLt's).  Returns the failing cases."""
    E4M3 = torch.float8_e4m3fn
    g = torch.Generator(device=device).manual_seed(0)
    bad = []
    cases = [("int", 200, 400, 512), ("int", 333, 272, 1024),
             ("rand", 200, 5376, 7168), ("rand", 4100, 5376, 7168), ("rand", 4100, 21504, 5376),
             ("rand", 200, 28672, 5376)]
    for kind, M, N, K in cases:
        if kind == "int":
            a = torch.randint(-4, 5, (M, K), device=device, generator=g).to(E4M3)
            w = torch.randint(-2, 3, (N, K), device=device, generator=g).to(E4M3)
        else:
            a = (torch.randn(M, K, device=device, generator=g) * 64).clamp(-448, 448).to(E4M3)
            w = (torch.randn(N, K, device=device, generator=g) * 64).clamp(-448, 448).to(E4M3)
        sa = torch.exp(torch.empty(M, 1, device=device).uniform_(-9.0, -2.0, generator=g))
        sb = torch.exp(torch.empty(N, device=device).uniform_(-9.5, -7.5, generator=g))
        if not _check_case(gemm, a, w, sa, sb):
            bad.append(f"{kind} M={M} N={N} K={K}")
        del a, w, sa, sb
    torch.cuda.synchronize(device)
    return bad


def probe(device="cuda") -> bool:  # noqa: ANN001
    """Import, compile and self-check the kernel once per process; on any failure the level stays unavailable."""
    _state.update(ok=False, gemm=None)
    if not fp8_gemm._state.get("triton"):
        _state["reason"] = "the Triton scale pass (the reference) is not usable"
        return False
    try:
        gemm = DslGemm()
        t0 = time.time()
        bad = selfcheck(gemm, device)
        _state["compile_s"] = gemm.compile_s
        if bad:
            _state["reason"] = "not bit-identical to two-pass on " + "; ".join(bad)
            return False
        _state.update(ok=True, gemm=gemm, reason=f"compiled in {gemm.compile_s:.1f} s, self-check "
                      f"{time.time() - t0 - (gemm.compile_s or 0):.1f} s")
        return True
    except Exception as exc:  # noqa: BLE001
        _state["reason"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
        return False


def block_want(block, sites=SITES) -> dict:  # noqa: ANN001
    """{(weight data_ptr, weight shape): site} for this block's FP8 qkv / out / fc1 weights."""
    attn = getattr(block, "attn", None)
    mlp = getattr(block, "mlp", None)
    layers = {"qkv": getattr(attn, "qkv_proj", None), "out": getattr(attn, "out_proj", None),
              "fc1": getattr(mlp, "fc1", None)}
    want = {}
    for site in sites:
        w = getattr(layers.get(site), "weight", None)
        if isinstance(w, torch.Tensor) and w.dtype == torch.float8_e4m3fn and w.dim() == 2:
            want[(w.data_ptr(), tuple(w.shape))] = site
    return want


def make_block_forward(orig_forward, sites=SITES):  # noqa: ANN001,ANN201
    @functools.wraps(orig_forward)       # keeps __wrapped__, so source fingerprints still see upstream's forward
    def forward(self, *args, **kwargs):  # noqa: ANN001,ANN202
        try:
            if torch.compiler.is_compiling():
                return orig_forward(self, *args, **kwargs)
        except AttributeError:
            pass
        prev = getattr(_tls, "want", None)
        _tls.want = block_want(self, sites)
        try:
            return orig_forward(self, *args, **kwargs)
        finally:
            _tls.want = prev

    forward._h3_tp2_cutlass = True
    return forward


def install_sites(tr=None, sites=SITES) -> bool:  # noqa: ANN001
    """Wrap ``MiniMaxH3DiTBlock.forward`` of the transformer module ``tr`` (default: vLLM-Omni's MiniMax H3) so the
    router knows the running block's weights."""
    if tr is None:
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as tr  # noqa: PLC0415
    cls = tr.MiniMaxH3DiTBlock
    if getattr(cls.forward, "_h3_tp2_cutlass", False):
        return True
    cls.forward = make_block_forward(cls.forward, sites)
    return True


def stats() -> dict[str, int]:
    return dict(_stats)


if os.environ.get("H3_TP2_FP8_CUTLASS_CPU_MIRROR") == "1":   # tests only: route CPU tensors through the eager mirror
    _state["cpu_mirror"] = True
