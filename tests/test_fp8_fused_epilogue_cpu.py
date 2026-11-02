"""CPU checks for H3_TP2_FP8_GEMM=fused (speed-plan item 4): the fused kernels' epilogue arithmetic equals the two-pass
reference, and a tile-by-tile emulation of the fused kernels (grouped tile order, edge wrap + masked store, K in
ascending BLOCK_K chunks) reproduces the two-pass result bit for bit when the fp32 accumulator is exact in any order
(integer-valued fp8 inputs).  The GEMM is emulated in fp32 on the CPU; the Triton kernels themselves are checked on
the GPU (tests/test_fp8_fused_gpu.py, scripts/fp8_epilogue_probe.py)."""
import unittest
from unittest.mock import patch

import torch

from h3_tp2 import fp8_gemm, fp8_swiglu

E4M3 = torch.float8_e4m3fn


def _bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _ulp_diff16(x, y):
    """Distance in units of the last place between two bf16/fp16 tensors (ordered integer keys; +0 == -0)."""
    def key(t):
        b = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(b < 0, -(b & 0x7FFF), b)
    return (key(x) - key(y)).abs()


def _scales(M, N, gen):
    # per-token activation scales over three decades, per-channel weight scales as from amax / 448 of 0.02-std weights
    sa = torch.exp(torch.empty(M).uniform_(-9.0, -2.0, generator=gen))
    sb = torch.exp(torch.empty(N).uniform_(-9.5, -7.5, generator=gen))
    return sa, sb


def _int_operands(M, N, K, gen):
    a = torch.randint(-4, 5, (M, K), generator=gen).to(torch.float32).to(E4M3)
    w = torch.randint(-2, 3, (N, K), generator=gen).to(torch.float32).to(E4M3)
    return a, w


def _rand_operands(M, N, K, gen):
    a = (torch.randn(M, K, generator=gen) * 100).clamp(-448, 448).to(E4M3)
    w = (torch.randn(N, K, generator=gen) * 100).clamp(-448, 448).to(E4M3)
    return a, w


def _emulate_acc_tile(a, w, rows, cols, block_k):
    """The kernel's mainloop for one tile: fp32 accumulator, K in ascending BLOCK_K chunks, masked K tail = zeros."""
    K = a.shape[1]
    acc = torch.zeros(len(rows), len(cols))
    for k0 in range(0, K, block_k):
        at = a[rows, k0:k0 + block_k].float()
        bt = w[cols, k0:k0 + block_k].float().t()
        acc = acc + at @ bt
    return acc


def _emulate_fused(a, w, sa, sb, bias, out_dtype, cfg):
    """Tile-by-tile emulation of _fp8_mm_fused: every program id, the grouped tile map, wrapped edge reads, masked store."""
    bm, bn, bk = cfg
    M, N = a.shape[0], w.shape[0]
    y = torch.zeros(M, N, dtype=out_dtype)
    written = torch.zeros(M, N, dtype=torch.int32)
    grid = -(-M // bm) * -(-N // bn)
    for pid in range(grid):
        pm, pn = fp8_gemm.tile_coords(pid, M, N, bm, bn, fp8_gemm._GROUP_M)
        offs_m = torch.arange(pm * bm, pm * bm + bm)
        offs_n = torch.arange(pn * bn, pn * bn + bn)
        acc = _emulate_acc_tile(a, w, offs_m % M, offs_n % N, bk)
        m_ok, n_ok = offs_m < M, offs_n < N
        tile = fp8_gemm.epilogue_torch(acc, sa[offs_m % M], sb[offs_n % N], None if bias is None else bias[offs_n % N], out_dtype)
        rr, cc = offs_m[m_ok], offs_n[n_ok]
        y[rr[:, None], cc[None, :]] = tile[m_ok][:, n_ok]
        written[rr[:, None], cc[None, :]] += 1
    return y, written


def _emulate_fused_swiglu(a, w, sa, sb, cfg):
    """Tile-by-tile emulation of _fp8_mm_swiglu: gate columns n and up columns half + n share each A tile."""
    bm, bn, bk = cfg
    M, width = a.shape[0], w.shape[0]
    half = width // 2
    y = torch.zeros(M, half, dtype=torch.bfloat16)
    written = torch.zeros(M, half, dtype=torch.int32)
    for pid in range(-(-M // bm) * -(-half // bn)):
        pm, pn = fp8_gemm.tile_coords(pid, M, half, bm, bn, fp8_gemm._GROUP_M)
        offs_m = torch.arange(pm * bm, pm * bm + bm)
        offs_n = torch.arange(pn * bn, pn * bn + bn)
        rows, cols = offs_m % M, offs_n % half
        acc = torch.cat([_emulate_acc_tile(a, w, rows, cols, bk), _emulate_acc_tile(a, w, rows, cols + half, bk)], dim=1)
        tile = fp8_swiglu.swiglu_epilogue_torch(acc, sa[rows], torch.cat([sb[cols], sb[cols + half]]), None)
        m_ok, n_ok = offs_m < M, offs_n < half
        rr, cc = offs_m[m_ok], offs_n[n_ok]
        y[rr[:, None], cc[None, :]] = tile[m_ok][:, n_ok]
        written[rr[:, None], cc[None, :]] += 1
    return y, written


def _two_pass(a, w, sa, sb, bias, out_dtype):
    one = torch.ones(())
    x32 = torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
    return x32, fp8_gemm.scale_bias_cast_torch(x32, sa, sb, bias, out_dtype)


class EpilogueMath(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)

    def test_epilogue_equals_two_pass_without_bias(self):
        M, N = 257, 1031
        acc = torch.randn(M, N, generator=self.gen) * 3e4
        sa, sb = _scales(M, N, self.gen)
        for dt in (torch.bfloat16, torch.float16):
            ref = fp8_gemm.scale_bias_cast_torch(acc, sa, sb, None, dt)
            self.assertTrue(torch.equal(_bits(fp8_gemm.epilogue_torch(acc, sa, sb, None, dt)), _bits(ref)), dt)

    def test_epilogue_order_is_scale_b_first(self):
        """The order matters for the rounding: (sb * acc) * sa, not (sa * sb) * acc (vLLM's CUTLASS epilogue order)."""
        acc = torch.randn(64, 4096, generator=self.gen) * 3e4
        sa, sb = _scales(64, 4096, self.gen)
        ours = fp8_gemm.epilogue_torch(acc, sa, sb, None, torch.float32)
        other = (sa.view(-1, 1) * sb.view(1, -1)) * acc
        self.assertTrue(torch.equal(ours, (sb.view(1, -1) * acc) * sa.view(-1, 1)))
        self.assertFalse(torch.equal(ours, other))   # fp32 results differ somewhere, so the order is pinned down

    def test_epilogue_with_bias_within_one_ulp_of_the_eager_reference(self):
        """The kernels use one fma; the eager reference adds separately (two roundings): at most 1 ulp apart."""
        M, N = 128, 2048
        acc = torch.randn(M, N, generator=self.gen) * 3e4
        sa, sb = _scales(M, N, self.gen)
        bias = (torch.randn(N, generator=self.gen) * 0.1).to(torch.bfloat16)
        out = fp8_gemm.epilogue_torch(acc, sa, sb, bias, torch.bfloat16)
        ref = fp8_gemm.scale_bias_cast_torch(acc, sa, sb, bias, torch.bfloat16)
        self.assertLessEqual(_ulp_diff16(out, ref).max().item(), 1)

    def test_swiglu_epilogue_equals_two_pass_plus_silu_and_mul(self):
        M, width = 96, 2 * 1500
        acc = torch.randn(M, width, generator=self.gen) * 3e6      # gate values up to ~ +-15 after the scales
        sa = torch.rand(M, generator=self.gen) * 0.01 + 0.001
        sb = torch.rand(width, generator=self.gen) * 0.01 + 0.001
        with patch.dict(fp8_gemm._state, {"triton": False}):
            ref = fp8_swiglu.scale_swiglu_reference(acc, sa, sb, None)
        out = fp8_swiglu.swiglu_epilogue_torch(acc, sa, sb, None)
        # same values everywhere; bit patterns too except the sign of zero products (vLLM's kernel, which the fused
        # kernels follow, gives +0 * silu sign where torch's native mul gives the IEEE sign)
        self.assertTrue(torch.equal(out.float(), ref.float()))
        nz = ref.float() != 0
        self.assertTrue(torch.equal(_bits(out)[nz], _bits(ref)[nz]))


class TileEmulation(unittest.TestCase):
    """The fused kernels' structure, emulated on the CPU, against the two-pass path."""

    def setUp(self):
        self.gen = torch.Generator().manual_seed(1)

    def test_tile_map_is_a_bijection(self):
        for M, N, bm, bn, gm in [(200, 400, 128, 128, 8), (42848, 21504, 128, 256, 8), (1, 16, 64, 64, 8),
                                 (1000, 96, 64, 32, 4), (21952, 14336, 256, 128, 8)]:
            nm, nn = -(-M // bm), -(-N // bn)
            seen = {fp8_gemm.tile_coords(p, M, N, bm, bn, gm) for p in range(nm * nn)}
            self.assertEqual(len(seen), nm * nn, (M, N, bm, bn))
            self.assertTrue(all(0 <= i < nm and 0 <= j < nn for i, j in seen))

    def test_fused_equals_two_pass_bitwise_on_exact_accumulators(self):
        M, N, K = 200, 160, 192                              # partial tiles in both directions, K tail for bk = 128
        a, w = _int_operands(M, N, K, self.gen)
        sa, sb = _scales(M, N, self.gen)
        with patch.dict(fp8_gemm._state, {"triton": False}):
            x32, ref = _two_pass(a, w, sa, sb, None, torch.bfloat16)
        self.assertTrue(torch.equal(x32, a.float() @ w.float().t()))   # exact accumulator: order cannot matter
        for cfg in [(64, 64, 64), (128, 32, 128), (32, 128, 32)]:
            out, written = _emulate_fused(a, w, sa, sb, None, torch.bfloat16, cfg)
            self.assertTrue(torch.equal(written, torch.ones_like(written)), cfg)    # every output stored exactly once
            self.assertTrue(torch.equal(_bits(out), _bits(ref)), cfg)

    def test_fused_random_inputs_differ_only_by_reduction_order(self):
        """With random fp8 inputs the K-chunked fp32 sum is a different order from a single matmul: the outputs
        agree to fp32 reduction noise, so after the bf16 rounding most are equal and the rest 1 ulp apart."""
        M, N, K = 96, 128, 512
        a, w = _rand_operands(M, N, K, self.gen)
        sa, sb = _scales(M, N, self.gen)
        with patch.dict(fp8_gemm._state, {"triton": False}):
            _, ref = _two_pass(a, w, sa, sb, None, torch.bfloat16)
        out, _ = _emulate_fused(a, w, sa, sb, None, torch.bfloat16, (64, 64, 128))
        d = _ulp_diff16(out, ref)
        self.assertGreater((d == 0).float().mean().item(), 0.95)
        rel = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        self.assertLess(rel, 1e-2)

    def test_fused_swiglu_equals_two_pass_swiglu_on_exact_accumulators(self):
        M, half, K = 150, 96, 256
        a, w = _int_operands(M, 2 * half, K, self.gen)
        sa = torch.rand(M, generator=self.gen) * 0.02 + 1e-3
        sb = torch.rand(2 * half, generator=self.gen) * 0.02 + 1e-3
        one = torch.ones(())
        x32 = torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        with patch.dict(fp8_gemm._state, {"triton": False}):
            ref = fp8_swiglu.scale_swiglu_reference(x32, sa, sb, None)
        for cfg in [(64, 32, 64), (128, 64, 128)]:
            out, written = _emulate_fused_swiglu(a, w, sa, sb, cfg)
            self.assertTrue(torch.equal(written, torch.ones_like(written)), cfg)
            self.assertTrue(torch.equal(out.float(), ref.float()), cfg)


class Dispatch(unittest.TestCase):
    def test_cpu_tensors_fall_back_to_two_pass(self):
        gen = torch.Generator().manual_seed(2)
        a, w = _rand_operands(32, 64, 128, gen)
        sa, sb = _scales(32, 64, gen)
        with patch.dict(fp8_gemm._state, {"triton": False, "fused": True}):
            out = fp8_gemm.scaled_mm_fused(a, w.t(), sa.view(-1, 1), sb, torch.bfloat16)
            ref = fp8_gemm.scaled_mm_tensor32(a, w.t(), sa.view(-1, 1), sb, torch.bfloat16)
        self.assertTrue(torch.equal(_bits(out), _bits(ref)))

    def test_fused_config_env(self):
        with patch.dict("os.environ", {"H3_TP2_FP8_FUSED_CFG": "128,256,64,8,3"}):
            self.assertEqual(fp8_gemm.fused_config(), (128, 256, 64, 8, 3))
        with patch.dict("os.environ", {"H3_TP2_FP8_FUSED_CFG": "128,256"}):
            with self.assertRaises(ValueError):
                fp8_gemm.fused_config()
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("H3_TP2_FP8_FUSED_CFG", None)
            self.assertEqual(fp8_gemm.fused_config(), fp8_gemm._DEFAULT_CFG)

    def test_stash_hook_hands_operands_when_gemm_swiglu_is_on(self):
        a = torch.zeros(8, 32, dtype=E4M3)
        b = torch.zeros(64, 32, dtype=E4M3).t()
        sa, sb = torch.ones(8, 1), torch.ones(64)
        fp8_swiglu._tls.stash_next = True
        fp8_swiglu._tls.expect_n = 64
        try:
            with patch.dict(fp8_swiglu._state, {"gemm_swiglu": True}), \
                    patch.object(torch, "_scaled_mm", side_effect=AssertionError("no fp32 product expected")):
                ph = fp8_swiglu._stash_hook(a, b, sa, sb, torch.bfloat16, None)
            self.assertEqual(tuple(ph.shape), (8, 64))
            ops, s_a, s_b, bias = fp8_swiglu._tls.stash
            self.assertIsInstance(ops, fp8_swiglu.GemmOperands)
            self.assertIs(ops.a, a)
            self.assertEqual(ops.shape, (8, 64))
            self.assertEqual(tuple(s_a.shape), (8,))
        finally:
            fp8_swiglu._tls.stash = None
            fp8_swiglu._tls.stash_next = False
            fp8_swiglu._tls.expect_n = 0

    def test_forward_runs_gemm_swiglu_on_an_operand_stash(self):
        class FC1:
            output_size_per_partition = 64

            def __call__(self, x):
                if getattr(fp8_swiglu._tls, "stash_next", False):
                    fp8_swiglu._tls.stash = (fp8_swiglu.GemmOperands(x, torch.zeros(16, 64)), torch.ones(4), torch.ones(64), None)
                    fp8_swiglu._tls.stash_next = False
                    return torch.empty(4, 64), None
                raise AssertionError("fc1 called twice")

        class FC2:
            def __call__(self, h):
                return h.sum(-1, keepdim=True), None

        class MLP:
            def __init__(self):
                self.fc1, self.fc2, self.act_fn = FC1(), FC2(), None

            def forward(self, x):  # replaced by _install_on
                raise AssertionError

        class NoLoRA:
            pass

        fp8_swiglu._install_on(MLP, level=1, lora_layer_type=NoLoRA)
        x = torch.zeros(4, 16)
        with patch.object(fp8_swiglu, "gemm_swiglu", side_effect=lambda a, b, sa, sb, bias: torch.full((4, 32), 2.0)) as g, \
                patch.object(fp8_swiglu, "scale_swiglu", side_effect=AssertionError("x32 path taken")):
            out = MLP.forward(MLP(), x)
        g.assert_called_once()
        self.assertTrue(torch.equal(out, torch.full((4, 1), 64.0)))
        self.assertIsNone(fp8_swiglu._tls.stash)


if __name__ == "__main__":
    unittest.main()
