# Inductor (torch 2.13.0+cu130, Triton 3.7.1, sm_121) kernel of vLLM QuantFP8.forward_native, dynamic per token,
# compiled with dynamic=True as CustomOp.maybe_compile does in serving (s77 = M, s27 = K symbolic); copied from
# /tmp/torchinductor_knapcio/ek/ on Spark_06 (2026-11-01 12:18).  ulysses_outq._quant_rows_kernel mirrors its arithmetic.

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': 'fp64', 'out_ptr1': '*fp32', 'out_ptr2': '*fp8e4nv', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_abs_clamp_max_mul_reciprocal_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 5016, 'r0_': 12840960}, 'backend_hash': 'AC2F7D1CA10DE599CEEED18D63D87835E89E81120B950A8ED01FD4032DC1C4E1', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_abs_clamp_max_mul_reciprocal_unsqueeze_0(in_ptr0, in_ptr1, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp3_index = tl.full([XBLOCK, R0_BLOCK], 2147483647, tl.int32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl_math.abs(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        _tmp3_next, _tmp3_index_next = triton_helpers.maximum_with_index(
            _tmp3, _tmp3_index, tmp2, rindex
        )
        _tmp3 = tl.where(r0_mask & xmask, _tmp3_next, _tmp3)
        _tmp3_index = tl.where(r0_mask & xmask, _tmp3_index_next, _tmp3_index)
    tmp3_val, tmp3_idx = triton_helpers.max_with_index(_tmp3, _tmp3_index, 1)
    tmp3 = tmp3_val[:, None]
    tmp5 = in_ptr1
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = (tmp4 / tmp6)
    tmp8 = tl.full([1, 1], 4.359654017857143e-06, tl.float32)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 1.0, tl.float32)
        tmp13 = (tmp12 / tmp9)
        tmp14 = tmp11 * tmp13
        tmp15 = tl.full([1, 1], -448.0, tl.float32)
        tmp16 = triton_helpers.maximum(tmp14, tmp15)
        tmp17 = tl.full([1, 1], 448.0, tl.float32)
        tmp18 = triton_helpers.minimum(tmp16, tmp17)
        tmp19 = tmp18.to(tl.float8e4nv)
        tl.store(out_ptr2 + (r0_1 + ks0*x0), tmp19, r0_mask & xmask)
