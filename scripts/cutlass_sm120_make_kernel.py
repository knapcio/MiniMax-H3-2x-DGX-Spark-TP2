"""Derive the FP8 + scaled-epilogue kernel from CUTLASS's sm120 CuTe-DSL dense_gemm.py example (BSD-3-Clause).

  python scripts/cutlass_sm120_make_kernel.py \
      <cutlass 4.7.1>/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py \
      scripts/cutlass_sm120_dsl_fp8_gemm.py
Every edit is an exact-anchor substitution (fails loudly if the example changes). Env knobs of the generated kernel, for
diagnosis only (defaults are the correct ones): DSL_NPERM=layout|plain16|plain8, DSL_B_LDSM=4|2, DSL_DIRECT_STORE=1|0.
"""
import re, sys
src = open(sys.argv[1]).read()

def sub(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        raise SystemExit(f"patch anchor found {n}x (want {count}): {old[:80]!r}")
    src = src.replace(old, new)

# 1. constructor: tile/warp layout, scheduler swizzle, epilogue stages and the scaled epilogue are parameters
sub("""    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
    ):
        self.acc_dtype = acc_dtype""",
"""    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        atom_layout=(2, 2, 1),
        swizzle_size=8,
        raster_along_m=False,
        epi_stage=4,
        scaled=False,
        mma_regs=232,
    ):
        self.acc_dtype = acc_dtype
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.epi_stage_cfg = epi_stage
        # scaled: y = sa[m] * (sb[n] * acc) in fp32 (h3_tp2.fp8_gemm order, no bias), one rounding to c_dtype
        self.scaled = scaled""")
sub("""        self.atom_layout = (2, 2, 1)""", """        self.atom_layout = tuple(atom_layout)""")
sub("""        self.mma_register_requirement = 232""", """        self.mma_register_requirement = mma_regs""")

# 2. FP8 warp MMA (mma.sync m16n8k32 e4m3 x e4m3 -> f32) when the operands are 8-bit
sub("""        self.mma_inst_mnk = (16, 8, 16)
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )""",
"""        if self.a_dtype.width == 8:
            self.mma_inst_mnk = (16, 8, 32)
            op = cute.nvgpu.warp.MmaFP8Op(self.a_dtype, self.acc_dtype, self.mma_inst_mnk)
        else:
            self.mma_inst_mnk = (16, 8, 16)
            op = cute.nvgpu.warp.MmaF16BF16Op(
                self.a_dtype,
                self.acc_dtype,
                self.mma_inst_mnk,
            )""")

# 2b. 8-bit operands: N permutation (8, atomN, 2):(1, 16, 8) as CUTLASS's sm120 builder / get_permutation_mnk, so
# one ldmatrix.x4 (four 8x16-byte matrices) feeds the two n8 MMAs of a warp's 16 contiguous columns; the example's
# plain N tiling (atomN * 8 * 2) is only right for 16-bit operands (it corrupts columns 16-31 of every 32 with fp8)
sub("""        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            # TODO: to leverage ldmatrix.x4, when self.atom_layout[1] is 1, mma tile is ((8x16)x2)
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )""", """        tC = cute.make_layout(self.atom_layout)
        if self.a_dtype.width == 8:
            nperm = os.environ.get("DSL_NPERM", "layout")
            if nperm == "layout":
                perm_n = cute.make_layout((8, self.atom_layout[1], 2), stride=(1, 16, 8))
            elif nperm == "plain16":
                perm_n = self.atom_layout[1] * 8 * 2
            else:
                perm_n = self.atom_layout[1] * 8
            permutation_mnk = (
                self.atom_layout[0] * self.mma_inst_mnk[0],
                perm_n,
                self.atom_layout[2] * self.mma_inst_mnk[2],
            )
        else:
            permutation_mnk = (
                self.atom_layout[0] * self.mma_inst_mnk[0],
                # TODO: to leverage ldmatrix.x4, when self.atom_layout[1] is 1, mma tile is ((8x16)x2)
                self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
                self.atom_layout[2] * self.mma_inst_mnk[2],
            )""")


sub("""            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),""",
"""            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), self.b_ldsm),""")
sub("""        self.scaled = scaled""", """        self.scaled = scaled
        self.b_ldsm = int(os.environ.get("DSL_B_LDSM", "4"))""")
sub("""import argparse
""", """import argparse
import os
""")

# 3. epilogue stage count is a parameter (the example's 8 stages take 32 KB of the 99 KB)
sub("""            self.smem_capacity,
            self.occupancy,
        )

        import sys

        if self.ab_stage == 0:
            print("ab_stage == 0, no enough shared memory. This case will be skipped.")
            sys.exit(0)""",
"""            self.smem_capacity,
            self.occupancy,
            self.epi_stage_cfg,
        )

        if self.ab_stage < 2:
            raise ValueError(f"ab_stage {self.ab_stage} < 2: not enough shared memory for this tile")""")
sub("""        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:""", """        smem_capacity: int,
        occupancy: int,
        epi_stage: int = 8,
    ) -> tuple[int, int]:""")
sub("""        epi_stage = 8
        c_bytes_per_stage""", """        c_bytes_per_stage""")

# 4. scheduler swizzle / raster order
sub("""        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )""", """        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
            self.swizzle_size,
            self.raster_along_m,
        )""")
sub("""        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:""", """        tile_shape_mnk: tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
        swizzle_size: int = 1,
        raster_along_m: bool = True,
    ) -> tuple[int, int, int]:""")
sub("""        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )""", """        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, swizzle_size, raster_along_m
        )""")

# 5. scale vectors through the host entry point into the kernel
sub("""        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):""", """        c: cute.Tensor,
        sa: cute.Tensor,
        sb: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):""")
sub("""            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(""", """            self.epi_smem_layout_staged,
            tile_sched_params,
            sa,
            sb,
        ).launch(""")
sub("""        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):""", """        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        mSA: cute.Tensor,
        mSB: cute.Tensor,
    ):""")

# 6. the scaled epilogue on the fp32 accumulators, in registers, before the bf16 conversion
sub("""                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOG
                # /////////////////////////////////////////////////////////////////////////////
""", """                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOG
                # /////////////////////////////////////////////////////////////////////////////
                if cutlass.const_expr(self.scaled):
                    # t = sb[n] * acc, y = sa[m] * t (fp32, no reassociation), rounded once below by .to(c_dtype)
                    # sa / sb are padded to the scheduler's rounded tile grid, so tiles past M / N read valid (ignored) scales
                    cC = cute.make_identity_tensor(
                        (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
                    )
                    tCcC = thr_mma.partition_C(cC)
                    m_base = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                    n_base = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                    for i in cutlass.range_constexpr(cute.size(accumulators)):
                        crd = tCcC[i]
                        t = mSB[n_base + crd[1]] * accumulators[i]
                        accumulators[i] = mSA[m_base + crd[0]] * t
""")

# 7. direct epilogue (default): registers -> global through the MMA's own C partition. The example's TMA-store
# epilogue slices the accumulator by flat index per 64x32 epi tile, which scrambles fp32 output and corrupts a few
# percent of bf16 values once the fp8 C layout and 128x128 tiles are used (bit-exact only at 64x64 tiles).
sub("""        self.scaled = scaled
""", """        self.scaled = scaled
        self.direct_store = int(os.environ.get("DSL_DIRECT_STORE", "1")) == 1
""")
sub("""            tile_sched_params,
            sa,
            sb,
        ).launch(""", """            tile_sched_params,
            sa,
            sb,
            c,
        ).launch(""")
sub("""        mSA: cute.Tensor,
        mSB: cute.Tensor,
    ):""", """        mSA: cute.Tensor,
        mSB: cute.Tensor,
        mCd: cute.Tensor,
    ):""")
i0 = src.index("                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(")
i1 = src.index("                tma_store_pipeline.producer_tail()\n") + len("                tma_store_pipeline.producer_tail()\n")
block = src[i0:i1]
block = "".join(("    " + l if l.strip() else l) for l in block.splitlines(True))
direct = """                if cutlass.const_expr(self.direct_store):
                    gCd = cute.local_tile(
                        mCd,
                        cute.slice_(self.tile_shape_mnk, (None, None, 0)),
                        (None, None, None),
                    )
                    tCgCd = thr_mma.partition_C(gCd[(None, None, *tile_coord_mnl)])
                    rD = cute.make_rmem_tensor_like(accumulators, self.c_dtype)
                    rD.store(accumulators.load().to(self.c_dtype))
                    # the swizzled scheduler rounds the tile grid up to a multiple of swizzle_size, so tiles
                    # past M (or N) exist: they are predicated element by element (and skipped when empty)
                    m_rows = mCd.shape[0]
                    n_cols = mCd.shape[1]
                    m_tile0 = tile_coord_mnl[0] * self.tile_shape_mnk[0]
                    n_tile0 = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                    if (m_tile0 + self.tile_shape_mnk[0] <= m_rows) and (
                        n_tile0 + self.tile_shape_mnk[1] <= n_cols
                    ):
                        cute.autovec_copy(rD, tCgCd)
                    else:
                        cDd = cute.make_identity_tensor(
                            (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
                        )
                        tCcDd = thr_mma.partition_C(cDd)
                        for i in cutlass.range_constexpr(cute.size(rD)):
                            if (m_tile0 + tCcDd[i][0] < m_rows) and (
                                n_tile0 + tCcDd[i][1] < n_cols
                            ):
                                tCgCd[i] = rD[i]
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
                else:
"""
src = src[:i0] + direct + block + src[i1:]


# 8. MMA-warp barrier before every stage release, as CUTLASS's C++ sm120 mainloop (sm120_mma_tma.hpp:
# NamedBarrier::sync(thr_size(tiled_mma), Sm120MainloopBarrier) before consumer_release). The example releases with
# lane 0 of each warp only and races with the other warps' ldmatrix reads once more than 2 stages let the producer
# refill early (wrong tiles, different on every run, at ab_stage 4/5/11; clean at ab_stage 2).
sub("""        self.load_register_requirement = 40""", """        self.load_register_requirement = 40
        self.mainloop_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )""")
src = re.sub(r"(?m)^( +)(mainloop_pipeline\.consumer_release\(mainloop_consumer_state\))$",
             r"\1self.mainloop_sync_barrier.arrive_and_wait()\n\1\2", src)
assert src.count("self.mainloop_sync_barrier.arrive_and_wait()") == 2, src.count("self.mainloop_sync_barrier.arrive_and_wait()")

# drop the example's CLI entry point (the probe drives the class directly)
src = src[: src.index('if __name__ == "__main__":')]
src = src.replace('"""\nA high-performance batched dense GEMM', '"""\nFP8 variant with a per-row/per-column scaled epilogue (H3 TP2 probe; derived from the CUTLASS example).\nA high-performance batched dense GEMM', 1)
open(sys.argv[2], "w").write(src)
print("ok")
