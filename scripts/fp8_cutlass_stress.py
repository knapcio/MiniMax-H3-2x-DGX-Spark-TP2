#!/usr/bin/env python3
"""Stress reproducer for the stage-release race of the CuTe-DSL sm120 FP8 GEMM (``H3_TP2_FP8_GEMM=cutlass``;
``results/20261103-cutlass-sm120/race.md``).

Runs many back-to-back routed-shaped GEMMs (M = 10240 by default; qkv N = 21504, fc1 N = 28672, K = 5376; random e4m3
operands and per-row / per-column fp32 scales, ``--nbuf`` activation buffers, one weight per shape, one shared output
buffer) on the current stream through ``fp8_cutlass.DslGemm.run`` (the dispatcher's launcher) while a busy kernel loop
runs on a second, non-blocking stream, and checks every DSL output bit for bit against ``fp8_gemm.scaled_mm_tensor32``
on the same operands.  The reference is computed once per (shape, buffer) and kept as checksums only (Triton: per row,
per 128-column N tile and per MMA warp column group (col % 32) // 16, 64-bit weighted sums of the bf16 bit patterns),
so memory stays below ~1.9 GB and the GEMM stream is never drained per call (one host sync every ``--window`` calls).
A mismatch makes the check kernel copy the 128 output values of that (row, N tile) into a capture buffer.

Report, per mismatching call and 128x128 tile: rows, rows within the tile, columns, the MMA warps (warp_m = (row % 32)
// 16, warp_n = (col % 32) // 16 for the 2x2 warps), the persistent CTA and its tile position (the kernel's swizzled
raster), and the attribution: which (k tile, 32-byte k block) of which smem stage was read with the A and / or B data
of fill g + ab_stage (the next TMA load into that stage: the stage was overwritten before the ldmatrix read it, a WAR)
or g - ab_stage (the previous load: read before the TMA data landed, a RAW); ``class`` WAR / RAW when that model
leaves < 2 % of the error energy.  At the end the reference checksums are recomputed (a flaky reference would show up
there, not as a DSL race).

The busy kernel that reproduces the race is ``chase`` (dependent random loads from a table larger than L2, i.e.
long-latency loads keeping the SM's load path busy next to the GEMM CTA); ``add`` (streaming elementwise) and ``none``
did not reproduce it.  Measured on Spark_06 with the pre-fix kernel: 5..44 mismatching calls per 240 s run; with the
fix 0 (race.md).

Usage (Spark_06, ~1.9 GB of GPU memory, ~4.5 minutes):
  ~/h3-long/.venv/bin/python scripts/fp8_cutlass_stress.py --seconds 240 --busy chase --busy-per-call 80 --nbuf 2 \
      --jsonl stress.jsonl [--save events.pt]
  ~/h3-long/.venv/bin/python scripts/fp8_cutlass_stress.py --reanalyze events.pt     # attribution again, no stress
Other knobs: --busy add|smem|chase|hostpoll|mm|mix|sleep|none, --m (e.g. 10241: a partial last row tile), --shapes,
--window, --detail / --detail-from / --detail-rows.  Exit status 1 if any call mismatched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from h3_tp2 import fp8_cutlass, fp8_gemm  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

E4M3 = torch.float8_e4m3fn
SHAPES = {"qkv": (21504, 5376), "fc1": (28672, 5376), "out": (5376, 7168)}
TN = 128          # N tile of the kernel (checksum granularity)
CAP = 4096        # captured (row, N tile) segments
BR = 32           # rows per check program


@triton.jit
def _rowtile_check(y_ptr, stride_y, M, ref_ptr, w_ptr, cap_vals, cap_meta, cap_cnt, flags, call_id, NT,
                   CAPN: tl.constexpr,
                   MODE: tl.constexpr, BR: tl.constexpr, TN: tl.constexpr):
    """MODE 0: write the reference checksums ref[(row, n tile, warp_n)]; MODE 1: compare and capture mismatches."""
    pr = tl.program_id(0)
    pt = tl.program_id(1)
    rows = pr * BR + tl.arange(0, BR)
    cols = tl.arange(0, TN)
    rmask = rows < M
    x16 = tl.load(y_ptr + rows[:, None].to(tl.int64) * stride_y + pt * TN + cols[None, :], mask=rmask[:, None], other=0)
    x = x16.to(tl.int64) + 32768
    w = tl.load(w_ptr + cols)
    g = (cols // 16) % 2                     # MMA warp along N (sm120 8-bit N permutation, 2 warps)
    w0 = tl.where(g == 0, w, 0)
    w1 = tl.where(g == 1, w, 0)
    c0 = tl.sum(x * w0[None, :], axis=1)
    c1 = tl.sum(x * w1[None, :], axis=1)
    off = (rows.to(tl.int64) * NT + pt) * 2
    if MODE == 0:
        tl.store(ref_ptr + off, c0, mask=rmask)
        tl.store(ref_ptr + off + 1, c1, mask=rmask)
    else:
        r0 = tl.load(ref_ptr + off, mask=rmask, other=0)
        r1 = tl.load(ref_ptr + off + 1, mask=rmask, other=0)
        b0 = (c0 != r0) & rmask
        b1 = (c1 != r1) & rmask
        bad = b0 | b1
        nbad = tl.sum(bad.to(tl.int32), axis=0)
        if nbad > 0:
            tl.atomic_add(flags + call_id, nbad)       # per call: mismatching (row, N tile) segments
            slot = tl.atomic_add(cap_cnt + rows * 0, 1, mask=bad)
            keep = bad & (slot < CAPN)
            slot64 = slot.to(tl.int64)
            tl.store(cap_vals + slot64[:, None] * TN + cols[None, :], x16, mask=keep[:, None])
            tl.store(cap_meta + slot64 * 4 + 0, (rows * 0 + call_id).to(tl.int64), mask=keep)
            tl.store(cap_meta + slot64 * 4 + 1, rows.to(tl.int64), mask=keep)
            tl.store(cap_meta + slot64 * 4 + 2, (rows * 0 + pt).to(tl.int64), mask=keep)
            tl.store(cap_meta + slot64 * 4 + 3, (b0.to(tl.int32) + 2 * b1.to(tl.int32)).to(tl.int64), mask=keep)


@triton.jit
def _smem_hammer(a_ptr, b_ptr, c_ptr, iters, BM: tl.constexpr, BK: tl.constexpr):
    """Busy kernel with a small shared-memory footprint (fits next to the GEMM CTA's ~86 KB): each iteration loads
    two fresh [BM, BK] fp16 tiles (L1/L2 hits) and feeds them to tl.dot, i.e. register -> smem -> ldmatrix traffic on
    the same SM as the GEMM's ldmatrix reads."""
    pid = tl.program_id(0)
    r = tl.arange(0, BM)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for i in range(iters):
        o = ((pid + i) % 8) * BM * BK
        a = tl.load(a_ptr + o + r[:, None] * BK + k[None, :])
        b = tl.load(b_ptr + o + k[:, None] * BM + r[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(c_ptr + pid * BM * BM + r[:, None] * BM + r[None, :], acc)


@triton.jit
def _chase(tab_ptr, out_ptr, steps, mask_n, VOLATILE: tl.constexpr):
    """Busy kernel with long-latency loads: per lane a dependent chain of random loads (a table larger than L2, or
    pinned host memory polled with volatile loads like NCCL's flag polling), keeping the SM's load path full of
    outstanding misses next to the GEMM CTA."""
    pid = tl.program_id(0)
    lane = tl.arange(0, 128)
    idx = ((pid * 128 + lane) * 7919 % mask_n).to(tl.int64)
    acc = tl.zeros((128,), dtype=tl.int32)
    for _ in range(steps):
        v = tl.load(tab_ptr + idx, volatile=VOLATILE)
        acc += v
        idx = (v.to(tl.int64) + idx * 31 + 17) % mask_n
    tl.store(out_ptr + pid * 128 + lane, acc)


class _HostPtr:
    """Pinned host memory handed to Triton by address (unified addressing: the host pointer is the device pointer)."""

    def __init__(self, t):  # noqa: ANN001
        self.t = t
        self.dtype = t.dtype

    def data_ptr(self):  # noqa: ANN201
        return self.t.data_ptr()


def rowtile_check(y, ref, wts, cap, mode, call_id=0):  # noqa: ANN001,ANN201
    M, N = y.shape
    NT = N // TN
    vals, meta, cnt, flags = cap
    _rowtile_check[(triton.cdiv(M, BR), NT)](y.view(torch.int16), y.stride(0), M, ref, wts, vals, meta, cnt, flags,
                                            call_id, NT,
                                            CAPN=CAP, MODE=mode, BR=BR, TN=TN)


def tile_schedule(M, N, swizzle, grid):  # noqa: ANN001,ANN201
    """The kernel's persistent schedule (raster along N, swizzle groups of ``swizzle`` M tiles: linear = m % sw +
    sw * n + sw * n_tiles * (m // sw); CTA c takes linear c, c + grid, ...).  Returns where(m, n) -> (CTA, position)
    and tile_at(CTA, position) -> (m, n) or None (past the grid or a padded tile past M)."""
    mt, nt = -(-M // 128), -(-N // 128)
    total = (mt + swizzle - 1) // swizzle * swizzle * nt

    def where(m, n):  # noqa: ANN001,ANN202
        lin = (m % swizzle) + swizzle * n + swizzle * nt * (m // swizzle)
        return lin % grid, lin // grid

    def tile_at(cta, pos):  # noqa: ANN001,ANN202
        lin = cta + pos * grid
        if pos < 0 or lin >= total:
            return None
        m = (lin // (swizzle * nt)) * swizzle + lin % swizzle
        return (m, (lin % (swizzle * nt)) // swizzle) if m < mt else None

    return where, tile_at


def make_operands(M, shapes, seed, dev, nbuf=1):  # noqa: ANN001,ANN201
    g = torch.Generator(device=dev).manual_seed(seed)
    K = SHAPES[shapes[0]][1]
    assert all(SHAPES[s][1] == K for s in shapes), "all shapes must share K (the A buffers)"
    AS = []
    for _ in range(nbuf):
        a = torch.empty(M, K, device=dev, dtype=E4M3)
        for i in range(0, M, 4096):
            m = min(4096, M - i)
            a[i:i + m] = (torch.randn(m, K, device=dev, generator=g) * 64).clamp(-448, 448).to(E4M3)
        AS.append((a, torch.exp(torch.empty(M, device=dev).uniform_(-9.0, -2.0, generator=g))))
    W = {}
    for s in shapes:
        N = SHAPES[s][0]
        w = torch.empty(N, K, device=dev, dtype=E4M3)
        for i in range(0, N, 4096):            # bounded fp32 temporaries
            n = min(4096, N - i)
            w[i:i + n] = (torch.randn(n, K, device=dev, generator=g) * 64).clamp(-448, 448).to(E4M3)
        sb = torch.exp(torch.empty(N, device=dev).uniform_(-9.5, -7.5, generator=g))
        W[s] = (w, sb)
    return AS, W


def ref_rows(a, w, sa, sb, r0, r1):  # noqa: ANN001,ANN201
    return fp8_gemm.scaled_mm_tensor32(a[r0:r1], w.t(), sa[r0:r1].view(-1, 1), sb, torch.bfloat16)


def ref_checksums(a, w, sa, sb, wts, cap, chunk):  # noqa: ANN001,ANN201
    M, N = a.shape[0], w.shape[0]
    ref = torch.empty(M * (N // TN) * 2, device=a.device, dtype=torch.int64)
    for r0 in range(0, M, chunk):
        r1 = min(M, r0 + chunk)
        y = ref_rows(a, w, sa, sb, r0, r1)
        rowtile_check(y, ref[r0 * (N // TN) * 2:], wts, cap, 0)
        del y
    return ref


def classify(C, names, obs, max_terms=8):  # noqa: ANN001,ANN201
    """Orthogonal matching pursuit of ``obs`` on candidate rows ``C`` (names (src, kt, chunk)), first restricted to the
    WAR terms (fill +ab_stage), then to the RAW terms (fill -ab_stage), then over all candidates; the first family
    that explains >= 98 % of the error energy names the event."""
    tot = float((obs ** 2).sum())

    def omp(idx):  # noqa: ANN001,ANN202
        sel, res, coef = [], obs.clone(), None
        Cs = C[idx]
        for _ in range(max_terms):
            j = int(((Cs @ res) ** 2 / (Cs ** 2).sum(1).clamp_min(1e-30)).argmax())
            if j in sel:
                break
            sel.append(j)
            X = Cs[sel].T
            coef = torch.linalg.lstsq(X, obs[:, None]).solution[:, 0]
            res = obs - X @ coef
            if float((res ** 2).sum()) < 1e-3 * tot:
                break
        expl = 1 - float((res ** 2).sum()) / max(tot, 1e-30)
        terms = [{"src": names[idx[k]][0], "kt": names[idx[k]][1], "chunk": names[idx[k]][2],
                  "coef": round(float(c), 3)} for k, c in zip(sel, coef.tolist())]
        return round(expl, 4), terms

    fam = {"WAR": [i for i, n in enumerate(names) if n[0].startswith("fill+")],
           "RAW": [i for i, n in enumerate(names) if n[0].startswith("fill-")],
           "other": list(range(len(names)))}
    out = {}
    for cls, idx in fam.items():
        if not idx:
            continue
        expl, terms = omp(torch.tensor(idx, device=C.device))
        out = {"class": cls, "explained": expl, "terms": terms}
        if expl >= 0.98:
            return out
    out["class"] = "unexplained"
    return out


def attribute_rows(a, w, sa, sb, row, cols, ywrong, yref, stages, sched):  # noqa: ANN001,ANN201
    """Explain the error of one output row on ``cols`` by replaced A data in shared memory.  Candidate terms, per k
    tile kt and 16-byte chunk (one 8x8 b16 ldmatrix row): the chunk holds the A data of fill g + f of the same smem
    stage, g = this CTA's fill number of (tile, kt), f = +ab_stage (the next TMA load into that stage: a read after
    the stage was released and refilled, a WAR) or -ab_stage (the previous load: a read before the TMA data landed, a
    RAW); fills cross tile boundaries along the CTA's persistent schedule ``sched`` = (where, tile_at, m tile, n tile,
    k tiles); per fill three variants: A from that fill, B from that fill, both.  Also: the chunk read as zeros, and
    the same row's k tile kt +- 1.  ``classify`` names the event."""
    K = a.shape[1]
    kts = K // 64
    where, tile_at, mt0, nt0 = sched
    cta, pos = where(mt0, nt0)
    rin = row % 128
    A = a[row].float().double()
    B = w.view(torch.uint8)[cols].view(E4M3).float().double()      # [n, K]
    s = (sa[row].double() * sb[cols].double())
    acc_ref = B @ A                                                # fp64 ~ exact fp32-accumulated product
    obs = ywrong.double() / s - acc_ref
    base = (yref.double() / s - acc_ref)                           # rounding noise of the reference itself
    Ak = A.view(kts, 64)
    cands, names = [], []

    def add(tag, src):  # noqa: ANN001,ANN202  src: [kts, 64] replacement bytes of this row's A
        contrib = (B.view(len(cols), kts, 4, 16) * (src - Ak).view(1, kts, 4, 16)).sum(-1)   # [n, kt, chunk]
        cands.append(contrib.permute(1, 2, 0).reshape(-1, len(cols)))
        names.extend((tag, kt, ch) for kt in range(kts) for ch in range(4))

    Bk = B.view(len(cols), kts, 4, 16)
    nin = (cols % 128).long()
    for f in (stages, -stages):
        src = torch.full_like(Ak, float("nan"))
        bsrc = torch.full_like(B, float("nan")).view(len(cols), kts, 64)
        for kt in range(kts):
            g = pos * kts + kt + f
            t = tile_at(cta, g // kts)
            if t is None:
                continue
            kk = g % kts
            if t[0] * 128 + rin < a.shape[0]:
                src[kt] = a[t[0] * 128 + rin, kk * 64:(kk + 1) * 64].float().double()
            nrow = t[1] * 128 + nin
            if int(nrow.max()) < w.shape[0]:
                bsrc[:, kt] = w.view(torch.uint8)[nrow, kk * 64:(kk + 1) * 64].view(E4M3).float().double()
        add(f"fill{f:+d}", src)                                    # A read from that fill, B correct
        bs = bsrc.view(len(cols), kts, 4, 16)
        for tag, aa in ((f"fill{f:+d}:B", Ak), (f"fill{f:+d}:AB", src)):   # B (and A) read from that fill
            contrib = (bs * aa.view(1, kts, 4, 16)).sum(-1) - (Bk * Ak.view(1, kts, 4, 16)).sum(-1)
            cands.append(contrib.permute(1, 2, 0).reshape(-1, len(cols)))
            names.extend((tag, kt, ch) for kt in range(kts) for ch in range(4))
    add("zero", torch.zeros_like(Ak))
    for d in (1, -1):
        src = torch.zeros_like(Ak)
        lo, hi = max(0, -d), min(kts, kts - d)
        src[lo:hi] = Ak[lo + d:hi + d]
        add(f"kt{d:+d}", src)
    C = torch.cat(cands)
    keep = ~torch.isnan(C).any(1)                                  # fills that do not exist (before the first tile)
    C = C[keep]
    names = [n for n, k in zip(names, keep.tolist()) if k]
    tot = float((obs ** 2).sum())
    out = classify(C, names, obs)
    out["rel_err"] = round((tot / max(float((acc_ref ** 2).sum()), 1e-30)) ** 0.5, 5)
    out["ref_noise"] = round((float((base ** 2).sum()) / max(float((acc_ref ** 2).sum()), 1e-30)) ** 0.5, 6)
    return out


def attribute_cols(a, w, sa, sb, rows, col, ywrong, yref, stages, sched):  # noqa: ANN001,ANN201
    """Column pattern (one output column wrong on many rows of a tile): wrong B data for that column.  Per (k tile,
    16-byte chunk) the least-squares replacement e of B[col, chunk] explaining all wrong rows at once ("free" fit:
    which chunk, how much is explained), then the named candidates as for A: B data of fill +- ab_stage of the same
    stage (WAR / RAW), zeros, k tile +- 1."""
    K = a.shape[1]
    kts = K // 64
    where, tile_at, mt0, nt0 = sched
    cta, pos = where(mt0, nt0)
    nin = col % 128
    R = torch.tensor(rows, device=a.device)
    A = a.view(torch.uint8)[R].view(E4M3).float().double()          # [r, K]
    Bn = w[col].float().double()
    s = sa[R].double() * sb[col].double()
    acc_ref = A @ Bn
    obs = ywrong.double() / s - acc_ref
    tot = float((obs ** 2).sum())
    Ac = A.view(len(rows), kts * 4, 16)
    best = None
    for c in range(kts * 4):
        sol = torch.linalg.lstsq(Ac[:, c, :], obs[:, None]).solution[:, 0]
        res = float(((Ac[:, c, :] @ sol - obs) ** 2).sum())
        if best is None or res < best[0]:
            best = (res, c, sol)
    res, c, sol = best
    kt, ch = divmod(c, 4)
    x = Bn.view(kts * 4, 16)[c] + sol
    free = {"kt": kt, "chunk": ch, "explained": round(1 - res / max(tot, 1e-30), 4),
            "b_true": Bn.view(kts * 4, 16)[c].round().tolist(), "b_read": [round(v, 1) for v in x.tolist()]}
    Bk = Bn.view(kts, 64)
    cands, names = [], []

    def add(tag, src):  # noqa: ANN001,ANN202
        contrib = (A.view(len(rows), kts, 4, 16) * (src - Bk).view(1, kts, 4, 16)).sum(-1)    # [r, kt, chunk]
        cands.append(contrib.permute(1, 2, 0).reshape(-1, len(rows)))
        names.extend((tag, k, h) for k in range(kts) for h in range(4))

    for f in (stages, -stages):
        src = torch.full_like(Bk, float("nan"))
        for k in range(kts):
            g = pos * kts + k + f
            t = tile_at(cta, g // kts)
            if t is not None and t[1] * 128 + nin < w.shape[0]:
                kk = g % kts
                src[k] = w[t[1] * 128 + nin, kk * 64:(kk + 1) * 64].float().double()
        add(f"fill{f:+d}", src)
    add("zero", torch.zeros_like(Bk))
    C = torch.cat(cands)
    keep = ~torch.isnan(C).any(1)
    C = C[keep]
    names = [n for n, k in zip(names, keep.tolist()) if k]
    out = classify(C, names, obs)
    out["free"] = free
    out["rel_err"] = round((tot / max(float((acc_ref ** 2).sum()), 1e-30)) ** 0.5, 5)
    return out


def tile_fit(a, w, sa, sb, mt, nt, yw_tile, stages, sched, hints=(), top=8, iters=3):  # noqa: ANN001,ANN201
    """Whole-tile attribution.  Warp (wm, wn) computes rows r with (r % 32) // 16 == wm and columns n with
    (n % 32) // 16 == wn; per 32-byte k block it reads its A rows and its B columns from the smem stage with ldmatrix.
    Model: for one (k tile kt, k block kb, fill offset f), every (A row, 16-byte chunk, reading warp) and every
    (B column, chunk, reading warp) was read either correctly or from fill g + f of the same stage (f = +ab_stage:
    the next TMA load into the stage, a WAR; -ab_stage: the previous load, a RAW).  The binary read flags are fitted
    by alternating minimisation for the ``top`` (kt, kb, f) ranked by a coarse 16x16-block fit plus the ``hints`` (the
    (f, kt, kb) of the per-row fits); the best explains the event if it leaves < 2 % of the error energy."""
    K = a.shape[1]
    kts = K // 64
    where, tile_at = sched
    cta, pos = where(mt, nt)
    dev = a.device
    ar = torch.arange(128, device=dev)
    A = a.view(torch.uint8)[mt * 128 + ar].view(E4M3).float().double()
    B = w.view(torch.uint8)[nt * 128 + ar].view(E4M3).float().double()
    acc = A @ B.T
    sc = sa[mt * 128 + ar].double()[:, None] * sb[nt * 128 + ar].double()[None, :]
    obs = yw_tile.double() / sc - acc
    tot = float((obs ** 2).sum())
    wq = (ar % 32) // 16                                   # warp index of a row (wm) or of a column (wn)
    blk = lambda X: X.view(8, 16, 8, 16).pow(2).sum((1, 3))  # noqa: E731
    e0 = blk(obs)
    cands = []
    srcs = {}
    for f in (stages, -stages):
        for kt in range(kts):
            g = pos * kts + kt + f
            t = tile_at(cta, g // kts)
            if t is None:
                continue
            kk = g % kts
            A2 = a.view(torch.uint8)[t[0] * 128 + ar][:, kk * 64:(kk + 1) * 64].view(E4M3).float().double()
            B2 = w.view(torch.uint8)[t[1] * 128 + ar][:, kk * 64:(kk + 1) * 64].view(E4M3).float().double()
            srcs[(f, kt)] = (A2, B2)
            for kb in range(2):
                sl = slice(kt * 64 + kb * 32, kt * 64 + kb * 32 + 32)
                s2 = slice(kb * 32, kb * 32 + 32)
                base = A[:, sl] @ B[:, sl].T
                EA = A2[:, s2] @ B[:, sl].T - base
                EB = A[:, sl] @ B2[:, s2].T - base
                H = torch.stack([e0, blk(obs - EA), blk(obs - EB), blk(obs - (A2[:, s2] @ B2[:, s2].T - base))])
                # separable scores: each (row, reading warp) or each (column, reading warp) old or new on its own
                rowsc = sum(torch.minimum((obs[:, wq == q] ** 2).sum(1), ((obs - EA)[:, wq == q] ** 2).sum(1)).sum()
                            for q in range(2))
                colsc = sum(torch.minimum((obs[wq == q] ** 2).sum(0), ((obs - EB)[wq == q] ** 2).sum(0)).sum()
                            for q in range(2))
                cands.append((float(H.min(0)[0].sum()), min(float(rowsc), float(colsc)), f, kt, kb))
    trial = [c[2:] for c in sorted(cands)[:top]]
    trial += [c[2:] for c in sorted(cands, key=lambda c: c[1])[:top] if c[2:] not in trial]
    trial += [h for h in hints if h not in trial and (h[0], h[1]) in srcs]
    best = None
    for f, kt, kb in trial:
        A2, B2 = srcs[(f, kt)]
        Ac = A[:, kt * 64 + kb * 32:kt * 64 + kb * 32 + 32].view(128, 2, 16)
        Bc = B[:, kt * 64 + kb * 32:kt * 64 + kb * 32 + 32].view(128, 2, 16)
        dA = A2[:, kb * 32:kb * 32 + 32].view(128, 2, 16) - Ac
        dB = B2[:, kb * 32:kb * 32 + 32].view(128, 2, 16) - Bc
        for order in (("a", "b"), ("b", "a")):                               # A flags first, then B first
            xa = torch.zeros(128, 2, 2, device=dev, dtype=torch.float64)   # [row, chunk, reading wn]
            xb = torch.zeros(128, 2, 2, device=dev, dtype=torch.float64)   # [col, chunk, reading wm]

            def model(xa, xb):  # noqa: ANN001,ANN202
                E = torch.zeros(128, 128, device=dev, dtype=torch.float64)
                for c in range(2):
                    Ar = Ac[:, c][:, None, :] + xa[:, c, :][:, wq][:, :, None] * dA[:, c][:, None, :]    # [r, n, 16]
                    Bn = Bc[:, c][None, :, :] + xb[:, c, :][:, wq].T[:, :, None] * dB[:, c][None, :, :]
                    E += (Ar * Bn).sum(-1) - Ac[:, c] @ Bc[:, c].T
                return E

            for _ in range(iters):
                for which in order:
                    for c in range(2):
                        for q in range(2):
                            x = xa if which == "a" else xb
                            r = []
                            for v in (0.0, 1.0):
                                x[:, c, q] = v
                                err = (obs - model(xa, xb)) ** 2
                                # rows (cols) read by warp q: the columns (rows) of that warp
                                r.append(err[:, wq == q].sum(1) if which == "a" else err[wq == q, :].sum(0))
                            x[:, c, q] = (r[1] < r[0]).double()
            res = float(((obs - model(xa, xb)) ** 2).sum())
            if best is None or res < best[0]:
                best = (res, f, kt, kb, xa.clone(), xb.clone())
    res, f, kt, kb, xa, xb = best
    a_new = {f"wn{q}": (xa[:, :, q].sum(1) > 0).nonzero().flatten().tolist() for q in range(2)}
    b_new = {f"wm{q}": (xb[:, :, q].sum(1) > 0).nonzero().flatten().tolist() for q in range(2)}
    return {"class": ("WAR" if f > 0 else "RAW") if res < 0.02 * tot else "unexplained",
            "explained": round(1 - res / max(tot, 1e-30), 4), "fill": f, "kt": kt, "kblock": kb,
            "a_rows_new": {k: v for k, v in a_new.items() if v}, "b_cols_new": {k: v for k, v in b_new.items() if v}}


def analyze(args, AS, W, kern, SCHED, calls, bad_calls, seg_per_call, emit, dev):  # noqa: ANN001,ANN201,C901
    """Per (call, tile) event: the exact wrong elements and their attribution (row pattern: wrong A and / or B data
    read by one or more warps; column pattern: wrong B data on many rows)."""
    compact = []
    M = args.m
    a, sa = AS[0]
    ref_cache = {}

    def ref_row(s, w, sb, row):  # noqa: ANN001,ANN202
        c0 = (row // args.ref_chunk) * args.ref_chunk
        key = (s, a.data_ptr(), c0)
        if key not in ref_cache:
            ref_cache.clear()
            ref_cache[key] = ref_rows(a, w, sa, sb, c0, min(M, c0 + args.ref_chunk))
        return ref_cache[key][row - c0]

    for cid in sorted(bad_calls)[args.detail_from:args.detail_from + args.detail]:
        s, bi = calls[cid]
        a, sa = AS[bi]
        w, sb = W[s]
        N = SHAPES[s][0]
        tiles = {}
        for row, nt, bits, v in bad_calls[cid]:
            tiles.setdefault((row // 128, nt), []).append((row, bits, v))
        rep = {"call": cid, "shape": s, "abuf": bi, "M": M, "N": N, "bad_segments": seg_per_call.get(cid), "events": []}
        for (mt, nt), segs in sorted(tiles.items())[:8]:
            cta, pos = SCHED[s][0](mt, nt)
            cols = torch.arange(nt * TN, (nt + 1) * TN, device=dev)
            wrong = {}
            for row, bits, v in sorted(segs, key=lambda x: x[0]):
                yr = ref_row(s, w, sb, row)[nt * TN:(nt + 1) * TN]
                yw = v.view(torch.bfloat16)
                d = (yw.view(torch.int16) != yr.view(torch.int16)).nonzero().flatten()
                wrong[row] = (yw, yr, (d + nt * TN).tolist())
            all_cols = sorted({c for (_, _, dc) in wrong.values() for c in dc})
            ev = {"m_tile": mt, "n_tile": nt, "cta": cta, "tile_pos": pos, "n_rows": len(wrong),
                  "rows_in_tile": sorted(r % 128 for r in wrong)[:32],
                  "warp_m": sorted({(r % 32) // 16 for r in wrong}),
                  "n_cols": len(all_cols), "cols_in_tile": [c - nt * TN for c in all_cols][:32],
                  "warp_n": sorted({(c % 32) // 16 for c in all_cols}),
                  "n_diff": sum(len(dc) for (_, _, dc) in wrong.values())}
            ev["pattern"] = "column" if len(wrong) >= 8 and len(all_cols) <= 8 else "row"
            if ev["pattern"] == "row":
                ev["rows"] = []
                for row in sorted(wrong)[: args.detail_rows]:
                    yw, yr, dc = wrong[row]
                    if not dc:
                        continue
                    wn = sorted({(c % 32) // 16 for c in dc})
                    wcols = cols[((cols % 32) // 16 == wn[0])] if len(wn) == 1 else cols
                    sel = (wcols - nt * TN).long()
                    ev["rows"].append({"row": row, "row_in_tile": row % 128, "n_diff": len(dc), "warp_n": wn,
                                       "fit": attribute_rows(a, w, sa, sb, row, wcols, yw.float()[sel], yr.float()[sel],
                                                             kern.ab_stage, (*SCHED[s], mt, nt))})
            else:
                ev["cols"] = []
                rws = sorted(wrong)
                for c in all_cols[:8]:
                    j = c - nt * TN
                    yw = torch.stack([wrong[r][0][j] for r in rws]).float()
                    yr = torch.stack([wrong[r][1][j] for r in rws]).float()
                    ev["cols"].append({"col": c, "col_in_tile": j,
                                       "fit": attribute_cols(a, w, sa, sb, rws, c, yw, yr, kern.ab_stage,
                                                             (*SCHED[s], mt, nt))})
            fits = [x["fit"] for x in ev.get("rows", []) + ev.get("cols", [])]
            hints = []
            for f_ in fits:
                for t_ in f_.get("terms", []):
                    if t_["src"].startswith("fill"):
                        h = (int(t_["src"][4:].split(":")[0]), t_["kt"], t_["chunk"] // 2)
                        if h not in hints:
                            hints.append(h)
            yw_tile = torch.stack([wrong[r][0] if r in wrong else ref_row(s, w, sb, r)[nt * TN:(nt + 1) * TN]
                                   for r in range(mt * 128, mt * 128 + 128)]).float()
            ev["tile_fit"] = tile_fit(a, w, sa, sb, mt, nt, yw_tile, kern.ab_stage, SCHED[s], hints[:8])
            classes = sorted({f["class"] for f in fits})
            kts = sorted({t["kt"] for f in fits if f["class"] in ("WAR", "RAW") for t in f["terms"]})
            ev["class"] = "+".join(classes)
            tf = ev["tile_fit"]
            compact.append({"call": cid, "shape": s, "m_tile": mt, "n_tile": nt, "cta": cta, "tile_pos": pos,
                            "rows": ev["n_rows"], "warp_m": ev["warp_m"], "warp_n": ev["warp_n"], "class": tf["class"],
                            "explained": tf["explained"], "fill": tf["fill"], "kt": tf["kt"], "kblock": tf["kblock"],
                            "a_rows_new": tf["a_rows_new"], "b_cols_new": tf["b_cols_new"],
                            "row_fit_class": ev["class"]})
            rep["events"].append(ev)
        emit({"mismatch": rep})
    ref_cache.clear()
    return compact



def main():  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=10240)
    ap.add_argument("--shapes", nargs="+", default=["qkv", "fc1"], choices=["qkv", "fc1"])
    ap.add_argument("--seconds", type=float, default=240.0, help="stop launching after this many seconds")
    ap.add_argument("--calls", type=int, default=200_000, help="upper bound on GEMM calls")
    ap.add_argument("--window", type=int, default=16,
                    help="calls between host syncs (the GPU is never drained per call)")
    ap.add_argument("--busy", default="add", choices=["add", "smem", "chase", "hostpoll", "mm", "mix", "sleep", "none"])
    ap.add_argument("--busy-mb", type=int, default=32, help="size of each add buffer")
    ap.add_argument("--busy-per-call", type=float, default=0.0, help="busy kernels per GEMM (0: ~ GEMM time, measured)")
    ap.add_argument("--ref-chunk", type=int, default=2560, help="rows per reference GEMM (bounds the fp32 product)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nbuf", type=int, default=1, help="A buffers (activations); calls cycle shapes x buffers")
    ap.add_argument("--detail", type=int, default=20, help="mismatching calls reported in detail")
    ap.add_argument("--detail-rows", type=int, default=12, help="wrong rows per call reported in detail")
    ap.add_argument("--detail-from", type=int, default=0, help="skip the first mismatching calls in the detail report")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--save", default="", help="torch.save the captured wrong segments here (offline analysis)")
    ap.add_argument("--reanalyze", default="", help="analyse a --save file again (regenerates the operands; no stress)")
    args = ap.parse_args()
    saved = None
    if args.reanalyze:
        saved = torch.load(args.reanalyze, weights_only=False)
        for k in ("m", "shapes", "seed", "nbuf", "ref_chunk"):
            setattr(args, k, saved["args"][k])
    dev = torch.device("cuda")
    out_f = open(args.jsonl, "a") if args.jsonl else None

    def emit(r):  # noqa: ANN001,ANN202
        print(json.dumps(r), flush=True)
        if out_f:
            out_f.write(json.dumps(r) + "\n")
            out_f.flush()

    torch.zeros(1, device=dev)
    assert fp8_gemm.probe_triton(dev), "Triton scale pass (the reference) unavailable"
    gemm = fp8_cutlass.DslGemm()
    bad_self = fp8_cutlass.selfcheck(gemm, dev)
    kern = gemm.kernel
    env = {k: v for k, v in os.environ.items() if k.startswith("RACE_")}
    emit({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "kernel": fp8_cutlass.__file__, "compile_s": gemm.compile_s,
          "ab_stage": kern.ab_stage, "selfcheck_bad": bad_self, "args": vars(args), "env": env,
          "device": torch.cuda.get_device_name(), "free_gb": round(torch.cuda.mem_get_info()[0] / 2**30, 2)})

    M = args.m
    AS, W = make_operands(M, args.shapes, args.seed, dev, args.nbuf)
    a, sa = AS[0]
    wts = torch.randint(1, 2**31 - 1, (TN,), device=dev, dtype=torch.int64,
                        generator=torch.Generator(device=dev).manual_seed(7))
    cap = (torch.zeros(CAP, TN, device=dev, dtype=torch.int16), torch.zeros(CAP, 4, device=dev, dtype=torch.int64),
           torch.zeros(1, device=dev, dtype=torch.int32), torch.zeros(args.calls + 1, device=dev, dtype=torch.int32))
    t0 = time.time()
    REF = {(s, b): ref_checksums(AS[b][0], W[s][0], AS[b][1], W[s][1], wts, cap, args.ref_chunk)
           for s in args.shapes for b in range(args.nbuf)}
    torch.cuda.synchronize()
    assert int(cap[2]) == 0
    t_ref = time.time() - t0
    SCHED = {s: tile_schedule(M, SHAPES[s][0], kern.swizzle_size, gemm.max_clusters) for s in args.shapes}
    if saved is not None:
        calls = saved["calls"]
        bad_calls = {c: [(r, nt, b, v.to(dev)) for r, nt, b, v in segs] for c, segs in saved["bad"].items()}
        compact = analyze(args, AS, W, kern, SCHED, calls, bad_calls, {}, emit, dev)
        emit({"reanalysis": args.reanalyze, "events": compact})
        return 0
    nmax = max(SHAPES[s][0] for s in args.shapes)
    out_flat = torch.empty(M * nmax, device=dev, dtype=torch.bfloat16)
    OUT = {s: out_flat[:M * SHAPES[s][0]].view(M, SHAPES[s][0]) for s in args.shapes}

    # busy work on a second, non-blocking stream
    side = torch.cuda.Stream(device=dev, priority=0)
    nel = args.busy_mb * 2**20 // 4
    bx = torch.randn(nel, device=dev)
    by = torch.randn(nel, device=dev)
    bz = torch.empty(nel, device=dev)
    ma = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    mb = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    mc = torch.empty(2048, 2048, device=dev, dtype=torch.bfloat16)
    ha = torch.randn(8 * 32 * 32, device=dev, dtype=torch.float16)
    hb = torch.randn(8 * 32 * 32, device=dev, dtype=torch.float16)
    hgrid = 4 * gemm.max_clusters
    hc = torch.empty(hgrid * 32 * 32, device=dev, dtype=torch.float32)
    cgrid = 8 * gemm.max_clusters
    cout = torch.empty(cgrid * 128, device=dev, dtype=torch.int32)
    if args.busy == "chase":
        ctab = torch.randint(0, 2**30, (args.busy_mb * 2**18,), device=dev, dtype=torch.int32)
    elif args.busy == "hostpoll":
        ctab = _HostPtr(torch.randint(0, 2**30, (2**16,), dtype=torch.int32).pin_memory())
    tick = [0]

    def busy_one():  # noqa: ANN202
        kind = args.busy
        if kind == "mix":
            kind = "add" if tick[0] % 2 == 0 else "mm"
        tick[0] += 1
        if kind == "add":
            torch.add(bx, by, out=bz)
        elif kind == "mm":
            torch.mm(ma, mb, out=mc)
        elif kind == "smem":
            _smem_hammer[(hgrid,)](ha, hb, hc, 2000, BM=32, BK=32, num_warps=4, num_stages=1)
        elif kind == "chase":
            _chase[(cgrid,)](ctab, cout, 64, ctab.numel(), VOLATILE=False, num_warps=4)
        elif kind == "hostpoll":
            _chase[(cgrid,)](ctab, cout, 16, ctab.t.numel(), VOLATILE=True, num_warps=4)
        elif kind == "sleep":
            torch.cuda._sleep(200_000)

    def ms(fn, n=3):  # noqa: ANN001,ANN202
        fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        best = 1e9
        for _ in range(n):
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1))
        return best

    s0 = args.shapes[0]
    g_ms = ms(lambda: gemm.run(a, W[s0][0], sa, W[s0][1], OUT[s0]))
    b_ms = ms(busy_one, 5) if args.busy != "none" else 0.0
    per_call = args.busy_per_call or (min(100.0, max(1.0, g_ms / b_ms)) if b_ms else 0.0)
    emit({"setup_s": round(time.time() - t0, 1), "ref_s": round(t_ref, 1), "gemm_ms_alone": round(g_ms, 2),
          "busy_ms_alone": round(b_ms, 3), "busy_per_call": round(per_call, 2),
          "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})

    calls, seen = [], 0
    bad_calls = {}
    acc_busy = 0.0
    t_start = time.time()
    i = 0
    while i < args.calls and time.time() - t_start < args.seconds:
        for _ in range(args.window):
            s = args.shapes[i % len(args.shapes)]
            bi = (i // len(args.shapes)) % args.nbuf
            a, sa = AS[bi]
            w, sb = W[s]
            acc_busy += per_call
            with torch.cuda.stream(side):
                while acc_busy >= 1.0:
                    busy_one()
                    acc_busy -= 1.0
            gemm.run(a, w, sa, sb, OUT[s])
            rowtile_check(OUT[s], REF[(s, bi)], wts, cap, 1, i)
            calls.append((s, bi))
            i += 1
        torch.cuda.synchronize()
        n = int(cap[2])
        if n > seen:
            meta = cap[1][seen:min(n, CAP)].cpu().tolist()
            vals = cap[0][seen:min(n, CAP)].clone()
            for j, (cid, row, nt, bits) in enumerate(meta):
                bad_calls.setdefault(cid, []).append((row, nt, bits, vals[j]))
            seen = n
        if i % (args.window * 32) == 0:
            emit({"progress": i, "s": round(time.time() - t_start, 1),
                  "mismatching_calls": int((cap[3][:i] > 0).sum()), "captured": seen})
    torch.cuda.synchronize()
    wall = time.time() - t_start
    flags = cap[3][:len(calls)].cpu()
    bad_ids = flags.nonzero().flatten().tolist()
    seg_per_call = {c: int(flags[c]) for c in bad_ids}

    if args.save and bad_calls:
        torch.save({"calls": calls, "bad": {c: [(r, nt, b, v.cpu()) for r, nt, b, v in segs]
                                            for c, segs in bad_calls.items()}, "args": vars(args)}, args.save)
    compact = analyze(args, AS, W, kern, SCHED, calls, bad_calls, seg_per_call, emit, dev)

    # reference stability: recompute the checksums and compare
    ref_stable = {}
    for (s, b), ref in REF.items():
        r2 = ref_checksums(AS[b][0], W[s][0], AS[b][1], W[s][1], wts, cap, args.ref_chunk)
        ref_stable[f"{s}/{b}"] = bool(torch.equal(r2, ref))
    by_shape = {s: sum(1 for c in calls if c[0] == s) for s in args.shapes}
    bad_by_shape = {s: sum(1 for c in bad_ids if calls[c][0] == s) for s in args.shapes}
    emit({"summary": {"calls": len(calls), "calls_by_shape": by_shape, "mismatching_calls": len(bad_ids),
                      "first_bad_calls": bad_ids[:20], "events": compact,
                      "mismatching_by_shape": bad_by_shape, "captured_segments": seen,
                      "capture_overflow": seen > CAP, "wall_s": round(wall, 1), "ref_stable": ref_stable,
                      "busy": args.busy, "busy_per_call": round(per_call, 2), "ab_stage": kern.ab_stage,
                      "env": env, "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}})
    return 1 if bad_ids else 0


if __name__ == "__main__":
    sys.exit(main())
