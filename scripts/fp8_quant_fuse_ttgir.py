"""Offline compiler check for H3_TP2_FP8_QUANT_FUSE (no GPU needed: ``triton.compile`` with an explicit sm_121 target).

1. The layout of every ``tt.reduce`` operand and every load / store in upstream's modulation kernels
   (``indexed_modulation._rms_norm_indexed_scale_shift_kernel`` / ``_indexed_gate_rms_norm_scale_shift_kernel``) and in
   the fused copies (``fp8_quant_fuse``, both modes): the RMS reduction tree is fixed by the reduce operand's layout.
2. The normalised TTGIR (SSA names and locations stripped) of upstream vs mode 0: identical up to the bf16 rounding
   of the output means the RMS arithmetic is upstream's; only the quant follows.
3. The PTX instruction sequence of the RMS reduction (up to the first rsqrt) of upstream vs both modes.
4. The division / fp8 conversion instructions of every fused kernel (the serving Inductor quant kernel uses
   ``div.full.f32`` and ``cvt.rn.satfinite.e4m3x2.f32``).

Usage (Spark_06, CPU only): CUDA_VISIBLE_DEVICES= ~/h3-long/.venv/bin/python scripts/fp8_quant_fuse_ttgir.py
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3_tp2 import fp8_quant_fuse as fq  # noqa: E402

T = GPUTarget("cuda", 121, 32)
MIN = fq.MIN_SCALE
CE = {"hidden_size": 5376, "eps": 1e-6, "block_n": 8192}


def compile_(fn, sig, ce, warps=8, div16=()):
    attrs = {(list(sig).index(n),): [["tt.divisibility", 16]] for n in div16}   # as at runtime: aligned pointers and
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=ce, attrs=attrs), target=T,  # 16-multiple strides
                       options={"num_warps": warps})
    return k.asm["ttgir"], k.asm["ptx"]


def layouts(name, ir):
    print("==", name)
    lines = ir.splitlines()
    for i, line in enumerate(lines):
        if '"tt.reduce"' in line:
            j = i
            while "}) :" not in lines[j]:
                j += 1
            print("   reduce", lines[j].strip()[-70:])
        elif re.search(r"\btt\.(load|store)\b", line) and "tensor<" in line:
            m = re.findall(r"tensor<\d+x!tt\.ptr<\w+>, #\w+>", line)
            print("   ", "load " if "tt.load" in line else "store", m[0] if m else "")
    for k, v in re.findall(r"^(#\w+) = (#ttg\.blocked<\{[^\n]*\}>)", ir, re.M):
        print("   ", k, v)


def norm_ir(ir):
    out = []
    for line in ir[ir.index("tt.func"):].splitlines():
        t = re.sub(r"loc\([^)]*\)", "", line)
        t = re.sub(r"%[\w.#]+", "%", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t and not t.startswith("#loc"):
            out.append(t)
    return out


def body_until_truncf(lines):
    """The function body (after the constants) up to the bf16 rounding of the output (the last truncf to bf16)."""
    body = [ln for ln in lines if not ln.startswith(("tt.func", "% = arith.constant"))]
    last = max(i for i, ln in enumerate(body) if "arith.truncf" in ln and "xbf16" in ln.split(" to ")[-1])
    return [ln for ln in body[:last + 1] if "tt.addptr" not in ln and "tt.splat %" not in ln
            and "arith.muli" not in ln]


def red_ops(ptx):
    body = ptx[ptx.index("{", ptx.index(".entry")):]
    out = []
    for line in body.splitlines():
        t = line.strip()
        if t.startswith("rsqrt"):
            break
        op = t.split(" ")[0].split("\t")[0]
        if (re.match(r"(fma|mul|add|sub|cvt|mov\.b32)\b", op) and ".f32" in t) or op.startswith(("shfl", "bar", "st.shared", "ld.shared")):
            out.append(op)
    return out


def insns(ptx):
    return (sorted(set(re.findall(r"\b(div\.[a-z0-9.]+|rcp\.[a-z0-9.]+)", ptx))),
            sorted(set(re.findall(r"\b(cvt\.[a-z0-9.]*e4m3[a-z0-9.]*)", ptx))))


def main() -> None:
    from vllm_omni.diffusion.layers import indexed_modulation as im

    up = {"output_ptr": "*bf16", "x_ptr": "*bf16", "weight_ptr": "*bf16", "shift_ptr": "*bf16", "scale_ptr": "*bf16",
          "indices_ptr": "*i64", "hidden_size": "constexpr", "eps": "constexpr", "stride_x_row": "i32",
          "stride_shift_row": "i32", "stride_scale_row": "i32", "stride_indices": "i32", "row_offset": "i32",
          "block_n": "constexpr"}
    up_d = ("output_ptr", "x_ptr", "weight_ptr", "shift_ptr", "scale_ptr", "indices_ptr", "stride_x_row",
            "stride_shift_row", "stride_scale_row")
    ours = {"q_ptr": "*fp8e4nv", "s_ptr": "*fp32", "ph_ptr": "*bf16", **{k: v for k, v in up.items() if k != "output_ptr"
            and k != "block_n"}, "fp8_max": "fp32", "block_n": "constexpr", "MIN_SF": "constexpr", "MAXV": "constexpr",
            "RELOAD": "constexpr"}
    ours_d = ("q_ptr", "s_ptr", "ph_ptr") + up_d[1:]
    up2 = {"residual_out_ptr": "*bf16", "modulated_out_ptr": "*bf16", "residual_ptr": "*bf16", "gate_ptr": "*bf16",
           "branch_ptr": "*bf16", "weight_ptr": "*bf16", "shift_ptr": "*bf16", "scale_ptr": "*bf16",
           "indices_ptr": "*i64", "hidden_size": "constexpr", "eps": "constexpr", "stride_residual_row": "i32",
           "stride_gate_row": "i32", "stride_branch_row": "i32", "stride_shift_row": "i32", "stride_scale_row": "i32",
           "stride_indices": "i32", "row_offset": "i32", "block_n": "constexpr"}
    up2_d = ("residual_out_ptr", "modulated_out_ptr", "residual_ptr", "gate_ptr", "branch_ptr", "weight_ptr",
             "shift_ptr", "scale_ptr", "indices_ptr", "stride_residual_row", "stride_gate_row", "stride_branch_row",
             "stride_shift_row", "stride_scale_row")
    ours2 = {"residual_out_ptr": "*bf16", "q_ptr": "*fp8e4nv", "s_ptr": "*fp32", "ph_ptr": "*bf16",
             **{k: v for k, v in up2.items() if k not in ("residual_out_ptr", "modulated_out_ptr", "block_n")},
             "fp8_max": "fp32", "block_n": "constexpr", "MIN_SF": "constexpr", "MAXV": "constexpr", "RELOAD": "constexpr"}
    ours2_d = ("residual_out_ptr", "q_ptr", "s_ptr", "ph_ptr") + up2_d[2:]
    qce = dict(CE, MIN_SF=MIN, MAXV=448.0)

    pairs = [("rms", im._rms_norm_indexed_scale_shift_kernel, up, up_d, fq._rms_norm_indexed_scale_shift_quant_kernel,
              ours, ours_d),
             ("gate_rms", im._indexed_gate_rms_norm_scale_shift_kernel, up2, up2_d,
              fq._indexed_gate_rms_norm_scale_shift_quant_kernel, ours2, ours2_d)]
    for name, up_fn, up_sig, up_div, fn, sig, div in pairs:
        ir_u, ptx_u = compile_(up_fn, up_sig, CE, div16=up_div)
        layouts(f"upstream {name}", ir_u)
        for rl in (0, 1):
            ir_f, ptx_f = compile_(fn, sig, dict(qce, RELOAD=rl), div16=div)
            layouts(f"fused {name} mode {rl}", ir_f)
            a, b = body_until_truncf(norm_ir(ir_u)), body_until_truncf(norm_ir(ir_f))
            same = a == b
            print(f"   TTGIR up to the bf16 rounding identical to upstream: {same} ({len(a)} / {len(b)} ops)")
            if not same:
                print("\n".join(list(difflib.unified_diff(a, b, lineterm="", n=1))[:60]))
            print(f"   PTX RMS reduction sequence identical to upstream: {red_ops(ptx_u) == red_ops(ptx_f)}"
                  f" ({len(red_ops(ptx_u))} ops)")
            print("   division / fp8 conversion:", insns(ptx_f))
    other = [("unpack_quant", fq._unpack_quant_kernel,
              {"recv_ptr": "*bf16", "q_ptr": "*fp8e4nv", "s_ptr": "*fp32", "rows": "i32", "C": "i32", "fp8_max": "fp32",
               "P": "constexpr", "BLOCK_C": "constexpr", "MIN_SF": "constexpr", "MAXV": "constexpr"},
              {"P": 2, "BLOCK_C": 4096, "MIN_SF": MIN, "MAXV": 448.0}, 8),
             ("silu_mul_quant", fq._silu_mul_quant_kernel,
              {"x_ptr": "*bf16", "q_ptr": "*fp8e4nv", "s_ptr": "*fp32", "d": "i32", "stride_x": "i32", "fp8_max": "fp32",
               "BLOCK": "constexpr", "MIN_SF": "constexpr", "MAXV": "constexpr"},
              {"BLOCK": 16384, "MIN_SF": MIN, "MAXV": 448.0}, 16)]
    for name, fn, sig, ce, w in other:
        _, ptx = compile_(fn, sig, ce, warps=w)
        print(f"== {name}: division / fp8 conversion: {insns(ptx)}, local memory: {'ld.local' in ptx}")


if __name__ == "__main__":
    main()
