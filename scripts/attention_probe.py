#!/usr/bin/env python3
"""Isolated H3-shaped attention microbenchmark; no serving or default changes.

The FMHA-v2 experiment uses FP16 QK reduction and therefore changes numerical
precision. Synthetic inputs and a kernel microbenchmark do not establish H3
output quality or whole-render speed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


HEADS, HEAD_DIM = 28, 128


def error_stats(reference, candidate, chunk=4096):
    if reference.shape != candidate.shape or reference.ndim < 1 or chunk < 1:
        raise ValueError('attention outputs must have matching shapes and a positive chunk size')
    count = finite = 0
    maximum = total = square = ref_square = 0.0
    for start in range(0, reference.shape[0], chunk):
        a = reference[start:start + chunk].float()
        b = candidate[start:start + chunk].float()
        valid = torch.isfinite(a) & torch.isfinite(b)
        count += valid.numel(); finite += int(valid.sum().item())
        if valid.any().item():
            delta = (a[valid] - b[valid]).abs()
            maximum = max(maximum, float(delta.max().item()))
            total += float(delta.sum().item()); square += float((delta * delta).sum().item())
            ref_square += float((a[valid] * a[valid]).sum().item())
    return {'values': count, 'nonfinite_values': count - finite, 'max_abs': maximum if finite else None,
            'mean_abs': total / finite if finite else None,
            'relative_l2': math.sqrt(square / ref_square) if ref_square > 0 else None}


def fatal_cuda(exc):
    text = str(exc).lower()
    if 'out of memory' in text:
        return False
    return 'cuda error' in text or any(item in text for item in (
        'illegal memory access', 'device-side assert', 'unspecified launch failure',
        'misaligned address', 'device has been lost'))


def timeit(fn, iterations):
    started = time.monotonic(); result = fn(); torch.cuda.synchronize()
    warmup = time.monotonic() - started
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record(); result = fn(); end.record(); torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return result, {'warmup_seconds': warmup, 'milliseconds': samples, 'median_ms': statistics.median(samples)}


def fmha_wrapper(flashinfer, workspace):
    # The SM12x FMHA-v2 dispatch in FlashInfer 0.6.18.post1 is inside the
    # auto-backend branch. Forcing fa2 bypasses it even with FP16 QK enabled.
    return flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout='NHD', backend='auto')


def case(sequence, qk_scale, iterations):
    import flashinfer

    torch.manual_seed(1101)
    q = torch.randn((sequence, HEADS, HEAD_DIM), device='cuda', dtype=torch.bfloat16) * qk_scale
    k = torch.randn_like(q) * qk_scale
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    def cudnn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(
                q.permute(1, 0, 2).unsqueeze(0), k.permute(1, 0, 2).unsqueeze(0),
                v.permute(1, 0, 2).unsqueeze(0), is_causal=False, scale=scale)[0].permute(1, 0, 2).contiguous()
    ref, cudnn_timing = timeit(cudnn, iterations)
    result = {'sequence': sequence, 'heads': HEADS, 'head_dim': HEAD_DIM, 'qk_scale': qk_scale,
              'cudnn': {'timing': cudnn_timing}, 'backends': {}}
    if sequence <= 512:
        with sdpa_kernel([SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                q.float().permute(1, 0, 2).unsqueeze(0), k.float().permute(1, 0, 2).unsqueeze(0),
                v.float().permute(1, 0, 2).unsqueeze(0), is_causal=False, scale=scale)[0].permute(1, 0, 2).contiguous()
        result['cudnn']['vs_fp32_math'] = error_stats(math_ref, ref)
    else:
        math_ref = None

    def record(name, make):
        try:
            fn, metadata = make()
            output, timing = timeit(fn, iterations)
            data = {'status': 'ok', 'timing': timing, 'metadata': metadata, 'vs_cudnn': error_stats(ref, output)}
            if math_ref is not None: data['vs_fp32_math'] = error_stats(math_ref, output)
            result['backends'][name] = data
        except Exception as exc:
            if fatal_cuda(exc): raise
            result['backends'][name] = {'status': 'unavailable', 'error_type': type(exc).__name__, 'error': str(exc)[:400]}
            torch.cuda.synchronize()

    def fa2():
        return (lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False, kv_layout='NHD',
                                                               sm_scale=scale, backend='fa2'),
                {'backend': 'fa2', 'fp16_qk_reduction': False})
    record('flashinfer_fa2', fa2)

    def fmha():
        workspace = torch.empty(128 * 1024**2, dtype=torch.uint8, device='cuda')
        wrapper = fmha_wrapper(flashinfer, workspace)
        indptr = torch.tensor([0, sequence], dtype=torch.int32, device='cuda')
        wrapper.plan(indptr, indptr, HEADS, HEADS, HEAD_DIM, causal=False, sm_scale=scale,
                     q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, use_fp16_qk_reduction=True)
        chosen = str(getattr(wrapper, '_backend', 'unknown'))
        if chosen != 'fmha_v2':
            raise ValueError('expected fmha_v2 dispatch, got ' + chosen)
        return (lambda: wrapper.run(q, k, v), {'backend': chosen, 'fp16_qk_reduction': True})
    record('flashinfer_fmha_v2_fp16_qk', fmha)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sequence', type=int, default=43392)
    parser.add_argument('--iterations', type=int, default=3)
    args = parser.parse_args()
    try:
        if os.path.lexists(args.output) or not 512 <= args.sequence <= 65536 or not 1 <= args.iterations <= 5:
            raise ValueError('output must be new; sequence must be 512..65536 and iterations 1..5')
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
            raise ValueError('the probe requires an isolated SM121 GPU')
        torch.cuda.set_per_process_memory_fraction(0.10)
        results = [case(512, qk, args.iterations) for qk in (1.0, 4.0, 16.0)]
        if args.sequence != 512: results.append(case(args.sequence, 1.0, args.iterations))
        data = {'device': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(), 'cases': results,
                'caveat': ('Synthetic inputs only; the long sequence is inferred from a GEMM shape and must be '
                           'verified against actual attention inputs. FMHA-v2 uses FP16 QK reduction. '
                           'No render-quality or whole-request speed claim follows.')}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.h3-attention-probe-', dir=args.output.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
                stream.flush(); os.fsync(stream.fileno())
            os.link(name, args.output)
        finally:
            os.unlink(name)
        print('saved isolated attention probe', args.output, flush=True)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print('attention-probe:', type(exc).__name__, str(exc)[:600], file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
