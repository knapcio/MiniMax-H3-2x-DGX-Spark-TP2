#!/usr/bin/env python3
"""Validate and summarize an isolated attention probe, without promoting a backend."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys


BACKENDS = ('flashinfer_fa2', 'flashinfer_fmha_v2_fp16_qk')


def number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected a finite number')
    if positive and value <= 0:
        raise ValueError('expected a positive number')
    return float(value)


def timing(data):
    if not isinstance(data, dict):
        raise ValueError('timing data missing')
    values = data.get('milliseconds')
    if not isinstance(values, list) or not 1 <= len(values) <= 5:
        raise ValueError('timing samples missing')
    samples = [number(value, positive=True) for value in values]
    median = number(data.get('median_ms'), positive=True)
    if not math.isclose(median, statistics.median(samples), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('recorded timing median differs from samples')
    return median


def errors(data, expected):
    if not isinstance(data, dict) or type(data.get('values')) is not int or data['values'] != expected:
        raise ValueError('error statistics have the wrong element count')
    bad = data.get('nonfinite_values')
    if type(bad) is not int or not 0 <= bad <= expected:
        raise ValueError('invalid nonfinite-pair count')
    result = {'nonfinite_pairs': bad}
    for key in ('max_abs', 'mean_abs', 'relative_l2'):
        value = data.get(key)
        if value is not None:
            value = number(value)
            if value < 0:
                raise ValueError('negative error statistic')
        result[key] = value
    if bad < expected and (result['max_abs'] is None or result['mean_abs'] is None):
        raise ValueError('finite comparisons lack error statistics')
    return result


def analyze(data):
    if (not isinstance(data, dict) or data.get('capability') != [12, 1] or not isinstance(data.get('device'), str)
            or not isinstance(data.get('cases'), list) or len(data['cases']) != 4):
        raise ValueError('unexpected probe metadata or case count')
    results = {}
    for case in data['cases']:
        if not isinstance(case, dict) or case.get('heads') != 28 or case.get('head_dim') != 128:
            raise ValueError('unexpected attention shape')
        seq = case.get('sequence'); qk = number(case.get('qk_scale'), positive=True)
        if type(seq) is not int or (seq, qk) not in ((512, 1.), (512, 4.), (512, 16.), (43392, 1.)):
            raise ValueError('unexpected attention case')
        key = (seq, qk)
        if key in results:
            raise ValueError('duplicate attention case')
        cudnn = case.get('cudnn')
        if not isinstance(cudnn, dict):
            raise ValueError('cuDNN result missing')
        baseline = timing(cudnn.get('timing')); n = seq * 28 * 128
        control = errors(cudnn.get('vs_fp32_math'), n) if seq == 512 else None
        backends = case.get('backends')
        if not isinstance(backends, dict) or set(backends) != set(BACKENDS):
            raise ValueError('missing backend result')
        found = {}
        for name in BACKENDS:
            item = backends[name]
            if not isinstance(item, dict) or item.get('status') not in ('ok', 'unavailable'):
                raise ValueError('invalid backend status')
            if item['status'] == 'unavailable':
                found[name] = {'status': 'unavailable', 'reason': str(item.get('error', 'unknown'))[:300]}
                continue
            info = item.get('metadata', {})
            if not isinstance(info, dict):
                raise ValueError('backend metadata missing')
            if (name == 'flashinfer_fa2' and
                    (info.get('backend') != 'fa2' or info.get('fp16_qk_reduction') is not False)):
                raise ValueError('FA2 dispatch or numerical mode was not confirmed')
            if (name == 'flashinfer_fmha_v2_fp16_qk' and
                    (info.get('backend') != 'fmha_v2' or info.get('fp16_qk_reduction') is not True)):
                raise ValueError('FMHA-v2 dispatch or numerical mode was not confirmed')
            measured = timing(item.get('timing', {}))
            ratio = baseline / measured
            if not math.isfinite(ratio):
                raise ValueError('invalid speed ratio')
            result = {'status': 'ok', 'median_ms': measured, 'speed_ratio_vs_cudnn': ratio,
                      'vs_cudnn': errors(item.get('vs_cudnn'), n)}
            if seq == 512:
                result['vs_fp32_math'] = errors(item.get('vs_fp32_math'), n)
            found[name] = result
        results[key] = {'cudnn_median_ms': baseline, 'cudnn_vs_fp32_math': control, 'backends': found}
    return {'device': data['device'], 'torch': data.get('torch'), 'cudnn': data.get('cudnn'), 'cases': results}


def report(summary):
    lines = ['# Isolated H3 attention probe', '', f"Device: {summary['device']}; "
             f"PyTorch: {summary['torch']}; cuDNN: {summary['cudnn']}.", '',
             'The long sequence length is inferred, not measured at the serving attention call. '
             'These synthetic timings do not establish whole-render speed or output quality. '
             'FMHA-v2 uses FP16 QK reduction.', '',
             '| Sequence | Q/K scale | cuDNN median ms | Backend | Median ms | cuDNN/backend | Nonfinite pairs | Relative L2 vs cuDNN |',
             '| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for (seq, qk), case in sorted(summary['cases'].items()):
        for name, result in case['backends'].items():
            if result['status'] == 'unavailable':
                lines.append(f"| {seq} | {qk:g} | {case['cudnn_median_ms']:.3f} | {name} (unavailable) | — | — | — | — |")
            else:
                stats = result['vs_cudnn']; relative = stats['relative_l2']
                display = '—' if relative is None else f'{relative:.6g}'
                lines.append(f"| {seq} | {qk:g} | {case['cudnn_median_ms']:.3f} | {name} | "
                             f"{result['median_ms']:.3f} | {result['speed_ratio_vs_cudnn']:.3f} | "
                             f"{stats['nonfinite_pairs']} | {display} |")
    lines += ['', 'A ratio above 1 means that backend was faster in this isolated sample. '
              'Consult the JSON for all timing samples and FP32 reference comparisons; nonfinite pairs '
              'include either input to the comparison. No serving default has been changed.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    try:
        def reject_constant(value):
            raise ValueError('nonfinite JSON value: ' + value)
        data = json.loads(args.result.read_text(), parse_constant=reject_constant)
        print(report(analyze(data)), end='')
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        print('attention-analysis:', exc, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
