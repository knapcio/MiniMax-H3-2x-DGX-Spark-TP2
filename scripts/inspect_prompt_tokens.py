#!/usr/bin/env python3
"""Count the H3 tokenizer's text tokens in one prompt or a continuation plan.

This is a text-only measurement, not a safe model-context or quality limit.
Images, video, audio and presentation markup add tokens; the full model and
runtime can have tighter memory or context limits. No model is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def strings_from_plan(data: object) -> list[str]:
    if not isinstance(data, dict):
        raise ValueError('plan must be a JSON object')
    extra = data.get('extra_params', data)
    if not isinstance(extra, dict):
        raise ValueError('extra_params must be a JSON object')
    prompts = extra.get('continuation_prompts')
    if (not isinstance(prompts, list) or not prompts or len(prompts) > 10000
            or any(not isinstance(x, str) or not x.strip() for x in prompts)):
        raise ValueError('continuation_prompts must contain 1..10000 nonempty strings')
    return prompts


def inspect(prompts: list[str], tokenizer) -> dict:
    if not prompts or any(not isinstance(x, str) or not x.strip() for x in prompts):
        raise ValueError('prompts must be nonempty strings')
    rows = []
    for i, prompt in enumerate(prompts, 1):
        raw = prompt.encode('utf-8')
        # Match the H3 preprocessing path's tokenizer(text, add_special_tokens=False).
        ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        if not isinstance(ids, list) or any(type(x) is not int for x in ids):
            raise ValueError('tokenizer did not return one token-id list')
        rows.append({'window': i, 'utf8_bytes': len(raw), 'characters': len(prompt),
                     'text_tokens': len(ids), 'prompt_sha256': hashlib.sha256(raw).hexdigest()})
    counts = [x['text_tokens'] for x in rows]
    return {'window_count': len(rows), 'windows': rows, 'min_text_tokens': min(counts),
            'max_text_tokens': max(counts), 'sum_text_tokens_across_windows': sum(counts),
            'tokenizer_model_max_length_metadata': getattr(tokenizer, 'model_max_length', None),
            'limitations': ('Text tokens only. Repeated windows are separate encodings, not one continuous context. '
                            'Image/audio tokens and model/runtime limits are not measured. The tokenizer metadata '
                            'is not a verified safe prompt size or a measure of instruction following.')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', type=Path, required=True, help='local H3 tokenizer directory')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt-file', type=Path, help='UTF-8 prompt, including literal newlines')
    source.add_argument('--plan', type=Path, help='planner JSON or extra_params JSON with continuation_prompts')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.prompt_file:
            prompts = [args.prompt_file.read_bytes().decode('utf-8')]
        else:
            prompts = strings_from_plan(json.loads(args.plan.read_bytes().decode('utf-8')))
        from transformers import Qwen2TokenizerFast
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
        result = inspect(prompts, tokenizer)
        result['tokenizer_class'] = type(tokenizer).__name__
        result['tokenizer_json_sha256'] = hashlib.sha256((args.tokenizer_dir / 'tokenizer.json').read_bytes()).hexdigest()
        with args.report.open('x', encoding='utf-8') as report:
            json.dump(result, report, ensure_ascii=False, indent=2)
            report.write('\n')
        print(f"{result['window_count']} window(s), text tokens min/max/sum: "
              f"{result['min_text_tokens']}/{result['max_text_tokens']}/{result['sum_text_tokens_across_windows']}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f'inspect-prompt-tokens: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
