#!/usr/bin/env python3
"""Screen a video's speech with Whisper; results are not a listening or lip-sync verdict."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path


def words(text: str) -> list[str]:
    """Compare speech words without case or punctuation differences."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+", text, flags=re.UNICODE)


def word_error_rate(reference: str, transcript: str) -> float:
    """Levenshtein word error rate; a silent transcription against speech is 1."""
    expected, actual = words(reference), words(transcript)
    if not expected:
        raise ValueError("reference must contain at least one word")
    previous = list(range(len(actual) + 1))
    for i, reference_word in enumerate(expected, 1):
        current = [i] + [0] * len(actual)
        for j, actual_word in enumerate(actual, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (reference_word != actual_word))
        previous = current
    return previous[-1] / len(expected)


def assess(source: Path, *, model_name: str, language: str | None, reference: str | None,
           allow_download: bool, threads: int) -> dict:
    from faster_whisper import WhisperModel

    source = source.expanduser().resolve(strict=True)
    if not source.is_file() or threads < 1:
        raise ValueError("source must be a file and threads must be positive")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=threads,
                         local_files_only=not allow_download)
    segments, info = model.transcribe(str(source), language=language, beam_size=5, vad_filter=True,
                                      condition_on_previous_text=False)
    items = [{"start_s": round(item.start, 3), "end_s": round(item.end, 3), "text": item.text.strip(),
              "average_log_probability": round(item.avg_logprob, 4),
              "no_speech_probability": round(item.no_speech_prob, 4)} for item in segments]
    transcript = " ".join(item["text"] for item in items)
    return {"source": str(source), "source_sha256": digest, "model": model_name, "requested_language": language,
            "detected_language": info.language, "detected_language_probability": info.language_probability,
            "transcript": transcript, "reference": reference,
            "word_error_rate": None if reference is None else round(word_error_rate(reference, transcript), 4),
            "segments": items,
            "limitations": "ASR can mishear or hallucinate; review speech, identity, delivery and lip sync separately."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="video or audio with a speech track")
    parser.add_argument("--reference-file", type=Path, help="plain-text intended dialogue for word-error screening")
    parser.add_argument("--language", default=None, help="language code such as pl; omit to detect")
    parser.add_argument("--model", default="small", help="local faster-whisper model name or path")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-download", action="store_true", help="allow model download if absent from cache")
    parser.add_argument("--output", type=Path, help="write JSON to a new file instead of stdout")
    args = parser.parse_args()
    try:
        reference = None if args.reference_file is None else args.reference_file.read_text(encoding="utf-8").strip()
        if reference is not None and not words(reference):
            raise ValueError("reference file is empty")
        result = assess(args.source, model_name=args.model, language=args.language, reference=reference,
                        allow_download=args.allow_download, threads=args.threads)
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(text)
        else:
            print(text, end="")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"speech-qa: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
