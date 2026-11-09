"""Speech guard for continuation windows (opt-in ``H3_TP2_SPEECH_GUARD=<max character error rate>``, not output-preserving).

The chain of a long dialogue adds slips of the tongue the same lines rendered fresh do not make (one or two words per
window from the third window on in Polish, results/20261106-voice/ sections 12-13), and every lever that moves them
(the RoPE time of the window) costs the voice.  This guard leaves the sampling alone and judges the result instead: after
a window is denoised its audio latents are decoded, the waveform is transcribed on the CPU (``scripts/speech_qa.py``,
faster-whisper, in its own interpreter) and the transcript is compared with the ``<d>`` lines written in that window's
prompt.  A window whose script comes out with more than the tolerated character error rate is sampled again with
another seed, like the seam, cut and identity checks in ``cont_match``; the attempt with the lowest score is kept.

The comparison is the character error rate of the script with the word boundaries removed (``dziewczyna pisze`` heard
as ``dziewczy napisze`` is no error) and with free insertions before the first and after the last script character: the
window's audio also carries the hand-over tail of the previous window and whatever the model adds around the lines, and
none of that is the script's fault.  Phonetic respellings written into the prompt are mapped back to the real word with
``H3_TP2_SPEECH_RESPELL=<file>`` of ``prompt_spelling real_word`` lines.  The language comes from the ``[Polish]`` style
tag of the lines (``H3_TP2_SPEECH_LANGUAGE`` overrides it).  A window whose prompt has no spoken lines is not judged.

This module holds the parts that run without a GPU: configuration, the script of a window, the subprocess call, the
distance and the verdict.  The plugin decodes the audio and broadcasts rank 0's verdict.  Cost: one audio decode and
one transcription (3-4 s of CPU for an 11.5 s window with the ``small`` model) per attempt.

Both TP ranks must install the guard whenever the variable is set, because the re-sample decision is a collective
(rank 0 judges, rank 1 waits for the broadcast): a rank that silently ran without it would sample the next window
while the other re-samples this one, and the two ranks hang on mismatched collectives (2026-11-08 queue120, when the
respelling file existed on the head only).  So ``config_from_env`` reads only the environment, which is forwarded to
every rank, and never touches the file system; files are opened by rank 0 at scoring time, and a scoring error
accepts the window on both ranks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from dataclasses import dataclass, field

logger = logging.getLogger("vllm_omni.h3_tp2.speech")
KEEP_WINDOWS = 2  # the wav and transcript of this many windows stay on disk

DIALOGUE = re.compile(r"<d>\s*(?:\[([^\]]*)\])?\s*(.*?)</d>", flags=re.S)
LANGUAGES = {
    "polish": "pl", "english": "en", "german": "de", "french": "fr", "spanish": "es", "italian": "it",
    "portuguese": "pt", "dutch": "nl", "czech": "cs", "slovak": "sk", "ukrainian": "uk", "russian": "ru",
    "japanese": "ja", "chinese": "zh", "mandarin": "zh", "korean": "ko", "swedish": "sv", "norwegian": "no",
    "danish": "da", "finnish": "fi", "hungarian": "hu", "romanian": "ro", "turkish": "tr", "greek": "el",
}


@dataclass(frozen=True)
class Config:
    max_cer: float                 # re-sample when the script's character error rate exceeds this
    python: str = ""               # interpreter with faster-whisper (default ~/speech-venv/bin/python)
    script: str = ""               # scripts/speech_qa.py (default: next to this package)
    model: str = "small"           # faster-whisper model
    language: str = ""             # language code override ("" = the [Language] tag of the lines, else detection)
    threads: int = 6
    respell_file: str = ""         # `prompt_spelling real_word` lines, read by rank 0 when it first scores
    min_words: int = 2             # a window whose script has fewer words is not judged


def config_from_env(env=None) -> Config | None:  # noqa: ANN001
    env = os.environ if env is None else env
    raw = (env.get("H3_TP2_SPEECH_GUARD") or "").strip()
    if not raw or raw == "0":
        return None
    max_cer = float(raw)
    if not 0.0 < max_cer <= 1.0:
        raise ValueError("H3_TP2_SPEECH_GUARD must be a fraction in (0, 1]: the tolerated character error rate of a window's lines")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return Config(
        max_cer=max_cer,
        python=(env.get("H3_TP2_SPEECH_PY") or _default_python()),
        script=(env.get("H3_TP2_SPEECH_SCRIPT") or os.path.join(here, "scripts", "speech_qa.py")),
        model=(env.get("H3_TP2_SPEECH_MODEL") or "small"),
        language=(env.get("H3_TP2_SPEECH_LANGUAGE") or "").strip().lower(),
        threads=max(1, int(env.get("H3_TP2_SPEECH_THREADS") or 6)),
        respell_file=(env.get("H3_TP2_SPEECH_RESPELL") or "").strip(),
        min_words=max(1, int(env.get("H3_TP2_SPEECH_MIN_WORDS") or 2)),
    )


def _default_python() -> str:
    # a fixed default, the same on every rank; whether it exists only matters on the rank that scores
    return os.path.expanduser("~/speech-venv/bin/python")


def load_respellings(path: str) -> dict:
    """``prompt_spelling real_word`` per line, ``#`` comments; keys and values normalised like transcript words."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"respelling file {path}: expected 'prompt_spelling real_word', got {line!r}")
            out[normalise(parts[0])] = normalise(parts[1])
    return out


def normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", normalise(text), flags=re.UNICODE)


def script_of(prompt: str, respellings: dict | None = None) -> tuple[list[str], str | None]:
    """(script words, language code) of one window's prompt: the ``<d>[Language] ...</d>`` lines in order."""
    out: list[str] = []
    language = None
    for tag, text in DIALOGUE.findall(prompt or ""):
        if tag and language is None:
            language = LANGUAGES.get(normalise(tag).strip()) or (normalise(tag).strip() if len(tag.strip()) == 2 else None)
        out.extend((respellings or {}).get(w, w) for w in words(text))
    return out, language


def partial_distance(expected: str, heard: str) -> int:
    """Levenshtein distance of ``expected`` to the best-matching substring of ``heard`` (insertions before and after
    the match are free), so speech around the script costs nothing and a script that was never said costs its length."""
    n, m = len(expected), len(heard)
    if n == 0:
        return 0
    previous = [0] * (m + 1)
    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (expected[i - 1] != heard[j - 1]))
        previous = current
    return min(previous)


def judge(script: list[str], heard: list[str], max_cer: float) -> dict:
    """{"cer", "score", "chars", "ok"}: score = cer / max_cer (> 1.0 fails)."""
    expected = "".join(script)
    distance = partial_distance(expected, "".join(heard))
    cer = distance / max(len(expected), 1)
    return {"cer": round(cer, 4), "distance": distance, "chars": len(expected), "score": cer / max(max_cer, 1e-9),
            "ok": cer <= max_cer}


def write_wav(path: str, waveform, sample_rate: int, target_rate: int = 16000) -> float:  # noqa: ANN001
    """Mono 16-bit PCM at ``target_rate`` from a (channels, samples) or (samples,) float array; returns seconds."""
    import numpy as np

    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0) if x.shape[0] <= 8 else x.mean(axis=1)
    if sample_rate != target_rate and x.size:
        n = int(round(x.size * target_rate / sample_rate))
        x = np.interp(np.linspace(0.0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(target_rate)
        w.writeframes((x * 32767.0).astype("<i2").tobytes())
    return x.size / target_rate


def transcribe(cfg: Config, wav: str, language: str | None, out_json: str) -> list[str]:
    """Run ``speech_qa.py`` on the wav; returns the segment texts in order."""
    if os.path.exists(out_json):
        os.remove(out_json)
    cmd = [cfg.python, cfg.script, wav, "--model", cfg.model, "--threads", str(cfg.threads), "--output", out_json]
    if language:
        cmd += ["--language", language]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"speech_qa failed (rc {res.returncode}): {res.stderr.strip()[-400:]}")
    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)
    return [str(s.get("text", "")) for s in data.get("segments", [])]


class Guard:
    """Per-request state: the prompts (one per window) and the scratch directory."""

    def __init__(self, cfg: Config, prompts: list[str] | None = None):
        self.cfg = cfg
        self.prompts: list[str] = list(prompts or [])
        self.dir = tempfile.mkdtemp(prefix="h3-speech-")
        self.windows = 0
        self.history: list[tuple[int, float]] = []   # (window, cer) of every judged attempt, in order
        self._respellings: dict | None = None

    @property
    def respellings(self) -> dict:
        """The respelling table, read on first use (a missing file raises here, on the scoring rank, never at boot)."""
        if self._respellings is None:
            self._respellings = load_respellings(self.cfg.respell_file) if self.cfg.respell_file else {}
        return self._respellings

    def set_prompts(self, prompts) -> None:  # noqa: ANN001
        self.prompts = [str(p) for p in (prompts or [])]

    def script(self, index: int) -> tuple[list[str], str | None]:
        """Script words and language of window ``index`` (1-based); the last prompt serves windows beyond the list."""
        if not self.prompts:
            return [], None
        prompt = self.prompts[min(max(index, 1), len(self.prompts)) - 1]
        script, language = script_of(prompt, self.respellings)
        return script, (self.cfg.language or language or None)

    def score(self, index: int, waveform, sample_rate: int) -> dict:  # noqa: ANN001
        """Judge window ``index`` from its decoded waveform; {} when there is nothing to judge."""
        script, language = self.script(index)
        if len(script) < self.cfg.min_words:
            return {}
        self.windows += 1
        d = os.path.join(self.dir, f"w{self.windows:03d}")
        os.makedirs(d, exist_ok=True)
        wav = os.path.join(d, "window.wav")
        seconds = write_wav(wav, waveform, sample_rate)
        heard = transcribe(self.cfg, wav, language, os.path.join(d, "speech.json"))
        heard_words = [w for text in heard for w in words(text)]
        out = judge(script, heard_words, self.cfg.max_cer)
        self.history.append((int(index), float(out["cer"])))
        out.update({"window": index, "seconds": round(seconds, 2), "language": language, "script_words": len(script),
                    "heard_words": len(heard_words), "heard": " ".join(heard).strip()[:200], "dir": d})
        logger.info("h3_tp2 speech window %d: %.1f s, %s, script %d words, heard %d, CER %.4f (%d of %d chars), score %.2f: %s",
                    index, seconds, language or "auto", len(script), len(heard_words), out["cer"], out["distance"], out["chars"],
                    out["score"], out["heard"][:120])
        for old in sorted(x for x in os.listdir(self.dir) if x.startswith("w"))[:-KEEP_WINDOWS]:
            shutil.rmtree(os.path.join(self.dir, old), ignore_errors=True)
        return out

    def summary(self) -> str:
        """One line for the end of a request: the attempts of every re-sampled window and the worst of the windows' best rates
        (the retry loop keeps the attempt with the lowest score), e.g.
        ``6 windows judged, 2 re-sampled (w3 0.037 > 0.000; w4 0.046 > 0.053 > 0.000), best attempts CER max 0.009``."""
        if not self.history:
            return "no window with spoken lines was judged"
        per: dict[int, list[float]] = {}
        for window, cer in self.history:
            per.setdefault(window, []).append(cer)
        again = {w: v for w, v in per.items() if len(v) > 1}
        detail = "; ".join(f"w{w} " + " > ".join(f"{c:.3f}" for c in v) for w, v in sorted(again.items()))
        over = sorted(w for w, v in per.items() if min(v) > self.cfg.max_cer)
        text = (f"{len(per)} windows judged, {len(again)} re-sampled" + (f" ({detail})" if detail else "")
                + f", best attempts CER max {max(min(v) for v in per.values()):.3f}")
        if over:
            text += f", still above {self.cfg.max_cer:g} after every attempt: " + ", ".join(f"w{w}" for w in over)
        return text
