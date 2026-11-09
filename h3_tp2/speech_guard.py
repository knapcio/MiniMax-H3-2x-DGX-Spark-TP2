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
window's audio also carries whatever the model adds around the lines, and none of that is the script's fault.  The
guard is given the part of a window the finished clip keeps (``cont_match.heard_audio``): the leading steps of a
continuation window repeat the previous window's tail and are dropped when the clip is assembled, so a line the
window starts inside them is cut in the clip although the window alone says it whole
(``H3_TP2_SPEECH_WHOLE_WINDOW=1`` listens to the whole window as before 2026-11-09).  Phonetic respellings written into the prompt are mapped back to the real word with
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
    model: str = "small"           # faster-whisper model, or several separated by commas (all must hear the error)
    language: str = ""             # language code override ("" = the [Language] tag of the lines, else detection)
    threads: int = 6
    respell_file: str = ""         # `prompt_spelling real_word` lines, read by rank 0 when it first scores
    min_words: int = 2             # a window whose script has fewer words is not judged
    early_steps: tuple = ()        # MEASUREMENT: also judge the loop's clean audio estimate at these denoising steps
    early_stops: tuple = ()        # ((step, max CER or None), ...): give a failing attempt up at these denoising steps


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
        early_steps=_early_steps(env.get("H3_TP2_SPEECH_EARLY") or ""),
        early_stops=_early_stops(env.get("H3_TP2_SPEECH_EARLY_STOP") or ""),
    )


def _early_steps(text: str) -> tuple:
    from h3_tp2 import speech_early

    return speech_early.parse_steps(text)


def _early_stops(text: str) -> tuple:
    from h3_tp2 import speech_early

    return tuple(speech_early.parse_stops(text).items())


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


def misheard(script: list[str], heard: list[str]) -> list[tuple[int, str, str]]:
    """(index, script word, what was heard in its place) for every script word that did not come out letter for letter.

    The same alignment as ``partial_distance`` (word boundaries ignored, speech before and after the script free), read
    back: every script character is matched, replaced or missing, and a heard character with no partner is charged to
    the script word before it.  Among the alignments of that least distance the one that touches the fewest script
    words is taken, so a word that was not said is reported whole instead of being smeared over the letters it shares
    with a neighbour ("pisze że" heard as "pisze", "czynsz zalegają" heard as "czyż zalegają").  A word that was never
    said comes back with an empty string.
    """
    expected, said = "".join(script), "".join(heard)
    n, m = len(expected), len(said)
    if n == 0:
        return []
    owner = [k for k, word in enumerate(script) for _ in word]   # script character -> word index
    far = (1 << 30, 0)
    # cost[j][f] = (distance, script words touched) of the script so far against a stretch of speech ending at j, with
    # f = 1 when the word of the last script character already has an error; back[i][j][f] = (step, previous f)
    cost = [[(0, 0), far] for _ in range(m + 1)]
    back = []
    for i in range(1, n + 1):
        first = i == 1 or owner[i - 2] != owner[i - 1]   # this character opens a word
        row = [[far, far] for _ in range(m + 1)]
        steps = [[None, None] for _ in range(m + 1)]
        for j in range(m + 1):
            for f in (0, 1):
                was = 0 if first else f
                c = cost[j][f]                          # the script character was not said
                if c[0] < far[0] and (c[0] + 1, c[1] + 1 - was) < row[j][1]:
                    row[j][1], steps[j][1] = (c[0] + 1, c[1] + 1 - was), ("missing", f)
                if j == 0:
                    continue
                c = cost[j - 1][f]                      # matched or replaced
                if c[0] >= far[0]:
                    continue
                if expected[i - 1] == said[j - 1]:
                    if c < row[j][was]:
                        row[j][was], steps[j][was] = c, ("pair", f)
                elif (c[0] + 1, c[1] + 1 - was) < row[j][1]:
                    row[j][1], steps[j][1] = (c[0] + 1, c[1] + 1 - was), ("pair", f)
            if j:
                for f in (0, 1):                        # a heard character with no partner, charged to this word
                    c = row[j - 1][f]
                    if c[0] < far[0] and (c[0] + 1, c[1] + 1 - f) < row[j][1]:
                        row[j][1], steps[j][1] = (c[0] + 1, c[1] + 1 - f), ("extra", f)
        cost = row
        back.append(steps)
    chunks = [[] for _ in script]
    wrong = [False] * len(script)
    i = n
    j, f = min(((j, f) for j in range(m + 1) for f in (0, 1)), key=lambda at: cost[at[0]][at[1]])
    while i > 0:
        step, f_before = back[i - 1][j][f]
        word = owner[i - 1]
        if step == "pair":
            chunks[word].append(said[j - 1])
            wrong[word] = wrong[word] or expected[i - 1] != said[j - 1]
            i, j = i - 1, j - 1
        elif step == "missing":
            wrong[word] = True
            i -= 1
        else:
            chunks[word].append(said[j - 1])
            wrong[word] = True
            j -= 1
        f = f_before
    return [(k, script[k], "".join(reversed(chunks[k]))) for k in range(len(script)) if wrong[k]]


def phrases(items: list[tuple[int, str, str]]) -> list[tuple[str, str]]:
    """Neighbouring misheard words joined: [(3, "zaprośmy", "zapróś"), (4, "ją", "nio")] -> [("zaprośmy ją", "zapróśnio")]."""
    out: list[list] = []
    last = None
    for index, word, got in items:
        if last is not None and index == last + 1:
            out[-1][0] += " " + word
            out[-1][1] += got
        else:
            out.append([word, got])
        last = index
    return [(a, b) for a, b in out]


def judge(script: list[str], heard: list[str], max_cer: float) -> dict:
    """{"cer", "score", "chars", "ok", "misheard"}: score = cer / max_cer (> 1.0 fails); misheard as ``misheard``."""
    expected = "".join(script)
    distance = partial_distance(expected, "".join(heard))
    cer = distance / max(len(expected), 1)
    return {"cer": round(cer, 4), "distance": distance, "chars": len(expected), "score": cer / max(max_cer, 1e-9),
            "ok": cer <= max_cer, "misheard": misheard(script, heard) if distance else []}


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


def models_of(cfg: Config) -> list[str]:
    """The judges: ``H3_TP2_SPEECH_MODEL`` may list several models (``small,medium``)."""
    return [m.strip() for m in cfg.model.split(",") if m.strip()] or ["small"]


def transcribe(cfg: Config, wav: str, language: str | None, out_json: str, model: str | None = None) -> list[str]:
    """Run ``speech_qa.py`` on the wav with one model; returns the segment texts in order."""
    if os.path.exists(out_json):
        os.remove(out_json)
    cmd = [cfg.python, cfg.script, wav, "--model", model or models_of(cfg)[0], "--threads", str(cfg.threads), "--output", out_json]
    if language:
        cmd += ["--language", language]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        raise RuntimeError(f"speech_qa failed (rc {res.returncode}): {res.stderr.strip()[-400:]}")
    with open(out_json, encoding="utf-8") as f:
        data = json.load(f)
    return [str(s.get("text", "")) for s in data.get("segments", [])]


def judge_wav(cfg: Config, script: list[str], language: str | None, wav: str, directory: str,
              max_cer: float | None = None) -> tuple[dict, list[str], list[str]]:
    """(verdict, heard segment texts, heard words) of one window's wav against its script.

    With several judges (``Config.model`` = ``small,medium``) the window fails only when every one of them hears the
    lines wrong, so the verdict is the lowest rate and a judge that hears them right settles it: two Whisper sizes
    disagree by 1-4 characters on about a third of the windows (2026-11-09), which is the noise of a single judge.
    ``verdict["rates"]`` lists the rate of every judge that was asked.
    """
    limit = cfg.max_cer if max_cer is None else max_cer
    out, heard, heard_words, rates = None, [], [], []
    for model in models_of(cfg):
        texts = transcribe(cfg, wav, language, os.path.join(directory, f"speech-{model.replace('/', '_')}.json"), model)
        tokens = [w for text in texts for w in words(text)]
        verdict = judge(script, tokens, limit)
        rates.append(verdict["cer"])
        if out is None or verdict["cer"] < out["cer"]:
            out, heard, heard_words = verdict, texts, tokens
        if verdict["ok"]:
            break
    out["rates"] = rates
    return out, heard, heard_words


class Guard:
    """Per-request state: the prompts (one per window) and the scratch directory."""

    def __init__(self, cfg: Config, prompts: list[str] | None = None):
        self.cfg = cfg
        self.prompts: list[str] = list(prompts or [])
        self.dir = tempfile.mkdtemp(prefix="h3-speech-")
        self.windows = 0
        self.history: list[tuple[int, float]] = []   # (window, cer) of every judged attempt, in order
        self.slips: dict[int, list[dict[int, str]]] = {}   # window -> per attempt {script word index: word} misheard
        self.attempts: dict[int, list[str]] = {}   # window -> every attempt in order, as the summary prints it
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
        out, heard, heard_words = judge_wav(self.cfg, script, language, wav, d)
        self.history.append((int(index), float(out["cer"])))
        self.attempts.setdefault(int(index), []).append(f"{float(out['cer']):.3f}")
        self.slips.setdefault(int(index), []).append({k: word for k, word, _got in out.get("misheard", [])})
        wrong = "; ".join(f"{a} > {b or '-'}" for a, b in phrases(out.get("misheard", [])))
        out.update({"window": index, "seconds": round(seconds, 2), "language": language, "script_words": len(script),
                    "heard_words": len(heard_words), "heard": " ".join(heard).strip()[:200], "dir": d})
        logger.info("h3_tp2 speech window %d: %.1f s, %s, script %d words, heard %d, CER %.4f (%d of %d chars), score %.2f: %s%s",
                    index, seconds, language or "auto", len(script), len(heard_words), out["cer"], out["distance"], out["chars"],
                    out["score"], out["heard"][:120], f" | misheard: {wrong}" if wrong else "")
        for old in sorted(x for x in os.listdir(self.dir) if x.startswith("w"))[:-KEEP_WINDOWS]:
            shutil.rmtree(os.path.join(self.dir, old), ignore_errors=True)
        return out

    def peek(self, index: int, waveform, sample_rate: int, tag: str = "early", max_cer: float | None = None) -> dict:  # noqa: ANN001
        """Judge a waveform against the lines of window ``index`` without recording it: nothing enters the summary, the
        stubborn words or the window count.  The wav and transcripts stay in ``<dir>/<tag>`` ({} without lines).
        ``max_cer`` replaces the guard's limit for this verdict."""
        script, language = self.script(index)
        if len(script) < self.cfg.min_words:
            return {}
        d = os.path.join(self.dir, tag)
        os.makedirs(d, exist_ok=True)
        wav = os.path.join(d, "window.wav")
        seconds = write_wav(wav, waveform, sample_rate)
        out, heard, heard_words = judge_wav(self.cfg, script, language, wav, d, max_cer=max_cer)
        out.update({"window": index, "seconds": round(seconds, 2), "language": language, "script_words": len(script),
                    "heard_words": len(heard_words), "heard": " ".join(heard).strip()[:200], "dir": d})
        return out

    def given_up(self, index: int, step: int, cer: float) -> None:
        """Record an attempt of window ``index`` that was stopped at denoising step ``step`` (for the summary)."""
        self.attempts.setdefault(int(index), []).append(f"{float(cer):.3f} at step {int(step)}")

    def summary(self) -> str:
        """One line for the end of a request: the attempts of every re-sampled window and the worst of the windows' best rates
        (the retry loop keeps the attempt with the lowest score), e.g.
        ``6 windows judged, 2 re-sampled (w3 0.037 > 0.000; w4 0.046 > 0.053 > 0.000), best attempts CER max 0.009``."""
        if not self.history:
            return "no window with spoken lines was judged"
        per: dict[int, list[float]] = {}
        for window, cer in self.history:
            per.setdefault(window, []).append(cer)
        shown = {w: self.attempts.get(w) or [f"{c:.3f}" for c in v] for w, v in per.items()}
        again = {w: v for w, v in shown.items() if len(v) > 1}
        detail = "; ".join(f"w{w} " + " > ".join(v) for w, v in sorted(again.items()))
        over = sorted(w for w, v in per.items() if min(v) > self.cfg.max_cer)
        text = (f"{len(per)} windows judged, {len(again)} re-sampled" + (f" ({detail})" if detail else "")
                + f", best attempts CER max {max(min(v) for v in per.values()):.3f}")
        if over:
            text += f", still above {self.cfg.max_cer:g} after every attempt: " + ", ".join(f"w{w}{self.stubborn(w)}" for w in over)
        return text

    def stubborn(self, window: int) -> str:
        """`` (word word)``: the script words of ``window`` that came out wrong in every attempt, the ones to rewrite or
        respell in the prompt; empty when the attempts slipped on different words (or nothing was recorded)."""
        attempts = self.slips.get(window) or []
        if not attempts:
            return ""
        common = set(attempts[0])
        for attempt in attempts[1:]:
            common &= set(attempt)
        return f" ({' '.join(attempts[0][k] for k in sorted(common))})" if common else ""
