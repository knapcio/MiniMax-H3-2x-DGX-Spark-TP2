"""The speech guard's pure-python parts: the script of a window, the free-edge character distance, the verdict and
the configuration (no torch, no whisper)."""
import os
import tempfile
import unittest

from h3_tp2 import speech_guard as sg


class SpeechGuardText(unittest.TestCase):
    def test_script_of_takes_the_lines_and_the_language_tag(self):
        prompt = "A shop. Izabela: <d>[Polish] Widzę czynsz. Zalegają już dwa pełne miesiące.</d> Wokulski: <d>[Polish] Czynsz morze poczekać.</d>"
        script, language = sg.script_of(prompt, {"morze": "może"})
        self.assertEqual(language, "pl")
        self.assertEqual(script, ["widzę", "czynsz", "zalegają", "już", "dwa", "pełne", "miesiące", "czynsz", "może", "poczekać"])
        self.assertEqual(sg.script_of("no lines here"), ([], None))
        self.assertEqual(sg.script_of("<d>[en] Hello there.</d>")[1], "en")

    def test_partial_distance_ignores_speech_around_the_script(self):
        self.assertEqual(sg.partial_distance("czynszmożepoczekać", "iżeczekamynaniączynszmożepoczekaćdziewczynanie"), 0)
        self.assertEqual(sg.partial_distance("czynsz", "czyż"), 3)
        self.assertEqual(sg.partial_distance("abc", ""), 3)
        self.assertEqual(sg.partial_distance("", "anything"), 0)

    def test_a_moved_word_boundary_is_no_error_but_a_slip_is(self):
        script = ["dziewczyna", "pisze", "że", "przyszła", "wczoraj"]
        ok = sg.judge(script, ["dziewczy", "napisze", "że", "przyszła", "wczoraj"], 0.02)
        self.assertEqual((ok["cer"], ok["ok"]), (0.0, True))
        slip = sg.judge(script, ["dziewcza", "pisze", "że", "przyszła", "wczoraj"], 0.02)
        self.assertEqual(slip["distance"], 2)
        self.assertFalse(slip["ok"])
        self.assertGreater(slip["score"], 1.0)
        silent = sg.judge(script, [], 0.02)
        self.assertEqual(silent["cer"], 1.0)

    def test_config_from_env_and_respellings(self):
        self.assertIsNone(sg.config_from_env({}))
        self.assertIsNone(sg.config_from_env({"H3_TP2_SPEECH_GUARD": "0"}))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "respell.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# phonetic spellings\nmorze może\npieniondze pieniądze\n")
            cfg = sg.config_from_env({"H3_TP2_SPEECH_GUARD": "0.02", "H3_TP2_SPEECH_RESPELL": path,
                                      "H3_TP2_SPEECH_LANGUAGE": "PL", "H3_TP2_SPEECH_MODEL": "medium"})
            self.assertEqual((cfg.max_cer, cfg.language, cfg.model), (0.02, "pl", "medium"))
            self.assertEqual(sg.Guard(cfg).respellings, {"morze": "może", "pieniondze": "pieniądze"})
        self.assertTrue(cfg.script.endswith(os.path.join("scripts", "speech_qa.py")))
        with self.assertRaises(ValueError):
            sg.config_from_env({"H3_TP2_SPEECH_GUARD": "1.5"})

    def test_config_never_touches_the_file_system(self):
        # the worker rank has no respelling file and maybe no whisper interpreter: its config must still match rank 0's
        cfg = sg.config_from_env({"H3_TP2_SPEECH_GUARD": "0.01", "H3_TP2_SPEECH_RESPELL": "/nowhere/respell.txt",
                                  "H3_TP2_SPEECH_PY": "/nowhere/python"})
        self.assertEqual((cfg.max_cer, cfg.respell_file, cfg.python), (0.01, "/nowhere/respell.txt", "/nowhere/python"))
        g = sg.Guard(cfg, ["<d>[Polish] Dobrze. A teraz proszę spojrzeć.</d>"])
        with self.assertRaises(FileNotFoundError):
            g.score(1, [0.0] * 160, 16000)  # the scoring rank fails loudly; the plugin accepts the window on both ranks

    def test_guard_script_per_window_and_skips_windows_without_lines(self):
        cfg = sg.Config(max_cer=0.02, min_words=2)
        g = sg.Guard(cfg, ["scene only", "<d>[Polish] Proszę.</d>", "<d>[Polish] Dobrze. A teraz proszę spojrzeć.</d>"])
        self.assertEqual(g.script(1), ([], None))
        self.assertEqual(g.script(2), (["proszę"], "pl"))
        self.assertEqual(g.script(3)[0], ["dobrze", "a", "teraz", "proszę", "spojrzeć"])
        self.assertEqual(g.script(9)[0], g.script(3)[0])  # beyond the list: the last prompt
        self.assertEqual(g.score(1, [0.0] * 160, 16000), {})
        self.assertEqual(g.score(2, [0.0] * 160, 16000), {})  # one word < min_words


class SpeechGuardWav(unittest.TestCase):
    def test_write_wav_downmixes_and_resamples(self):
        import numpy as np
        import wave

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.wav")
            seconds = sg.write_wav(path, np.zeros((2, 32000), dtype=np.float32), 32000)
            with wave.open(path, "rb") as w:
                self.assertEqual((w.getnchannels(), w.getframerate(), w.getnframes()), (1, 16000, 16000))
        self.assertAlmostEqual(seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
