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


    def test_several_judges_fail_a_window_only_when_all_hear_the_error(self):
        heard = {"small": ["Dziewcza pisze, że przyszła."], "medium": ["Dziewczyna pisze, że przyszła."]}
        calls = []

        def fake(cfg, wav, language, out_json, model=None):
            calls.append(model)
            return heard[model]

        original, sg.transcribe = sg.transcribe, fake
        try:
            prompts = ["<d>[Polish] Dziewczyna pisze, że przyszła.</d>"]
            one = sg.Guard(sg.Config(max_cer=0.01, model="small"), prompts).score(1, [0.0] * 160, 16000)
            both = sg.Guard(sg.Config(max_cer=0.01, model="small,medium"), prompts).score(1, [0.0] * 160, 16000)
            agree = sg.Guard(sg.Config(max_cer=0.01, model="medium, small"), prompts).score(1, [0.0] * 160, 16000)
        finally:
            sg.transcribe = original
        self.assertFalse(one["ok"])
        self.assertTrue(both["ok"])
        self.assertEqual(both["rates"], [one["cer"], 0.0])
        self.assertEqual(agree["rates"], [0.0])      # the first judge heard it right: the second is not asked
        self.assertEqual(calls, ["small", "small", "medium", "medium"])
        self.assertEqual(sg.models_of(sg.Config(max_cer=0.01, model=" ")), ["small"])

    def test_summary_lists_resampled_windows_and_what_stayed_above_the_limit(self):
        g = sg.Guard(sg.Config(max_cer=0.01))
        self.assertEqual(g.summary(), "no window with spoken lines was judged")
        g.history = [(1, 0.0), (2, 0.0), (3, 0.037), (3, 0.0), (4, 0.046), (4, 0.053), (4, 0.0), (5, 0.0076), (6, 0.02), (6, 0.03), (6, 0.017)]
        self.assertEqual(g.summary(), "6 windows judged, 3 re-sampled (w3 0.037 > 0.000; w4 0.046 > 0.053 > 0.000; w6 0.020 > 0.030 > 0.017), "
                                      "best attempts CER max 0.017, still above 0.01 after every attempt: w6")


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


class MisheardTests(unittest.TestCase):
    def test_the_misheard_words_are_named_with_what_was_heard(self):
        script = ["najpierw", "zaprośmy", "ją", "na", "rozmowę"]
        heard = ["no", "najpierw", "zapróśnio", "na", "rozmowę", "tak"]   # speech around the script is free
        items = sg.misheard(script, heard)
        self.assertEqual([k for k, _w, _g in items], [1, 2])
        self.assertEqual("".join(g for _k, _w, g in items), "zapróśnio")
        self.assertEqual(sg.phrases(items), [("zaprośmy ją", "zapróśnio")])
        self.assertEqual(sg.judge(script, heard, 0.02)["misheard"], items)

    def test_a_moved_word_boundary_is_not_a_slip_and_a_missing_word_is(self):
        script = ["dziewczyna", "pisze", "że", "przyszła"]
        self.assertEqual(sg.misheard(script, ["dziewczy", "napisze", "że", "przyszła"]), [])
        self.assertEqual(sg.judge(script, ["dziewczy", "napisze", "że", "przyszła"], 0.02)["misheard"], [])
        self.assertEqual(sg.misheard(script, ["dziewczyna", "pisze", "przyszła"]), [(2, "że", "")])
        self.assertEqual(sg.misheard([], ["cokolwiek"]), [])

    def test_the_errors_charged_to_words_add_up_to_the_distance(self):
        script = ["widzę", "czynsz", "zalegają", "już", "dwa", "pełne", "miesiące"]
        heard = ["widzę", "czyż", "zalegają", "już", "dwa", "pełne", "mieniące"]
        items = sg.misheard(script, heard)
        self.assertEqual([w for _k, w, _g in items], ["czynsz", "miesiące"])
        self.assertEqual([g for _k, _w, g in items], ["czyż", "mieniące"])
        self.assertEqual(sg.partial_distance("".join(script), "".join(heard)), 4)

    def test_what_was_heard_in_place_of_the_words_rebuilds_the_speech(self):
        cases = [(["a", "jeśli", "przyjdzie", "tylko", "dlatego"], ["jeśli", "przydzie", "tylko", "dla", "tego", "że"]),
                 (["czynsz", "może", "poczekać", "dziewczyna", "nie"], ["czyż", "morze", "poczekać", "dziewczynanie"]),
                 (["sklep", "jest", "jej", "winien", "przeprosiny"], ["sklep", "jej", "winiem", "przeprosiny", "i"])]
        for script, heard in cases:
            items = sg.misheard(script, heard)
            got = {k: g for k, _w, g in items}
            self.assertIn("".join(got.get(k, w) for k, w in enumerate(script)), "".join(heard))
            self.assertTrue(all(w != g for _k, w, g in items))
        self.assertEqual([w for _k, w, _g in sg.misheard(*cases[2])], ["jest", "winien"])

    def test_the_summary_names_the_words_that_failed_in_every_attempt(self):
        g = sg.Guard(sg.Config(max_cer=0.01))
        g.history = [(1, 0.0), (4, 0.046), (4, 0.046), (4, 0.030), (6, 0.02), (6, 0.03)]
        g.slips = {1: [{}], 4: [{20: "zaprośmy", 21: "ją"}, {20: "zaprośmy", 21: "ją", 3: "wyjścia"}, {20: "zaprośmy", 21: "ją"}],
                   6: [{2: "czynsz"}, {9: "cierpliwy"}]}
        self.assertEqual(g.summary(), "3 windows judged, 2 re-sampled (w4 0.046 > 0.046 > 0.030; w6 0.020 > 0.030), best attempts CER max 0.030, "
                                      "still above 0.01 after every attempt: w4 (zaprośmy ją), w6")


if __name__ == "__main__":
    unittest.main()
