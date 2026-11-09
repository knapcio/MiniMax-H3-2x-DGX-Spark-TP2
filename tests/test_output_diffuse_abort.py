"""The sampling wrapper of h3_output: a sampling that raises (cancelled, or given up early by the speech guard) leaves
no latents file and no latents digest; one that returns keeps both."""

import glob
import os
import tempfile
import types
import unittest
from unittest import mock

import torch

from h3_tp2 import h3_output


def patched_pipeline(behaviour):
    class Pipeline:
        def decode(self, *a, **kw):
            return None

        def encode_prompt(self, *a, **kw):
            return None

        def diffuse(self, **kw):
            return behaviour(**kw)

    module = types.SimpleNamespace(MINIMAX_H3_AUDIO_SAMPLE_RATE=48000, MINIMAX_H3_FPS=24, MiniMaxH3Pipeline=Pipeline)
    h3_output.install(module, None, convert_u8=False)
    return Pipeline()


class DiffuseAbortTests(unittest.TestCase):
    def run_once(self, behaviour):
        with tempfile.TemporaryDirectory() as d:
            env = {"H3_TP2_SAVE_LATENTS": d, "H3_TP2_REPLAY_LATENTS": ""}
            with mock.patch.dict(os.environ, env), self.assertLogs("vllm_omni.h3_tp2", level="INFO") as logs:
                try:
                    out = patched_pipeline(behaviour).diffuse(seed=1)
                except KeyError as e:
                    out = e
            return out, [os.path.basename(p) for p in glob.glob(os.path.join(d, "*.pt"))], "\n".join(logs.output)

    def test_a_sampling_that_raises_saves_and_hashes_nothing(self):
        def give_up(**kw):
            raise KeyError("given up at step 10")

        out, files, log = self.run_once(give_up)
        self.assertIsInstance(out, KeyError)
        self.assertEqual(files, [])
        self.assertIn("phase diffuse", log)
        self.assertNotIn("latents", log)

    def test_a_sampling_that_returns_is_saved_and_hashed(self):
        latents = (torch.zeros(1, 2, 3), torch.ones(1, 2))
        out, files, log = self.run_once(lambda **kw: latents)
        self.assertIs(out, latents)
        self.assertEqual(len(files), 1)
        self.assertIn("latents sha256", log)
        self.assertIn("latents saved", log)


if __name__ == "__main__":
    unittest.main()
