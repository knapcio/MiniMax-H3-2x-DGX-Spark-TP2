"""The step-gated audio pin drops the request's audio edit at the configured fraction of the denoise steps."""
import types
import unittest

from h3_tp2 import audio_pin_steps as aps


def state(step, edit="EDIT", n_steps=20):
    return types.SimpleNamespace(step_index=step, extra={aps.STATE_KEY: edit, aps.SIGMAS_KEY: [0.0] * (n_steps + 1)})


class ParseAndRelease(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(aps.parse_fraction(None))
        self.assertIsNone(aps.parse_fraction(""))
        self.assertIsNone(aps.parse_fraction("1"))
        self.assertEqual(aps.parse_fraction("0.3"), 0.3)
        for bad in ("0", "1.5", "-0.1"):
            with self.assertRaises(ValueError):
                aps.parse_fraction(bad)

    def test_release_step_keeps_one_pinned_and_one_free_step(self):
        self.assertEqual(aps.release_step(0.3, 20), 6)
        self.assertEqual(aps.release_step(0.5, 37), 19)
        self.assertEqual(aps.release_step(0.01, 20), 1)
        self.assertEqual(aps.release_step(0.99, 20), 19)

    def test_release_due(self):
        self.assertFalse(aps.release_due(state(5), 0.3))
        self.assertTrue(aps.release_due(state(6), 0.3))
        self.assertTrue(aps.release_due(state(19), 0.3))
        self.assertFalse(aps.release_due(state(6, edit=None), 0.3))        # window 1: nothing pinned
        self.assertFalse(aps.release_due(state(6, n_steps=1), 0.3))


class Install(unittest.TestCase):
    def test_wrapper_clears_the_edit_once_and_calls_upstream(self):
        calls = []

        class Pipeline:
            def denoise_step(self, input_batch, *, states=None, **kwargs):
                calls.append([s.extra[aps.STATE_KEY] for s in (states or input_batch.states)])
                return "out"
        pp = types.SimpleNamespace(MiniMaxH3Pipeline=Pipeline)
        aps.install(pp, 0.3)
        early, late, first = state(2), state(7), state(7, edit=None)
        batch = types.SimpleNamespace(states=[early, late, first])
        self.assertEqual(Pipeline().denoise_step(batch), "out")
        self.assertEqual(calls, [["EDIT", None, None]])
        self.assertEqual(early.extra[aps.STATE_KEY], "EDIT")
        self.assertIsNone(late.extra[aps.STATE_KEY])
        self.assertEqual(Pipeline.denoise_step.__wrapped__.__name__, "denoise_step")


if __name__ == "__main__":
    unittest.main()
