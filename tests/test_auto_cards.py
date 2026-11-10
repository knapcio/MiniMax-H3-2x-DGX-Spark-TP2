"""CPU tests of h3_tp2/auto_cards.py: when the server adds face crops to a Ref2VA request, and that it never breaks one."""
import logging
import types
import unittest

from PIL import Image

from h3_tp2 import auto_cards as ac


def scene():
    return Image.new("RGB", (64, 32), (10, 20, 30))


class ParseTests(unittest.TestCase):
    def test_off(self):
        for text in (None, "", "0", "off", "no"):
            self.assertIsNone(ac.parse_max(text))

    def test_values(self):
        self.assertEqual(ac.parse_max("2"), 2)
        self.assertEqual(ac.parse_max(" 8 "), 8)
        for text in ("9", "-1", "two"):
            with self.assertRaises(ValueError):
                ac.parse_max(text)


class SingleImageTests(unittest.TestCase):
    def test_one_image(self):
        self.assertEqual(ac.single_image({"multi_modal_data": {"image": "a.png"}}), "a.png")
        self.assertEqual(ac.single_image({"multi_modal_data": {"image": ["a.png"]}}), "a.png")

    def test_left_alone(self):
        self.assertIsNone(ac.single_image("just text"))
        self.assertIsNone(ac.single_image({"prompt": "x"}))
        self.assertIsNone(ac.single_image({"multi_modal_data": {"image": ["a.png", "b.png"]}}))
        self.assertIsNone(ac.single_image({"multi_modal_data": {"image": "a.png", "video": ["v.mp4"]}}))

    def test_with_cards_appends_and_copies(self):
        prompt = {"prompt": "p", "multi_modal_data": {"image": "a.png", "audio": "x.wav"}}
        c1, c2 = scene(), scene()
        out = ac.with_cards(prompt, [c1, c2])
        self.assertEqual(out["multi_modal_data"]["image"], ["a.png", c1, c2])
        self.assertEqual(out["multi_modal_data"]["audio"], "x.wav")
        self.assertEqual(prompt["multi_modal_data"]["image"], "a.png")   # the original is untouched


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.mod = types.SimpleNamespace()

        def prepare(prompt, sampling, *, task=None, **kw):
            self.calls.append((prompt, task))
            return "prepared"

        self.mod.prepare_encoder_inputs = prepare
        self.cards = [Image.new("RGB", (512, 700)), Image.new("RGB", (512, 760))]
        self.made = []

        def maker(image, max_cards):
            self.made.append((image.size, max_cards))
            return self.cards[:max_cards]

        ac.install(self.mod, 2, loader=lambda v: [scene()], maker=maker)
        ac.install(self.mod, 2, loader=lambda v: [scene()], maker=maker)   # once

    def test_ref2va_with_one_image_gets_the_cards(self):
        with self.assertLogs(ac.logger, level="INFO") as logs:
            out = self.mod.prepare_encoder_inputs({"prompt": "p", "multi_modal_data": {"image": "a.png"}}, None, task="ref2va")
        self.assertEqual(out, "prepared")
        prompt, task = self.calls[-1]
        self.assertEqual(prompt["multi_modal_data"]["image"], ["a.png", *self.cards])
        self.assertEqual(self.made, [((64, 32), 2)])
        self.assertIn("2 face crop(s) added", logs.output[0])
        self.assertIn("512x700, 512x760", logs.output[0])

    def test_other_tasks_and_shapes_are_left_alone(self):
        for prompt, task in (({"prompt": "p", "multi_modal_data": {"image": "a.png"}}, "fl2va"),
                             ({"prompt": "p"}, "t2va"),
                             ({"prompt": "p", "multi_modal_data": {"image": ["a.png", "b.png"]}}, "ref2va"),
                             ({"prompt": "p", "multi_modal_data": {"image": "a.png", "video": ["v.mp4"]}}, "ref2va")):
            self.mod.prepare_encoder_inputs(prompt, None, task=task)
            self.assertEqual(self.calls[-1], (prompt, task))
        self.assertEqual(self.made, [])

    def test_no_face_adds_nothing(self):
        mod = types.SimpleNamespace(prepare_encoder_inputs=lambda prompt, sampling, *, task=None, **kw: prompt)
        ac.install(mod, 2, loader=lambda v: [scene()], maker=lambda image, n: [])
        with self.assertLogs(ac.logger, level="INFO") as logs:
            out = mod.prepare_encoder_inputs({"prompt": "p", "multi_modal_data": {"image": "a.png"}}, None, task="ref2va")
        self.assertEqual(out["multi_modal_data"]["image"], "a.png")
        self.assertIn("no face found", logs.output[0])

    def test_a_failure_never_breaks_the_request(self):
        def boom(image, n):
            raise RuntimeError("detector exploded")
        mod = types.SimpleNamespace(prepare_encoder_inputs=lambda prompt, sampling, *, task=None, **kw: prompt)
        ac.install(mod, 2, loader=lambda v: [scene()], maker=boom)
        with self.assertLogs(ac.logger, level="WARNING") as logs:
            out = mod.prepare_encoder_inputs({"prompt": "p", "multi_modal_data": {"image": "a.png"}}, None, task="ref2va")
        self.assertEqual(out["multi_modal_data"]["image"], "a.png")
        self.assertIn("detector exploded", logs.output[0])


class MakeCardsTests(unittest.TestCase):
    def test_runs_the_script_with_given_boxes_free_of_the_detector(self):
        # a stand-in script: writes two cards like make_character_cards.py does, so the parsing and ordering is tested
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "cards.py")
            with open(script, "w") as handle:
                handle.write("import sys, os\nfrom PIL import Image\nout=sys.argv[sys.argv.index('--out-dir')+1]\n"
                             "os.makedirs(out, exist_ok=True)\n"
                             "Image.new('RGB',(512,600)).save(os.path.join(out,'card_face1.png'))\n"
                             "Image.new('RGB',(512,700)).save(os.path.join(out,'card_face0.png'))\n"
                             "Image.new('RGB',(512,800)).save(os.path.join(out,'card_face10.png'))\n"
                             "print('H3_REF_IMAGES=x')\n")
            import sys
            cards = ac.make_cards(scene(), 2, python=sys.executable, script=script)
            self.assertEqual([c.size for c in cards], [(512, 700), (512, 600)])
            with self.assertRaises(RuntimeError):   # a broken script is an error the installer turns into a warning
                ac.make_cards(scene(), 2, python=sys.executable, script=script.replace("cards.py", "missing.py"))

    def test_no_face_exit_code_means_no_cards(self):
        import os
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "nf.py")
            with open(script, "w") as handle:
                handle.write("import sys\nprint('no face found', file=sys.stderr)\nsys.exit(1)\n")
            self.assertEqual(ac.make_cards(scene(), 2, python=sys.executable, script=script), [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
