"""Verify opt-in video form limits without starting or patching a live server."""
import unittest
from h3_tp2 import video_form_limit


class VideoFormLimitTests(unittest.TestCase):
    def test_only_implicit_video_post_limit_is_changed(self):
        class Request:
            def __init__(self, method, path):
                self.scope = {"type": "http", "method": method, "path": path}
            def form(self, *args, **kwargs):
                return args, kwargs
        self.assertTrue(video_form_limit.install(4 * video_form_limit.MIB, Request))
        for path in ("/v1/videos", "/v1/videos/sync"):
            self.assertEqual(Request("POST", path).form()[1]["max_part_size"], 4 * video_form_limit.MIB)
        self.assertEqual(Request("GET", "/v1/videos").form()[1], {})
        self.assertEqual(Request("POST", "/other").form()[1], {})
        self.assertEqual(Request("POST", "/v1/videos").form(max_part_size=19)[1]["max_part_size"], 19)
        self.assertFalse(video_form_limit.install(4 * video_form_limit.MIB, Request))
        with self.assertRaises(RuntimeError):
            video_form_limit.install(8 * video_form_limit.MIB, Request)

    def test_invalid_limit_is_rejected(self):
        for value in (0, -1, True, 65 * video_form_limit.MIB, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                video_form_limit.install(value, object)


if __name__ == "__main__":
    unittest.main()
