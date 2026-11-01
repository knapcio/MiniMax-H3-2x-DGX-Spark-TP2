"""Exercise multipart parsing through FastAPI when optional test dependencies exist."""
import unittest

try:
    from fastapi import FastAPI, Form
    from fastapi.testclient import TestClient
    from starlette.requests import Request
except ImportError:
    FastAPI = None

from h3_tp2 import video_form_limit


@unittest.skipIf(FastAPI is None, "FastAPI test dependencies not installed")
class RealMultipartTests(unittest.TestCase):
    def test_video_field_limit_increases_without_changing_other_routes(self):
        # Save and restore the class method so this test does not affect other tests.
        original = Request.form
        self.addCleanup(setattr, Request, "form", original)
        app = FastAPI()

        @app.post("/v1/videos/sync")
        @app.post("/other")
        async def endpoint(audio_reference: str = Form(...)):
            return {"length": len(audio_reference)}

        data = "a" * (video_form_limit.MIB + 1024)
        with TestClient(app) as client:
            send = lambda path: client.post(path, files={"audio_reference": (None, data)})
            self.assertEqual(send("/v1/videos/sync").status_code, 400)
            video_form_limit.install(2 * video_form_limit.MIB)
            response = send("/v1/videos/sync")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["length"], len(data))
            self.assertEqual(send("/other").status_code, 400)


if __name__ == "__main__":
    unittest.main()
