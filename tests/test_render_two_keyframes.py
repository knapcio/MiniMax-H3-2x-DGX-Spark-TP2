"""Actual local multipart capture; no H3 service or model is contacted."""
import email.parser
import email.policy
import base64
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

REPO = Path(__file__).resolve().parents[1]


class TwoKeyframeTransport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.parts = []; self.requests = 0; owner = self
        self.first = self.root / 'first; "quoted".png'; self.first.write_bytes(b'first-image')
        self.last = self.root / 'last; "quoted".jpg'; self.last.write_bytes(b'last-image')
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                prefix = ('Content-Type: ' + self.headers['Content-Type'] + '\r\nMIME-Version: 1.0\r\n\r\n').encode()
                message = email.parser.BytesParser(policy=email.policy.default).parsebytes(prefix + raw)
                owner.parts = [(p.get_param('name', header='content-disposition'), p.get_filename(), p.get_payload(decode=True))
                               for p in message.iter_parts()]
                owner.requests += 1
                # Timing must refer to sent snapshots even if original files change.
                owner.first.write_bytes(b'first-after-request'); owner.last.write_bytes(b'last-after-request')
                self.send_response(200); self.send_header('Content-Length', '10'); self.end_headers(); self.wfile.write(b'fake video')
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .05}, daemon=True); thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close(), thread.join(timeout=5)))
        self.env = os.environ.copy()
        for key in ('EXTRA_JSON', 'H3_PROMPT_FILE', 'H3_EXTRA_JSON_FILE', 'CONT_PROMPTS', 'CONT_PROMPTS_EXTRA',
                    'CONT_PROMPTS_REST', 'H3_END_IMAGE_FILE', 'H3_LORA_JSON', 'H3_AUDIO_FILE', 'AUDIO_URL', 'TASK'):
            self.env.pop(key, None)
        self.env.update(H3_API_HOST='127.0.0.1', H3_API_PORT=str(self.server.server_port),
                        H3_END_IMAGE_FILE=str(self.last), H3_RENDER_TIMEOUT_S='10', TMPDIR=str(self.root))

    def render(self, *, first=None, where='both', seconds='10', extra=None, **updates):
        env = dict(self.env, EXTRA_JSON=json.dumps(extra or {}), **updates)
        first = str(self.first) if first is None else first
        return subprocess.run(['bash', str(REPO / 'scripts/render.sh'), str(self.root / 'out.mp4'), seconds,
                               '960', '544', '20', '3101', 'He puts the letter down.', first, where],
                              env=env, capture_output=True, text=True, timeout=20)

    def test_real_http_sends_two_ordered_snapshots_and_hashes(self):
        result = self.render(extra={'long_video': True})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.requests, 1)
        references = [(filename, data) for name, filename, data in self.parts if name == 'input_references']
        self.assertEqual([p[1] for p in references], [b'first-image', b'last-image'])
        self.assertTrue(references[0][0].endswith('.png')); self.assertTrue(references[1][0].endswith('.jpg'))
        self.assertFalse(any(name == 'input_reference' for name, _, _ in self.parts))
        extra = json.loads(next(data for name, _, data in self.parts if name == 'extra_params'))
        self.assertEqual(extra['frame_indices'], [0, -1]); self.assertEqual(extra['task'], 'fl2va')
        timing = json.loads((self.root / 'out.mp4.timing.json').read_text())
        self.assertEqual(timing['keyframe_snapshots']['first']['sha256'], hashlib.sha256(b'first-image').hexdigest())
        self.assertEqual(timing['keyframe_snapshots']['last']['sha256'], hashlib.sha256(b'last-image').hexdigest())
        self.assertEqual(timing['keyframe_snapshots']['aligned_frames'], 243)
        self.assertEqual(timing['keyframe_snapshots']['requested_frames'], 240)
        self.assertEqual(timing['keyframe_snapshots']['alignment_tail_frames'], 3)
        self.assertFalse(list(self.root.glob('h3-text.*')))

    def test_invalid_pairs_fail_before_http(self):
        cases = [dict(first=''), dict(where='first'), dict(seconds='12'),
                 dict(extra={'task': 'ref2va'}), dict(extra={'frame_indices': [0]}),
                 dict(extra={'frame_indices': [False, -1]}),
                 dict(extra={'target': {'duration_seconds': 30}}),
                 dict(extra={'target': {'duration_seconds': None}})]
        for case in cases:
            with self.subTest(case=case):
                result = self.render(**case)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.requests, 0)

    def test_two_images_and_file_backed_audio_are_all_sent(self):
        source = self.root / 'voice.wav'; source.write_bytes(b'fake-audio-for-transport')
        result = self.render(extra={'audio_mode': 'lock_source'}, H3_AUDIO_FILE=str(source))
        self.assertEqual(result.returncode, 0, result.stderr)
        refs = [data for name, _, data in self.parts if name == 'input_references']
        self.assertEqual(refs, [b'first-image', b'last-image'])
        field = next(data for name, _, data in self.parts if name == 'audio_reference')
        encoded = json.loads(field)['audio_url'].split(',', 1)[1]
        self.assertEqual(base64.b64decode(encoded), b'fake-audio-for-transport')

    def test_fifo_keyframe_does_not_block_or_submit(self):
        self.first.unlink(); os.mkfifo(self.first)
        result = self.render()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.requests, 0)


if __name__ == '__main__':
    unittest.main()
