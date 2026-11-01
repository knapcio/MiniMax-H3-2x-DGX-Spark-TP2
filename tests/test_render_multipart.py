"""Exercise actual curl multipart encoding with punctuation and file-backed audio."""
import base64
import email.parser
import email.policy
import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

REPO = Path(__file__).resolve().parents[1]


class RealMultipartTests(unittest.TestCase):
    def test_prompt_and_json_with_semicolons_are_sent_verbatim(self):
        fields = {}
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                data = self.rfile.read(int(self.headers['Content-Length']))
                header = ('Content-Type: ' + self.headers['Content-Type'] + '\r\nMIME-Version: 1.0\r\n\r\n').encode()
                message = email.parser.BytesParser(policy=email.policy.default).parsebytes(header + data)
                for part in message.iter_parts():
                    fields[part.get_param('name', header='content-disposition')] = part.get_payload(decode=True)
                self.send_response(200); self.end_headers(); self.wfile.write(b'fake video')
        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        server.timeout = 20
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'audio.mp3'; source.write_bytes(b'audio-test' * 15000)
            image = root / 'image.png'; image.write_bytes(b'image-test')
            prompt = 'The woman listens; Wokulski says <d>[Polish] Dzień dobry.</d>'
            extra = {'long_video': True, 'audio_mode': 'lock_source', 'continuation_prompts': [prompt, 'Later; continue.']}
            env = os.environ.copy()
            for key in ('AUDIO_URL', 'H3_AUDIO_MAX_FIELD_BYTES', 'CONT_PROMPTS', 'CONT_PROMPTS_REST', 'CONT_PROMPTS_EXTRA'):
                env.pop(key, None)
            env.update({'H3_API_HOST': '127.0.0.1', 'H3_API_PORT': str(server.server_port),
                        'H3_AUDIO_FILE': str(source), 'EXTRA_JSON': json.dumps(extra), 'TASK': 'fl2va'})
            result = subprocess.run(['bash', str(REPO / 'scripts/render.sh'), str(root / 'out.mp4'),
                                     '30', '960', '544', '20', '3', prompt, str(image), 'first'],
                                    env=env, capture_output=True, text=True, timeout=30)
            thread.join(timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(fields['prompt'].decode(), prompt)
            actual_extra = json.loads(fields['extra_params'])
            self.assertEqual(actual_extra['continuation_prompts'], extra['continuation_prompts'])
            audio = json.loads(fields['audio_reference'])['audio_url'].split(',', 1)[1]
            self.assertEqual(base64.b64decode(audio), source.read_bytes())
            self.assertEqual(fields['input_reference'], image.read_bytes())


if __name__ == '__main__':
    unittest.main()
