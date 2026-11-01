"""Real HTTP multipart tests above Linux's per-argument exec limit; CPU only."""
import email.parser
import email.policy
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

REPO = Path(__file__).resolve().parents[1]


class TextFileRenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parts = {}
        self.received = 0
        self.mutate_sources = False
        test = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_args):
                pass

            def do_POST(self):
                data = self.rfile.read(int(self.headers['Content-Length']))
                header = ('Content-Type: ' + self.headers['Content-Type'] + '\r\nMIME-Version: 1.0\r\n\r\n').encode()
                message = email.parser.BytesParser(policy=email.policy.default).parsebytes(header + data)
                for part in message.iter_parts():
                    test.parts[part.get_param('name', header='content-disposition')] = {
                        'data': part.get_payload(decode=True), 'filename': part.get_filename()}
                test.received += 1
                if test.mutate_sources:
                    (test.root / 'prompt;input.txt').write_text('changed after receiving request')
                    (test.root / 'extra;input.json').write_text('no longer JSON')
                self.send_response(200)
                self.send_header('Content-Length', '10')
                self.end_headers()
                self.wfile.write(b'fake video')

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), self.server.server_close(), thread.join(timeout=5)))
        bindir = self.root / 'bin'
        bindir.mkdir()
        # Audit actual Python/curl exec boundaries, including timing emission.
        # Mac also rejects a regression here even though its exec limit differs.
        self.audit = self.root / 'exec-audit.jsonl'
        for program, executable in [('python3', sys.executable), ('curl', shutil.which('curl'))]:
            path = bindir / program
            path.write_text(f'''#!{sys.executable}
import json, os, sys
row = {{"program": {program!r}, "max_argument_bytes": max(len(os.fsencode(s)) for s in sys.argv),
       "max_environment_bytes": max(len(os.fsencode(k + "=" + v)) for k, v in os.environ.items()),
       "extra_json_exported": "EXTRA_JSON" in os.environ}}
with open(os.environ["H3_EXEC_AUDIT"], "a") as stream:
    stream.write(json.dumps(row) + "\\n")
if max(row["max_argument_bytes"], row["max_environment_bytes"]) >= 65536:
    raise SystemExit("oversized exec argument/environment in text-file transport")
os.execv({executable!r}, [{executable!r}, *sys.argv[1:]])
''')
            path.chmod(0o755)
        self.env = os.environ.copy()
        for name in ('EXTRA_JSON', 'H3_PROMPT_FILE', 'H3_EXTRA_JSON_FILE', 'H3_TEXT_MAX_FIELD_BYTES',
                     'CONT_PROMPTS', 'CONT_PROMPTS_REST', 'CONT_PROMPTS_EXTRA', 'AUDIO_URL', 'H3_AUDIO_FILE',
                     'H3_LORA_JSON', 'H3_AUDIO_FLOW_SHIFT', 'H3_VIDEO_FLOW_SHIFT', 'TASK'):
            self.env.pop(name, None)
        temporary = self.root / 'temp; "quoted",comma\\path'
        temporary.mkdir()
        self.env.update(H3_VENV=str(self.root), H3_API_HOST='127.0.0.1', H3_API_PORT=str(self.server.server_port),
                        H3_EXEC_AUDIT=str(self.audit), TMPDIR=str(temporary), H3_RENDER_TIMEOUT_S='15')

    def render(self, prompt=b'', extra=None, *, duration=30, positional='', **environment):
        source = self.root / 'prompt;input.txt'
        source.write_bytes(prompt)
        params = self.root / 'extra;input.json'
        params.write_bytes(json.dumps(extra or {}, ensure_ascii=False).encode())
        image = self.root / 'image; "test".png'
        image.write_bytes(b'image-content-for-upload')
        env = dict(self.env, H3_PROMPT_FILE=str(source), H3_EXTRA_JSON_FILE=str(params), **environment)
        result = subprocess.run(['bash', str(REPO / 'scripts/render.sh'), str(self.root / 'out.mp4'),
                                 str(duration), '960', '544', '20', '3101', positional, str(image), 'first'],
                                env=env, capture_output=True, timeout=30)
        return result

    def verify_transport(self, prompt):
        self.assertEqual(self.received, 1)
        timing = json.loads((self.root / 'out.mp4.timing.json').read_bytes())
        self.assertEqual(hashlib.sha256(self.parts['prompt']['data']).hexdigest(), hashlib.sha256(prompt).hexdigest())
        self.assertEqual(timing['prompt'].encode('utf-8'), prompt)
        for field in ('prompt', 'extra_params'):
            data = self.parts[field]['data']
            self.assertEqual(timing['text_fields'][field], {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            self.assertIsNone(self.parts[field]['filename'])  # <file is text, not an uploaded file.
        extra = json.loads(self.parts['extra_params']['data'])
        self.assertEqual(timing['extra_params'], extra)
        self.assertEqual(self.parts['input_reference']['data'], b'image-content-for-upload')
        self.assertIsNotNone(self.parts['input_reference']['filename'])
        audit = [json.loads(line) for line in self.audit.read_text().splitlines()]
        self.assertTrue(any(row['program'] == 'curl' for row in audit))
        self.assertGreaterEqual(sum(row['program'] == 'python3' for row in audit), 4)
        self.assertTrue(all(row['max_argument_bytes'] < 65536 and row['max_environment_bytes'] < 65536 for row in audit))
        self.assertFalse(any(row['extra_json_exported'] for row in audit))
        temporary = Path(self.env['TMPDIR'])
        self.assertFalse(list(temporary.glob('h3-text.*')))
        return timing, extra

    def test_large_prompt_and_explicit_extra_keep_utf8_semicolons_and_newlines(self):
        self.mutate_sources = True  # timing must read private snapshots, not these changed source files.
        prompt = ('Zażółć gęślą; <d>[Polish] Dzień dobry.</d>\r\n' * 4000 + '\n\n').encode('utf-8')
        explicit = [prompt.decode(), 'Second; scene.\n', 'Third; scene.\n']
        result = self.render(prompt, {'long_video': True, 'continuation_prompts': explicit}, CONT_PROMPTS='auto')
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        _, sent = self.verify_transport(prompt)
        self.assertEqual(sent['continuation_prompts'], explicit)
        self.assertGreater(len(self.parts['prompt']['data']), 131072)
        self.assertGreater(len(self.parts['extra_params']['data']), 131072)

    def test_auto_29_prompts_remain_in_files_through_curl_and_timing(self):
        prompt = ('Zażółć; camera continues.\n' * 5000).encode('utf-8')
        result = self.render(prompt, {'long_video': True}, duration=300, CONT_PROMPTS='auto',
                             H3_TEXT_MAX_FIELD_BYTES=str(8 * 1024 * 1024))
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        timing, sent = self.verify_transport(prompt)
        self.assertEqual(timing['effective_duration'], 300.0)
        self.assertEqual(len(sent['continuation_prompts']), 29)
        self.assertIn(prompt.decode(), sent['continuation_prompts'][1])
        self.assertGreater(len(self.parts['prompt']['data']), 131072)
        self.assertGreater(len(self.parts['extra_params']['data']), 4 * 1024 * 1024)

    def test_generated_extra_over_default_limit_is_rejected_before_http(self):
        result = self.render(b'x' * 40000, {'long_video': True}, duration=300, CONT_PROMPTS='auto')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'field limit 1048576', result.stderr)
        self.assertEqual(self.received, 0)
        self.assertFalse(list(Path(self.env['TMPDIR']).glob('h3-text.*')))

    def test_prompt_limit_counts_utf8_bytes(self):
        result = self.render('ą'.encode() * 600, H3_TEXT_MAX_FIELD_BYTES='1000')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'limit 1000', result.stderr)
        self.assertEqual(self.received, 0)

    def test_explicit_extra_file_over_default_limit_is_rejected_before_http(self):
        result = self.render(b'prompt', {'notes': 'x' * 1048576})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'field limit 1048576', result.stderr)
        self.assertEqual(self.received, 0)

    def test_conflicting_file_and_legacy_sources_are_rejected(self):
        for environment, positional in [({}, 'positional prompt'), ({'EXTRA_JSON': '{}'}, '')]:
            with self.subTest(environment=environment, positional=positional):
                result = self.render(b'file prompt', positional=positional, **environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b'choose', result.stderr)
                self.assertEqual(self.received, 0)

    def test_invalid_utf8_is_rejected_before_http(self):
        result = self.render(b'bad\xff')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.received, 0)

    def test_dash_prefixed_relative_paths_are_values_not_helper_options(self):
        prompt = b'Scene; literal prompt.\r\n\n'
        (self.root / '-prompt.txt').write_bytes(prompt)
        (self.root / '-extra.json').write_text('{"notes":"literal; value"}')
        (self.root / '-ref.png').write_bytes(b'image-content-for-upload')
        # Cover the legacy image interface as well as each new file option.
        for field in ('image', 'prompt', 'extra', 'all'):
            with self.subTest(field=field):
                self.received = 0
                self.parts = {}
                self.audit.unlink(missing_ok=True)
                env = dict(self.env, EXTRA_JSON='{"notes":"literal; value"}')
                positional = prompt.decode()
                if field in ('prompt', 'all'):
                    env['H3_PROMPT_FILE'] = '-prompt.txt'
                    positional = ''
                if field in ('extra', 'all'):
                    env.pop('EXTRA_JSON')
                    env['H3_EXTRA_JSON_FILE'] = '-extra.json'
                image = '-ref.png' if field in ('image', 'all') else './-ref.png'
                result = subprocess.run(
                    ['bash', str(REPO / 'scripts/render.sh'), str(self.root / 'out.mp4'),
                     '30', '960', '544', '20', '3101', positional, image, 'first'],
                    cwd=self.root, env=env, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                _, sent = self.verify_transport(prompt)
                self.assertEqual(sent['notes'], 'literal; value')

    def test_legacy_auto_expansion_over_128k_uses_file_transport(self):
        # Original failure: the input argument/env fit, generated extra did not.
        prompt = 'A steady scene; Polish dialogue.\n' * 600
        env = dict(self.env, EXTRA_JSON='{"long_video":true}', CONT_PROMPTS='auto')
        result = subprocess.run(['bash', str(REPO / 'scripts/render.sh'), str(self.root / 'out.mp4'),
                                 '300', '960', '544', '20', '3101', prompt], env=env,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        timing = json.loads((self.root / 'out.mp4.timing.json').read_bytes())
        self.assertEqual(len(timing['extra_params']['continuation_prompts']), 29)
        self.assertEqual(timing['prompt'], prompt)
        self.assertGreater(timing['text_fields']['extra_params']['bytes'], 131072)
        self.assertEqual(hashlib.sha256(self.parts['extra_params']['data']).hexdigest(),
                         timing['text_fields']['extra_params']['sha256'])
        audit = [json.loads(line) for line in self.audit.read_text().splitlines()]
        self.assertFalse(any(row['extra_json_exported'] for row in audit))


if __name__ == '__main__':
    unittest.main()
