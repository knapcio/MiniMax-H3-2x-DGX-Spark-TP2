"""Exercise the real retry gates and ordered failure handling without GPUs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name('queue-retry300-followups-20261031.py')
spec = importlib.util.spec_from_file_location('retry_followups', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class RetryFollowupsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.results = self.base / 'results'; self.results.mkdir()
        self.retry = self.results / 'retry'; self.retry.mkdir()
        self.root = self.results / 'isolated'; self.root.mkdir()
        image = self.results / 'koninck.png'; image.write_bytes(b'painting')
        self.image_sha = hashlib.sha256(b'painting').hexdigest()
        (self.retry / 'runner.exit').write_text('0\n')
        (self.retry / 'events.log').write_text('H3-RETRY300-START x\nH3-RETRY300-END rc=0 x\n')
        (self.retry / 'STATUS').write_text(queue.TAG + ' ok\n')
        (self.retry / (queue.TAG + '.verify.txt')).write_text('decode: OK\nverify: OK\n')
        (self.retry / (queue.TAG + '.mp4')).write_bytes(b'video')
        self.timing = {'http': '200', 'curl_rc': 0, 'duration': 300, 'size': '960x544', 'steps': 20,
                       'seed': 1101, 'task': 'fl2va', 'image': str(image), 'elapsed_s': 9000,
                       'extra_params': {'long_video': True, 'long_video_mode': 'continuation', 'preencode_mp4': True}}
        self.timing_path = self.retry / (queue.TAG + '.mp4.timing.json')
        self.timing_path.write_text(json.dumps(self.timing))
        for name in ('head', 'worker'):
            (self.retry / (queue.TAG + f'.mem-{name}.csv')).write_text('1,10000,0,0\n2,9000,0,0\n')
        self.probe = lambda _: {'streams': [{'codec_type': 'video', 'width': 960, 'height': 544,
                                             'avg_frame_rate': '24/1', 'nb_frames': '7205'},
                                            {'codec_type': 'audio'}], 'format': {'duration': '300.21'}}
        for duration in (10, 15):
            name = f'quality{duration}-direct-960x544-20261031'
            tag = f'd{duration}_960x544_s20_seed1101_fl2va'
            folder = self.results / name; folder.mkdir()
            (folder / 'STATUS').write_text(tag + ' ok\n')
            (folder / (tag + '.verify.txt')).write_text('verify: OK\n')
            (folder / (tag + '.mp4')).write_bytes(b'video')
            (folder / 'source.sha256').write_text(self.image_sha + '\n')
            data = dict(self.timing, duration=duration)
            (folder / (tag + '.mp4.timing.json')).write_text(json.dumps(data))
            (self.root / name).symlink_to(folder, target_is_directory=True)
        for duration in (16, 60):
            name = f'quality{duration}-960x544-20261031'
            folder = self.results / name; folder.mkdir()
            (folder / 'source.sha256').write_text(self.image_sha + '\n')
            (self.root / name).symlink_to(folder, target_is_directory=True)
        (self.root / 'koninck.png').symlink_to(image)
        (self.root / 'experiment300-960x544-20261031').symlink_to(self.retry, target_is_directory=True)

    def validate(self):
        return queue.validate_retry(self.retry, self.results, image_sha=self.image_sha, media_probe=self.probe)

    def test_verified_retry_reports_hash_and_minimum_memory(self):
        result = self.validate()
        self.assertEqual(result['native_frames'], 7205)
        self.assertEqual(result['min_memavailable_mib'], {'head': 9000, 'worker': 9000})
        with mock.patch.object(queue, 'IMAGE_SHA', self.image_sha):
            queue.validate_controls_and_links(self.root, self.results, self.retry)

    def test_bad_retry_never_passes_gate(self):
        for kind in ('exit', 'event', 'status', 'decode', 'timing', 'memory', 'frames', 'image'):
            with self.subTest(kind=kind):
                other = RetryFollowupsTests(); other.setUp()
                try:
                    if kind == 'exit': (other.retry / 'runner.exit').write_text('1\n')
                    if kind == 'event': (other.retry / 'events.log').write_text('H3-RETRY300-START x\n')
                    if kind == 'status': (other.retry / 'STATUS').write_text(queue.TAG + ' failed\n')
                    if kind == 'decode': (other.retry / (queue.TAG + '.verify.txt')).write_text('verify: FAIL\n')
                    if kind == 'timing': other.timing_path.write_text(json.dumps(dict(other.timing, http='500')))
                    if kind == 'memory': (other.retry / (queue.TAG + '.mem-worker.csv')).write_text('1,10000\n2,100\n')
                    if kind == 'frames':
                        other.probe = lambda _: {'streams': [{'codec_type': 'video', 'width': 960, 'height': 544,
                                                               'avg_frame_rate': '24/1', 'nb_frames': '7199'},
                                                              {'codec_type': 'audio'}], 'format': {'duration': '300'}}
                    if kind == 'image': (other.results / 'koninck.png').write_bytes(b'changed')
                    with self.assertRaises(ValueError): other.validate()
                finally:
                    other.doCleanups()

    def test_invalid_links_or_short_controls_do_not_pass(self):
        with mock.patch.object(queue, 'IMAGE_SHA', self.image_sha):
            link = self.root / 'experiment300-960x544-20261031'; link.unlink()
            link.symlink_to(self.results / 'quality16-960x544-20261031', target_is_directory=True)
            with self.assertRaises(ValueError): queue.validate_controls_and_links(self.root, self.results, self.retry)
            link.unlink(); link.symlink_to(self.retry, target_is_directory=True)
            (self.results / 'quality15-direct-960x544-20261031' / 'STATUS').write_text('failed\n')
            with self.assertRaises(ValueError): queue.validate_controls_and_links(self.root, self.results, self.retry)

    def test_xid_query_failure_or_error_stops_progress(self):
        with mock.patch.object(queue.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['journalctl'])):
            with self.assertRaises(subprocess.CalledProcessError): queue.ensure_no_xid(self.root, 'failed')
        with mock.patch.object(queue.subprocess, 'run', return_value=mock.Mock(stdout='kernel NVRM: Xid 13\n')):
            with self.assertRaises(ValueError): queue.ensure_no_xid(self.root, 'xid')
            self.assertIn('Xid 13', (self.root / 'xid-head-xid.txt').read_text())
        with mock.patch.object(queue.subprocess, 'run', return_value=mock.Mock(stdout='other kernel event\n')):
            queue.ensure_no_xid(self.root, 'ok')
            self.assertIn('no Xid', (self.root / 'xid-worker-ok.txt').read_text())

    def test_vanished_runner_is_held_without_waiting_for_deadline(self):
        (self.retry / 'runner.exit').unlink()
        with mock.patch.object(queue, 'runner_alive', return_value=False), mock.patch.object(queue.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'vanished'):
                queue.wait_for_retry(self.retry, max_wait=300, interval=30)

    def run_main(self, retry_error=None, queue_error=None):
        argv = ['watch', '--root', str(self.root), '--results', str(self.results), '--retry', str(self.retry),
                '--manifest-sha', 'a' * 64]
        exact = self.root / 'exact.mp4'; exact.write_bytes(b'exact')
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(queue, 'verify_bundle'), \
             mock.patch.object(queue, 'wait_for_retry'), \
             mock.patch.object(queue, 'validate_retry', side_effect=retry_error, return_value={'ok': True}), \
             mock.patch.object(queue, 'validate_controls_and_links'), \
             mock.patch.object(queue, 'ensure_no_xid'), \
             mock.patch.object(queue, 'make_exact_and_screens', return_value=exact), \
             mock.patch.object(queue.shutil, 'disk_usage', return_value=mock.Mock(free=32 * 1024 ** 3)), \
             mock.patch.object(queue, 'run_queue', side_effect=queue_error) as children:
            result = queue.main()
            return result, children.call_args_list

    def test_failed_predecessor_never_starts_queue(self):
        result, calls = self.run_main(retry_error=ValueError('failed retry'))
        self.assertEqual(result, 1)
        self.assertEqual(calls, [])
        events = (self.root / 'aspect-events-20261031.log').read_text()
        self.assertIn('H3-RETRY-FOLLOWUPS-HOLD phase=predecessor', events)
        self.assertNotIn('H3-ASPECT-EXPERIMENT300-DONE', events)

    def test_failed_first_queue_stops_later_queues(self):
        result, calls = self.run_main(queue_error=subprocess.CalledProcessError(1, ['camera']))
        self.assertEqual(result, 1)
        self.assertEqual([call.args[2] for call in calls], ['camera-speed'])
        self.assertIn('H3-RETRY-FOLLOWUPS-HOLD phase=camera-speed',
                      (self.root / 'aspect-events-20261031.log').read_text())

    def test_success_runs_all_comparisons_in_order(self):
        result, calls = self.run_main()
        self.assertEqual(result, 0)
        self.assertEqual([call.args[2] for call in calls], ['camera-speed', 'window175', 'human-pace'])
        self.assertIn('H3-RETRY-FOLLOWUPS-DONE quality-review-required',
                      (self.root / 'aspect-events-20261031.log').read_text())


if __name__ == '__main__':
    unittest.main()
