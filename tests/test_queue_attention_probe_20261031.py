"""CPU-only safety checks; these do not run a GPU benchmark."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / 'results/20261030-300s-readiness/queue-attention-probe-20261031.py'
spec = importlib.util.spec_from_file_location('attention_queue_test', path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class AttentionQueueTests(unittest.TestCase):
    def test_predecessor_gate_fails_closed(self):
        main = 'H3-RETRY-FOLLOWUPS-DONE timestamp'
        cloud = 'H3-ASPECT-CLOUDLIGHT-DONE matched-30s timestamp'
        self.assertEqual(subject.event_state([]), 'wait')
        self.assertEqual(subject.event_state([main]), 'wait')
        self.assertEqual(subject.event_state([main, cloud]), 'ready')
        self.assertEqual(subject.event_state([main, cloud, 'H3-ASPECT-HOLD error']), 'hold')
        self.assertEqual(subject.event_state([main, cloud, 'H3-RETRY-FOLLOWUPS-HOLD error']), 'hold')

    def test_request_metrics_require_both_finite_zero_values(self):
        base = ('vllm_omni:num_requests_running{model_name="h3"} 0.0\n'
                'vllm_omni:num_requests_waiting{model_name="h3"} 0\n')
        self.assertTrue(subject.metrics_idle(base))
        self.assertFalse(subject.metrics_idle(base.replace(' 0.0', ' 1.0')))
        for bad in ('', base.splitlines()[0], base.replace(' 0.0', ' NaN'), base.replace(' 0.0', ' -1')):
            with self.assertRaises(subject.Hold):
                subject.metrics_idle(bad)

    def test_bundle_rejects_tampering_and_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = {}
            for name in ('attention_probe.py', 'queue-attention-probe.py'):
                (root / name).write_text(name)
                data[name] = hashlib.sha256(name.encode()).hexdigest()
            manifest = json.dumps(data)
            (root / 'manifest.json').write_text(manifest)
            digest = hashlib.sha256(manifest.encode()).hexdigest()
            subject.verified_bundle(root, digest)
            (root / 'attention_probe.py').write_text('changed')
            with self.assertRaises(subject.Hold): subject.verified_bundle(root, digest)
            (root / 'attention_probe.py').unlink()
            (root / 'attention_probe.py').symlink_to(root / 'queue-attention-probe.py')
            with self.assertRaises(subject.Hold): subject.verified_bundle(root, digest)

    def test_idle_guard_checks_jobs_pids_memory_and_both_gpus(self):
        good = [{'pid': 11, 'mem_kib': 30000000, 'gpu': 1},
                {'pid': 12, 'mem_kib': 30000000, 'gpu': 2}]
        with patch.object(subject, 'request_state', return_value=True), patch.object(subject, 'node_snapshot', side_effect=good):
            self.assertEqual(subject.check_pair([11, 12], idle_gpu=True, minimum_kib=20000000), good)
        with patch.object(subject, 'request_state', return_value=False):
            with self.assertRaises(subject.Busy): subject.check_pair([11, 12], idle_gpu=True, minimum_kib=1)
        with patch.object(subject, 'request_state', return_value=True), patch.object(subject, 'node_snapshot', side_effect=good):
            with self.assertRaises(subject.Hold): subject.check_pair([11, 13], idle_gpu=True, minimum_kib=1)
        with patch.object(subject, 'request_state', return_value=True), patch.object(subject, 'node_snapshot', side_effect=good):
            with self.assertRaises(subject.Busy): subject.check_pair([11, 12], idle_gpu=True, minimum_kib=31000000)
        busy = [good[0], dict(good[1], gpu=70)]
        with patch.object(subject, 'request_state', return_value=True), patch.object(subject, 'node_snapshot', side_effect=busy):
            with self.assertRaises(subject.Busy): subject.check_pair([11, 12], idle_gpu=True, minimum_kib=1)

    def test_guard_stops_its_own_child_when_request_arrives(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            with patch.object(subject, 'check_pair', side_effect=subject.Busy('new request')):
                with self.assertRaises(subject.Busy): subject.monitor_child(child, [11, 12], Path('/tmp'), 60)
            self.assertIsNotNone(child.poll())
        finally:
            subject.stop_child(child)


if __name__ == '__main__':
    unittest.main()
