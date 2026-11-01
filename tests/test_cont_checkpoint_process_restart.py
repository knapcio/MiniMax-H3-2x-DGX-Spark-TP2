"""Two fresh OS processes per rank exercise CPU-only checkpoint recovery."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'tests/distributed_checkpoint_restart_cpu.py'


class ProcessRestartTests(unittest.TestCase):
    def run_ranks(self, root, phase, common_window, *, kill_after_save=False, stop_window=2, total_frames=617):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        env = os.environ.copy()
        env.update({'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'PYTHONPATH': str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')})
        processes = []
        try:
            for rank in (0, 1):
                command = [sys.executable, str(WORKER), '--phase', phase, '--rank', str(rank), '--port', str(port),
                           '--root', str(root), '--common-window', str(common_window), '--stop-window', str(stop_window),
                           '--total-frames', str(total_frames)]
                if kill_after_save: command.append('--wait-after-save')
                processes.append(subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                  text=True))
            if kill_after_save:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if any(process.poll() is not None for process in processes):
                        self.fail('a rank exited before simulated process loss')
                    try:
                        for rank in (0, 1):
                            json.loads((root / f'save-rank-{rank}.json').read_text())
                            (root / f'waiting-rank-{rank}').read_text()
                        break
                    except (OSError, ValueError):
                        time.sleep(.05)
                else:
                    self.fail('ranks did not report saved checkpoints before deadline')
                for process in processes: process.kill()
                for rank, process in enumerate(processes):
                    output, _ = process.communicate(timeout=15)
                    self.assertNotEqual(process.returncode, 0, f'rank {rank} did not terminate: {output}')
                return [json.loads((root / f'{phase}-rank-{rank}.json').read_text()) for rank in (0, 1)]
            for rank, process in enumerate(processes):
                output, _ = process.communicate(timeout=90)
                self.assertEqual(process.returncode, 0, f'{phase} rank {rank} failed:\n{output[-6000:]}')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill(); process.wait(timeout=10)
        return [json.loads((root / f'{phase}-rank-{rank}.json').read_text()) for rank in (0, 1)]

    def check_restart(self, corrupt_rank_one, *, kill_after_save=False, stop_window=2, total_frames=617):
        with tempfile.TemporaryDirectory(prefix='h3-cpu-process-restart-') as temp:
            root = Path(temp); common = stop_window - 1 if corrupt_rank_one else stop_window
            saved = self.run_ranks(root, 'save', common, kill_after_save=kill_after_save,
                                   stop_window=stop_window, total_frames=total_frames)
            if corrupt_rank_one:
                state = root / 'rank-1' / ('1' * 32) / f'rank-0001-window-{stop_window:06d}.safetensors'
                with state.open('r+b') as out:
                    out.seek(-1, 2); old = out.read(1); out.seek(-1, 2); out.write(bytes([old[0] ^ 1]))
            resumed = self.run_ranks(root, 'resume', common, stop_window=stop_window, total_frames=total_frames)
            for before, after in zip(saved, resumed, strict=True):
                self.assertNotEqual(before['pid'], after['pid'])
                self.assertEqual(before['expected_av_sha256'], after['expected_av_sha256'])
                self.assertEqual(before['rendered_windows'], stop_window)
                self.assertEqual(before['total_windows'], after['total_windows'])
                self.assertEqual(after['rendered_windows'], before['total_windows'] - common)

    @unittest.skipUnless(dist.is_gloo_available(), 'Gloo backend required')
    def test_two_fresh_ranks_resume_latest_common_window(self):
        self.check_restart(corrupt_rank_one=False)

    @unittest.skipUnless(dist.is_gloo_available(), 'Gloo backend required')
    def test_fresh_ranks_fall_back_after_one_rank_is_corrupted(self):
        self.check_restart(corrupt_rank_one=True)

    @unittest.skipUnless(dist.is_gloo_available(), 'Gloo backend required')
    def test_fresh_ranks_resume_after_both_previous_processes_are_killed(self):
        self.check_restart(corrupt_rank_one=False, kill_after_save=True)

    @unittest.skipUnless(dist.is_gloo_available(), 'Gloo backend required')
    def test_300_second_plan_recovers_after_window_20(self):
        self.check_restart(corrupt_rank_one=False, kill_after_save=True, stop_window=20, total_frames=7213)


if __name__ == '__main__':
    unittest.main()
