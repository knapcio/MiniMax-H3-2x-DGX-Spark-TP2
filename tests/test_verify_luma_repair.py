import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_luma_repair.py'
spec = importlib.util.spec_from_file_location('verify_luma_repair', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LumaRepairTests(unittest.TestCase):
    def test_frame_delta_and_chroma_invariant(self):
        original = bytes([100] * 16 + [128] * 8)
        luma = bytes([104] * 16 + [128] * 8)
        self.assertEqual(module.compare_frame(original, original, 4, 4), (False, 0))
        self.assertEqual(module.compare_frame(original, luma, 4, 4), (True, 4))
        with self.assertRaisesRegex(ValueError, 'chroma'):
            module.compare_frame(original, bytes([100] * 16 + [129] * 8), 4, 4)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            module.compare_frame(original, luma[:-1], 4, 4)

    def test_real_encoded_luma_edit_and_color_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, repaired, changed_color = (root / name for name in ('source.mp4', 'luma.mp4', 'chroma.mp4'))
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=gray:s=64x64:r=24:d=1',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=1',
                            '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                            '-shortest', str(source)], check=True)
            for dst, expression in ((repaired, 'lutyuv=y=val+1'), (changed_color, 'lutyuv=u=val+1')):
                subprocess.run(['ffmpeg', '-v', 'error', '-i', str(source), '-map', '0:v:0', '-map', '0:a:0',
                                '-vf', expression, '-fps_mode', 'passthrough', '-c:v', 'libx264', '-crf', '0',
                                '-pix_fmt', 'yuv420p', '-c:a', 'copy', str(dst)], check=True)
            result = module.inspect(source, repaired, limit=1)
            self.assertEqual(result['frames'], 24)
            self.assertEqual(result['changed_luma_frames'], list(range(24)))
            self.assertEqual(result['max_luma_change'], 1)
            self.assertTrue(result['chroma_identical'])
            self.assertTrue(result['video_timing_identical'])
            self.assertTrue(result['aac_packets_identical'])
            with self.assertRaisesRegex(ValueError, 'chroma'):
                module.inspect(source, changed_color)


if __name__ == '__main__':
    unittest.main()
