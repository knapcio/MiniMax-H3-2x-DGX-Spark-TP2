import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/combine_repairs.py'
spec = importlib.util.spec_from_file_location('combine_repairs', SCRIPT)
combine = importlib.util.module_from_spec(spec); spec.loader.exec_module(combine)


class CombineRepairsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='h3-combine-test-'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.left = self.root/'left.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=128x96:rate=24:duration=3',
                        '-f','lavfi','-i','sine=frequency=440:sample_rate=32000:duration=3',
                        '-c:v','libx264','-crf','0','-pix_fmt','yuv420p','-c:a','aac','-t','3',str(self.left)],check=True)

    def modified(self, start=40):
        out = self.root/f'right-{start}.mp4'
        subprocess.run(['ffmpeg','-v','error','-i',str(self.left),'-vf',
                        f"drawbox=x=5:y=5:w=20:h=20:color=white:t=fill:enable='gte(n,{start})'",
                        '-c:v','libx264','-crf','0','-pix_fmt','yuv420p','-c:a','copy',str(out)],check=True)
        return out

    def test_combines_selected_frames_and_keeps_aac(self):
        right, output = self.modified(), self.root/'combined.mp4'
        before = [combine.digest(self.left), combine.digest(right)]
        result = combine.combine(self.left, right, output, 30)
        self.assertEqual((result['frames'], result['cut_frame'], result['duration_seconds']), (72, 30, 3))
        self.assertEqual(combine.exact.frame_hashes(output,72),
                         combine.exact.frame_hashes(self.left,72)[:30] + combine.exact.frame_hashes(right,72)[30:])
        self.assertEqual(combine.exact.audio_packets(output), combine.exact.audio_packets(self.left))
        self.assertEqual([combine.digest(self.left), combine.digest(right)], before)
        self.assertEqual(combine.digest(output), result['output_sha256'])

    def test_changed_boundary_or_invalid_cut_does_not_publish(self):
        right = self.modified(start=30)
        for cut in (1, 30, 71):
            with self.subTest(cut=cut):
                out = self.root/f'bad-{cut}.mp4'
                with self.assertRaises(ValueError): combine.combine(self.left, right, out, cut)
                self.assertFalse(out.exists())

    def test_different_audio_and_existing_output_are_rejected(self):
        right = self.root/'other-audio.mp4'
        subprocess.run(['ffmpeg','-v','error','-i',str(self.left),'-f','lavfi','-i',
                        'sine=frequency=700:sample_rate=32000:duration=3','-map','0:v:0','-map','1:a:0',
                        '-c:v','copy','-c:a','aac','-t','3',str(right)],check=True)
        out = self.root/'audio.mp4'
        with self.assertRaisesRegex(ValueError,'AAC'): combine.combine(self.left,right,out,30)
        self.assertFalse(out.exists())
        existing = self.root/'exists.mp4'; existing.write_bytes(b'keep')
        with self.assertRaises(ValueError): combine.combine(self.left,self.left,existing,30)
        self.assertEqual(existing.read_bytes(),b'keep')
        link = self.root/'dangling.mp4'; os.symlink(self.root/'missing',link)
        with self.assertRaises(ValueError): combine.combine(self.left,self.left,link,30)
        self.assertTrue(link.is_symlink())


if __name__ == '__main__': unittest.main()
