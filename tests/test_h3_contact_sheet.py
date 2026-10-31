import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

from PIL import Image


SCRIPT=Path(__file__).resolve().parents[1]/'scripts/h3_contact_sheet.py'
spec=importlib.util.spec_from_file_location('h3_contact_sheet',SCRIPT)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class H3ContactSheetTests(unittest.TestCase):
    def test_extracts_one_labeled_sample_per_second_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(72):
                color=((200,20,20) if i<24 else (20,200,20) if i<48 else (20,20,200))
                Image.new('RGB',(64,64),color).save(root/f'frame-{i:03d}.png')
            source=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-framerate','24','-i',str(root/'frame-%03d.png'),
                            '-c:v','libx264','-crf','0','-pix_fmt','yuv420p',str(source)],check=True)
            before=module.digest(source); out=root/'sheet.png'
            result=module.make(source,out,1)
            self.assertEqual(result['sample_seconds'],[0,1,2])
            self.assertEqual(result['source_frames'],72)
            self.assertEqual(result['source_sha256'],before)
            self.assertEqual(result['output_sha256'],module.digest(out))
            with Image.open(out) as image:
                for col,dominant in enumerate((0,1,2)):
                    rgb=image.getpixel((col*480+240,56+240))
                    self.assertEqual(rgb.index(max(rgb)),dominant)
            with self.assertRaisesRegex(ValueError,'output must be new'):
                module.make(source,out,1)
            self.assertEqual(module.digest(source),before)

    def test_rejects_invalid_interval_before_reading_source(self):
        for interval in (0,-1,1.5,True):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                module.make(Path('unused.mp4'),Path('out.png'),interval)


if __name__=='__main__': unittest.main()
