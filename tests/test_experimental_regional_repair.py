import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/experimental_regional_repair.py'
spec = importlib.util.spec_from_file_location('experimental_regional_repair', SCRIPT)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class RegionalRepairTests(unittest.TestCase):
    def test_no_events_and_bounded_events(self):
        with mock.patch.object(subject.repair, 'regional_events_from_series', return_value={}):
            self.assertEqual(subject.choose_events(100, []), {})
        with mock.patch.object(subject.repair, 'regional_events_from_series', return_value={x: () for x in range(30)}):
            self.assertEqual(len(subject.choose_events(100, [])), 30)

    def test_too_many_or_invalid_events_are_rejected(self):
        for events in ({x: () for x in range(31)}, {-1: ()}, {100: ()}):
            with self.subTest(events=events), mock.patch.object(subject.repair, 'regional_events_from_series', return_value=events):
                with self.assertRaises(ValueError):
                    subject.choose_events(100, [])

    def test_no_overwrite_or_source_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, video, report = root / 'source.mp4', root / 'video.mp4', root / 'report.json'
            source.write_bytes(b'original')
            video.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                subject.build(source, 'x', video, report, expected_frames=384, width=960, height=544)
            self.assertEqual(video.read_bytes(), b'keep')
            with self.assertRaises(ValueError):
                subject.build(source, 'x', source, report, expected_frames=384, width=960, height=544)
            self.assertEqual(source.read_bytes(), b'original')

    def test_skips_when_no_qualifying_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, video, report = root / 'source.mp4', root / 'video.mp4', root / 'report.json'
            source.write_bytes(b'original')
            expected = subject.digest(source)
            with mock.patch.object(subject, 'source_info'), mock.patch.object(subject.regional, 'inspect', return_value=(24, 384, [])), \
                    mock.patch.object(subject, 'choose_events', return_value={}):
                self.assertFalse(subject.build(source, expected, video, report, expected_frames=384, width=960, height=544))
            self.assertFalse(video.exists())
            self.assertFalse(report.exists())

    def test_wrong_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'source.mp4'
            source.write_bytes(b'original')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                subject.source_info(source, 'bad', 384, 960, 544)


if __name__ == '__main__':
    unittest.main()
