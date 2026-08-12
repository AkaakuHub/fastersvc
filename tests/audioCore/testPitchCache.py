import os
import tempfile
import unittest
from pathlib import Path

import torch

from rebuild_pitch_cache import publish_cache_item


class PitchCacheTest(unittest.TestCase):
    def test_publishes_pitch_atomically_without_copying_waveform(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            source_wave = source / "0.wav"
            source_wave.write_bytes(b"waveform")
            source_pitch = source / "0.pt"
            torch.save((torch.zeros(1, 2), 7), source_pitch)
            pitch = torch.tensor([[120.0, 121.0]])

            publish_cache_item(source_wave, source_pitch, output, pitch)

            cached_pitch, speaker_id = torch.load(output / "0.pt", weights_only=True)
            self.assertTrue(os.path.samefile(source_wave, output / "0.wav"))
            self.assertTrue(torch.equal(cached_pitch, pitch))
            self.assertEqual(speaker_id, 7)
            self.assertFalse((output / "0.pt.saving").exists())


if __name__ == "__main__":
    unittest.main()
