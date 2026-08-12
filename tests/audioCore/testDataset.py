import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio

from module.dataset import Dataset


class DatasetTest(unittest.TestCase):
    def create_item(self, directory, sample_rate=24000, sample_count=960, frame_count=2):
        torchaudio.save(directory / "0.wav", torch.zeros(1, sample_count), sample_rate)
        torch.save((torch.zeros(1, frame_count), 0), directory / "0.pt")

    def test_loads_aligned_training_item(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.create_item(directory)

            waveform, pitch, _ = Dataset(directory)[0]

            self.assertEqual(waveform.shape, (960,))
            self.assertEqual(pitch.shape, (1, 2))

    def test_rejects_unexpected_sample_rate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.create_item(directory, sample_rate=16000)

            with self.assertRaises(ValueError):
                Dataset(directory)[0]

    def test_rejects_misaligned_pitch_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            self.create_item(directory, frame_count=3)

            with self.assertRaises(ValueError):
                Dataset(directory)[0]


if __name__ == "__main__":
    unittest.main()
