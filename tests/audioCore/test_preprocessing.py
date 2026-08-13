import unittest

import torch

from module.preprocessing import split_complete_segments, trim_silence


class PreprocessingTest(unittest.TestCase):
    def test_trims_silence_relative_to_recording_level(self):
        waveform = torch.cat([
            torch.zeros(1, 8),
            torch.ones(1, 16),
            torch.zeros(1, 8),
        ], dim=1)

        trimmed = trim_silence(waveform, top_db=20, frame_size=4, hop_size=2)

        self.assertEqual(trimmed.shape, (1, 20))
        self.assertTrue(torch.equal(trimmed[:, 2:-2], torch.ones(1, 16)))

    def test_rejects_fully_silent_recording(self):
        trimmed = trim_silence(torch.zeros(1, 32), frame_size=4, hop_size=2)

        self.assertEqual(trimmed.shape, (1, 0))

    def test_discards_incomplete_tail_instead_of_padding_silence(self):
        waveform = torch.arange(10).reshape(1, 10)

        segments = split_complete_segments(waveform, 4)

        self.assertEqual(len(segments), 2)
        self.assertTrue(torch.equal(segments[0], torch.tensor([[0, 1, 2, 3]])))
        self.assertTrue(torch.equal(segments[1], torch.tensor([[4, 5, 6, 7]])))


if __name__ == "__main__":
    unittest.main()
