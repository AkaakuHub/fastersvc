import unittest

import torch

from module.decoder import Decoder


class DecoderTest(unittest.TestCase):
    def test_synthesizes_bounded_waveform_at_expected_length(self):
        decoder = Decoder(
            channels=[16, 12, 8, 6, 4],
            cond_channels=[16, 12, 8, 6, 4],
            content_channels=8,
        )
        content = torch.randn(1, 8, 2)
        pitch = torch.full((1, 1, 2), 180.0)
        loudness = torch.ones(1, 1, 2)

        waveform = decoder.synthesize(content, pitch, loudness)

        self.assertEqual(waveform.shape, (1, 960))
        self.assertLessEqual(waveform.abs().max().item(), 1)


if __name__ == "__main__":
    unittest.main()
