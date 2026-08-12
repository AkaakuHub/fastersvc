import unittest

import torch

from module.decoder import Decoder, FiLM


class DecoderTest(unittest.TestCase):
    def test_film_builds_one_shift_and_scale_from_both_conditions(self):
        film = FiLM(channels=4, cond_channels=3)

        shift, scale = film(torch.randn(1, 3, 8), torch.randn(1, 3, 8))

        self.assertEqual(shift.shape, (1, 4, 8))
        self.assertEqual(scale.shape, (1, 4, 8))

    def test_synthesizes_bounded_waveform_at_expected_length(self):
        decoder = Decoder(
            channels=[16, 12, 8, 4],
            cond_channels=[16, 12, 8, 4],
            content_channels=8,
        )
        content = torch.randn(1, 8, 2)
        pitch = torch.full((1, 1, 2), 180.0)
        loudness = torch.ones(1, 1, 10)

        waveform = decoder.synthesize(content, pitch, loudness)

        self.assertEqual(waveform.shape, (1, 960))
        self.assertLessEqual(waveform.abs().max().item(), 1)

    def test_rejects_excitation_that_differs_from_content_timing(self):
        decoder = Decoder(
            channels=[16, 12, 8, 4],
            cond_channels=[16, 12, 8, 4],
            content_channels=8,
        )

        with self.assertRaises(ValueError):
            decoder(
                torch.randn(1, 8, 2),
                torch.ones(1, 1, 10),
                torch.randn(1, 1, 959),
            )


if __name__ == "__main__":
    unittest.main()
