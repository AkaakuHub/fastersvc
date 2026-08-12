import unittest

import torch

from module.audio import a_weighting, normalize_ssl_input, perceptual_loudness


class AudioPreprocessingTest(unittest.TestCase):
    def test_normalizes_each_waveform_independently(self):
        waveforms = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 14.0, 18.0, 22.0],
        ])

        normalized = normalize_ssl_input(waveforms)

        self.assertTrue(torch.allclose(normalized.mean(dim=1), torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(normalized.var(dim=1, unbiased=False), torch.ones(2), atol=1e-6))

    def test_rejects_ambiguous_channel_dimension(self):
        with self.assertRaises(ValueError):
            normalize_ssl_input(torch.zeros(1, 1, 100))

    def test_perceptual_loudness_has_decoder_frame_timing(self):
        loudness = perceptual_loudness(torch.zeros(2, 96 * 5))

        self.assertEqual(loudness.shape, (2, 1, 5))
        self.assertTrue(torch.equal(loudness, torch.full_like(loudness, -80.0)))

    def test_a_weighting_uses_the_reference_frequency_scale(self):
        weighting = a_weighting(torch.tensor([0.0, 1000.0]))

        self.assertEqual(weighting[0].item(), -80.0)
        self.assertAlmostEqual(weighting[1].item(), 0.0, places=2)

    def test_perceptual_loudness_applies_a_weighting(self):
        sample_rate = 24000
        samples = torch.arange(480 * 8) / sample_rate
        low_tone = torch.sin(2 * torch.pi * 50 * samples)
        reference_tone = torch.sin(2 * torch.pi * 1000 * samples)

        low_loudness = perceptual_loudness(low_tone.unsqueeze(0)).mean()
        reference_loudness = perceptual_loudness(reference_tone.unsqueeze(0)).mean()

        self.assertGreater((reference_loudness - low_loudness).item(), 20.0)

    def test_perceptual_loudness_rejects_partial_frames(self):
        with self.assertRaises(ValueError):
            perceptual_loudness(torch.zeros(1, 97))


if __name__ == "__main__":
    unittest.main()
