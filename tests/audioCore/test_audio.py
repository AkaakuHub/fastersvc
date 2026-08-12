import unittest

import torch

from module.audio import normalize_ssl_input


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


if __name__ == "__main__":
    unittest.main()
