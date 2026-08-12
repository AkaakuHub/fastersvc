import unittest
from unittest.mock import patch

import numpy as np
import torch

from module.adversarial import discriminator_loss, generator_adversarial_loss
from module.common import compute_f0_harvest
from module.discriminator import DiscriminatorS
from module.loss import MultiResolutionSTFTLoss


class TrainingContractsTest(unittest.TestCase):
    def test_multiresolution_loss_is_zero_for_identical_waveforms(self):
        waveform = torch.randn(1, 4096)
        loss = MultiResolutionSTFTLoss()(waveform, waveform)
        self.assertLess(loss.item(), 1e-6)

    def test_lsgan_uses_real_one_and_generated_zero_targets(self):
        real = [torch.ones(1, 4)]
        generated = [torch.zeros(1, 4)]
        self.assertEqual(discriminator_loss(real, generated).item(), 0)
        self.assertEqual(generator_adversarial_loss(real).item(), 0)

    def test_discriminator_scales_downsample_higher_scales(self):
        waveform = torch.randn(1, 1, 16000)
        scale_one = DiscriminatorS(scale=1)
        scale_two = DiscriminatorS(scale=2)
        self.assertGreater(scale_one.pool(waveform).shape[-1], scale_two.pool(waveform).shape[-1])

    @patch("module.common.pw.harvest")
    def test_batched_harvest_does_not_call_dio(self, harvest):
        harvest.return_value = np.ones(10), np.arange(10)
        waveforms = torch.zeros(2, 1600)
        result = compute_f0_harvest(waveforms, sample_rate=16000, segment_size=320)
        self.assertEqual(harvest.call_count, 2)
        self.assertEqual(result.shape, (2, 1, 5))


if __name__ == "__main__":
    unittest.main()
