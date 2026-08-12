import unittest
from unittest.mock import patch

import numpy as np
import torch

from module.adversarial import discriminator_loss, generator_adversarial_loss
from module.common import compute_f0, compute_f0_harvest
from module.discriminator import MultiScaleDiscriminator
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
        discriminator = MultiScaleDiscriminator(
            num_scales=3,
            channels=8,
            max_channels=32,
            max_groups=4,
            num_layers=2,
        )
        logits, _ = discriminator(waveform)
        lengths = [logit.shape[-1] for logit in logits]
        self.assertGreater(lengths[0], lengths[1])
        self.assertGreater(lengths[1], lengths[2])

    @patch("module.common.pw.harvest")
    def test_batched_harvest_does_not_call_dio(self, harvest):
        harvest.return_value = np.ones(10), np.arange(10)
        waveforms = torch.zeros(2, 1600)
        result = compute_f0_harvest(waveforms, sample_rate=16000, segment_size=320)
        self.assertEqual(harvest.call_count, 2)
        self.assertEqual(result.shape, (2, 1, 5))

    @patch("module.common.compute_f0_harvest")
    def test_f0_frame_size_tracks_resampled_sample_rate(self, harvest):
        harvest.return_value = torch.ones(1, 1, 5)
        waveform = torch.zeros(1, 2400)

        result = compute_f0(waveform, sample_rate=24000, segment_size=480)

        self.assertEqual(harvest.call_args.args[1:], (16000, 320))
        self.assertEqual(result.shape, (1, 1, 5))


if __name__ == "__main__":
    unittest.main()
