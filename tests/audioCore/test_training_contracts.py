import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from module.adversarial import discriminator_loss, generator_adversarial_loss
from module.common import compute_f0, compute_f0_harvest
from module.discriminator import MultiScaleDiscriminator
from module.loss import MultiResolutionSTFTLoss
from module.pitch_estimator import PitchEstimator
from module.training import DEFAULT_DECODER_LEARNING_RATE, atomic_save, crop_aligned_batch, learning_rate_at_step, step_scaled_optimizer, training_data_loader


class TrainingContractsTest(unittest.TestCase):
    def test_training_loader_prefetches_into_pinned_memory_for_cuda(self):
        dataset = torch.utils.data.TensorDataset(torch.zeros(4, 1))

        loader = training_data_loader(dataset, batch_size=2, workers=2, device=torch.device("cuda"))

        self.assertEqual(loader.num_workers, 2)
        self.assertTrue(loader.pin_memory)
        self.assertTrue(loader.persistent_workers)

    def test_atomic_save_replaces_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pt"

            atomic_save({"step": 3}, path)

            self.assertEqual(torch.load(path, weights_only=True), {"step": 3})
            self.assertFalse(Path(f"{path}.saving").exists())

    def test_pitch_estimator_covers_world_training_range(self):
        estimator = PitchEstimator()

        identifiers = estimator.freq2id(torch.tensor([0.0, 1100.0]))

        self.assertEqual(identifiers[0].item(), 0)
        self.assertLess(identifiers[1].item(), estimator.output_channels)
        self.assertEqual(estimator.output_channels, 512)

    def test_uses_paper_learning_rate_decay(self):
        self.assertEqual(DEFAULT_DECODER_LEARNING_RATE, 0.0001)
        self.assertEqual(learning_rate_at_step(0.001, 99999), 0.001)
        self.assertEqual(learning_rate_at_step(0.001, 100000), 0.0005)
        self.assertEqual(learning_rate_at_step(0.001, 200000), 0.00025)

    @patch("module.training.torch.randint")
    def test_crops_waveform_and_pitch_at_the_same_frame(self, randint):
        randint.return_value = torch.tensor([[1]])
        waveform = torch.arange(12).reshape(1, 12)
        pitch = torch.arange(4).reshape(1, 1, 4)

        cropped_waveform, cropped_pitch = crop_aligned_batch(
            waveform,
            pitch,
            sample_count=6,
            frame_size=3,
        )

        self.assertTrue(torch.equal(cropped_waveform, torch.tensor([[3, 4, 5, 6, 7, 8]])))
        self.assertTrue(torch.equal(cropped_pitch, torch.tensor([[[1, 2]]])))

    def test_unscales_gradients_before_clipping(self):
        parameter = torch.nn.Parameter(torch.tensor([10.0]))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
        loss = parameter * 100.0

        gradient_norm = step_scaled_optimizer(loss, optimizer, scaler, [parameter], 1.0)

        self.assertAlmostEqual(gradient_norm.item(), 100.0, places=3)
        self.assertAlmostEqual(parameter.item(), 9.0, places=3)

    def test_multiresolution_loss_is_zero_for_identical_waveforms(self):
        waveform = torch.randn(1, 4096)
        loss_function = MultiResolutionSTFTLoss()
        loss = loss_function(waveform, waveform)
        self.assertLess(loss.item(), 1e-6)
        self.assertEqual(
            len(tuple(loss_function.buffers())),
            len(loss_function.fft_sizes),
        )

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
