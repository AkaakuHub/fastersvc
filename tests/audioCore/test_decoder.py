import unittest

import torch

from module.decoder import Decoder, FiLM


class DecoderTest(unittest.TestCase):
    def test_film_builds_one_shift_and_scale_from_both_conditions(self):
        film = FiLM(channels=4, cond_channels=3)

        shift, scale = film(torch.randn(1, 3, 8), torch.randn(1, 3, 8))

        self.assertEqual(shift.shape, (1, 4, 8))
        self.assertEqual(scale.shape, (1, 4, 8))

    def test_synthesizes_finite_waveform_at_expected_length(self):
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
        self.assertTrue(torch.isfinite(waveform).all())

    def test_conditions_each_upsample_block_at_its_output_resolution(self):
        decoder = Decoder(
            channels=[16, 12, 8, 4],
            cond_channels=[16, 12, 8, 4],
            content_channels=8,
        )
        condition_lengths = []
        hooks = [
            block.register_forward_pre_hook(
                lambda _, inputs: condition_lengths.append(inputs[1].shape[-1])
            )
            for block in decoder.ups
        ]

        decoder(
            torch.randn(1, 8, 2),
            torch.ones(1, 1, 10),
            torch.randn(1, 1, 960),
        )
        for hook in hooks:
            hook.remove()

        self.assertEqual(condition_lengths, [12, 48, 192, 960])

    def test_uses_the_declared_channel_count_for_each_upsample_block(self):
        decoder = Decoder(
            channels=[16, 12, 8, 4],
            cond_channels=[16, 12, 8, 4],
            content_channels=8,
        )

        output_channels = [block.residual.out_channels for block in decoder.ups]

        self.assertEqual(output_channels, [16, 12, 8, 4])

    def test_uses_wavegrad_initialization_for_waveform_blocks(self):
        torch.manual_seed(3)
        decoder = Decoder(
            channels=[16, 12, 8, 4],
            cond_channels=[16, 12, 8, 4],
            content_channels=8,
        )

        weight = decoder.ups[0].residual.weight[:, :, 0]
        gram = weight.transpose(0, 1) @ weight

        self.assertTrue(torch.allclose(gram, torch.eye(gram.shape[0]), atol=1e-5))
        self.assertTrue(torch.equal(
            decoder.ups[0].film.input_convs[0].conv.bias,
            torch.zeros_like(decoder.ups[0].film.input_convs[0].conv.bias),
        ))

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
