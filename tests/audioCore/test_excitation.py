import unittest

import torch

from module.excitation import generate_excitation, interpolate_f0


class ExcitationTest(unittest.TestCase):
    def test_interpolates_pitch_across_the_entire_frame(self):
        pitch = torch.tensor([[[200.0]]])

        interpolated = interpolate_f0(
            pitch,
            frame_size=4,
            previous_f0=torch.tensor([[[100.0]]]),
        )

        self.assertTrue(torch.allclose(
            interpolated,
            torch.tensor([[[125.0, 150.0, 175.0, 200.0]]]),
        ))

    def test_uses_sine_for_voiced_and_noise_for_unvoiced_frames(self):
        f0 = torch.tensor([[[100.0, 0.0]]])
        noise = torch.ones(1, 1, 8)

        excitation, _ = generate_excitation(
            f0,
            frame_size=4,
            sample_rate=8000,
            noise=noise,
        )

        self.assertAlmostEqual(excitation[:, :, -1:].item(), 0.3)
        self.assertTrue(torch.all(excitation[:, :, :-1] < 0.3))

    def test_preserves_phase_across_adjacent_chunks(self):
        f0 = torch.full((1, 1, 2), 200.0)
        noise = torch.zeros(1, 1, 8)
        first, phase = generate_excitation(
            f0,
            frame_size=4,
            sample_rate=8000,
            noise=noise,
        )
        second, _ = generate_excitation(
            f0,
            phase=phase,
            frame_size=4,
            sample_rate=8000,
            noise=noise,
        )
        full, _ = generate_excitation(
            torch.full((1, 1, 4), 200.0),
            frame_size=4,
            sample_rate=8000,
            noise=torch.zeros(1, 1, 16),
        )

        self.assertTrue(torch.allclose(torch.cat([first, second], dim=2), full, atol=1e-6))

    def test_preserves_pitch_interpolation_across_changing_chunks(self):
        first_pitch = torch.full((1, 1, 2), 200.0)
        second_pitch = torch.full((1, 1, 2), 400.0)
        first, phase = generate_excitation(
            first_pitch,
            phase=0.3,
            frame_size=4,
            sample_rate=8000,
            noise=torch.zeros(1, 1, 8),
        )
        second, _ = generate_excitation(
            second_pitch,
            phase=phase,
            frame_size=4,
            sample_rate=8000,
            noise=torch.zeros(1, 1, 8),
            previous_f0=first_pitch[:, :, -1:],
        )
        full, _ = generate_excitation(
            torch.cat([first_pitch, second_pitch], dim=2),
            phase=0.3,
            frame_size=4,
            sample_rate=8000,
            noise=torch.zeros(1, 1, 16),
        )

        self.assertTrue(torch.allclose(torch.cat([first, second], dim=2), full, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
