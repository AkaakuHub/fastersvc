import unittest

import torch

from module.excitation import generate_excitation


class ExcitationTest(unittest.TestCase):
    def test_uses_sine_for_voiced_and_noise_for_unvoiced_frames(self):
        f0 = torch.tensor([[[100.0, 0.0]]])
        noise = torch.ones(1, 1, 8)

        excitation, _ = generate_excitation(
            f0,
            frame_size=4,
            sample_rate=8000,
            noise=noise,
        )

        self.assertTrue(torch.all(excitation[:, :, 4:] == 0.3))
        self.assertTrue(torch.all(excitation[:, :, :4] < 0.3))

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


if __name__ == "__main__":
    unittest.main()
