import unittest

import torch

from module.content_features import align_content_frames, select_hubert_content


class ContentFeaturesTest(unittest.TestCase):
    def test_selects_ninth_hubert_layer_without_mixing_speaker_richer_layers(self):
        hidden_states = tuple(torch.full((1, 2, 3), float(index)) for index in range(13))

        content = select_hubert_content(hidden_states)

        self.assertTrue(torch.equal(content, torch.full((1, 3, 2), 9.0)))

    def test_aligns_content_frames_with_linear_interpolation(self):
        features = torch.tensor([[[0.0, 4.0]]])

        aligned = align_content_frames(features, 4)

        self.assertTrue(torch.allclose(aligned, torch.tensor([[[0.0, 1.0, 3.0, 4.0]]])))

    def test_rejects_absent_hubert_layer(self):
        with self.assertRaises(ValueError):
            select_hubert_content((torch.zeros(1, 2, 3),))


if __name__ == "__main__":
    unittest.main()
