import unittest

import torch

from module.common import match_features
from module.index import IndexForOnnx


class FeatureMatchingTest(unittest.TestCase):
    def test_selects_and_averages_nearest_reference_frames(self):
        source = torch.tensor([[[0.9], [0.1]]])
        reference = torch.tensor([[[1.0, 0.8, 0.0], [0.0, 0.2, 1.0]]])

        result = match_features(source, reference, k=2, metrics="cos")

        expected = torch.tensor([[[0.9], [0.1]]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_onnx_index_is_not_a_trainable_parameter(self):
        index = IndexForOnnx(torch.randn(1, 8, 16))
        self.assertEqual(list(index.parameters()), [])
        self.assertEqual(list(index.buffers())[0].shape, (1, 8, 16))

    def test_onnx_index_uses_the_same_default_metric_as_conversion(self):
        reference = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 100.0], [0.0, 1.0, 1.0, 1.0, 0.0]]])
        source = torch.tensor([[[1.0], [0.0]]])

        expected = match_features(source, reference)
        actual = IndexForOnnx(reference)(source)

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
