import torch.nn as nn

from .common import match_features


# kNN based style convertor for onnx exporting
class IndexForOnnx(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.register_buffer("index", index)

    def forward(self, x, metrics='L2'):
        return match_features(x, self.index, metrics=metrics)
