import torch.nn as nn


def normalized_convolution(
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1):
    convolution = nn.Conv1d(
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        groups=groups,
    )
    convolution.weight.data.normal_(0.0, 0.02)
    return nn.utils.weight_norm(convolution)


class ScaleDiscriminator(nn.Module):
    def __init__(
            self,
            channels=16,
            downsample_scales=(4, 4, 4),
            max_channels=512):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                normalized_convolution(1, channels, 15),
                nn.LeakyReLU(0.2),
            ),
        ])
        input_channels = channels
        for scale in downsample_scales:
            output_channels = min(input_channels * scale, max_channels)
            self.layers.append(nn.Sequential(
                normalized_convolution(
                    input_channels,
                    output_channels,
                    scale * 10 + 1,
                    stride=scale,
                    padding=scale * 5,
                    groups=input_channels // 4,
                ),
                nn.LeakyReLU(0.2),
            ))
            input_channels = output_channels
        output_channels = min(input_channels * 2, max_channels)
        self.layers.extend([
            nn.Sequential(
                normalized_convolution(input_channels, output_channels, 5, padding=2),
                nn.LeakyReLU(0.2),
            ),
            normalized_convolution(output_channels, 1, 3, padding=1),
        ])

    def forward(self, waveform):
        features = []
        for layer in self.layers:
            waveform = layer(waveform)
            features.append(waveform)
        return waveform, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            num_scales=3,
            channels=16,
            downsample_scales=(4, 4, 4),
            max_channels=512):
        super().__init__()
        self.sub_discs = nn.ModuleList([
            ScaleDiscriminator(channels, downsample_scales, max_channels)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool1d(
            kernel_size=4,
            stride=2,
            padding=1,
            count_include_pad=False,
        )

    def forward(self, waveform):
        features = []
        logits = []
        for discriminator in self.sub_discs:
            logit, scale_features = discriminator(waveform)
            logits.append(logit)
            features.extend(scale_features)
            waveform = self.downsample(waveform)
        return logits, features


class Discriminator(MultiScaleDiscriminator):
    def forward(self, waveform):
        return super().forward(waveform.unsqueeze(1))
