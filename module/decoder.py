import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DCC
from .excitation import generate_excitation


def initialize_wavegrad_convolution(module):
    if isinstance(module, nn.Conv1d):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FiLM(nn.Module):
    def __init__(self, channels, cond_channels, condition_count=2):
        super().__init__()
        self.input_convs = nn.ModuleList([
            DCC(cond_channels, cond_channels, 3, 1)
            for _ in range(condition_count)
        ])
        self.output_convs = nn.ModuleList([
            DCC(cond_channels, channels * 2, 3, 1)
            for _ in range(condition_count)
        ])
        for convolution in (*self.input_convs, *self.output_convs):
            nn.init.xavier_uniform_(convolution.conv.weight)
            nn.init.zeros_(convolution.conv.bias)

    def forward(self, *conditions):
        if len(conditions) != len(self.input_convs):
            raise ValueError("FiLM condition count differs from its configured inputs")
        shifts = []
        scales = []
        for input_conv, output_conv, condition in zip(
                self.input_convs,
                self.output_convs,
                conditions):
            condition = input_conv(condition)
            condition = F.leaky_relu(condition, 0.2)
            shift, scale = output_conv(condition).chunk(2, dim=1)
            shifts.append(shift)
            scales.append(scale)
        return sum(shifts), sum(scales)


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4):
        super().__init__()
        self.factor = factor

        self.down_res = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = DCC(input_channels, input_channels, 3, 1)
        self.c2 = DCC(input_channels, input_channels, 3, 2)
        self.c3 = DCC(input_channels, output_channels, 3, 4)

    def forward(self, x):
        x = F.interpolate(x, size=x.shape[-1] // self.factor, mode='nearest')
        res = self.down_res(x)
        x = F.leaky_relu(x, 0.2)
        x = self.c1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.c2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.c3(x)
        return x + res


class Upsample(nn.Module):
    def __init__(self, input_channels, output_channels, cond_channels, factor=4):
        super().__init__()
        self.factor = factor

        self.residual = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = DCC(input_channels, output_channels, 3, 1)
        self.c2 = DCC(output_channels, output_channels, 3, 3)
        self.c3 = DCC(output_channels, output_channels, 3, 9)
        self.c4 = DCC(output_channels, output_channels, 3, 27)
        self.film = FiLM(output_channels, cond_channels)

    @staticmethod
    def affine(x, shift, scale):
        return shift + scale * x

    def forward(self, x, source_condition, loudness_condition):
        residual = F.interpolate(x, scale_factor=self.factor, mode='linear', align_corners=False)
        residual = self.residual(residual)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, scale_factor=self.factor, mode='linear', align_corners=False)
        x = self.c1(x)
        shift, scale = self.film(source_condition, loudness_condition)
        x = self.affine(x, shift, scale)
        x = F.leaky_relu(x, 0.2)
        x = self.c2(x)
        x = x + residual
        residual = x
        x = self.affine(x, shift, scale)
        x = F.leaky_relu(x, 0.2)
        x = self.c3(x)
        x = self.affine(x, shift, scale)
        x = F.leaky_relu(x, 0.2)
        x = self.c4(x)
        return x + residual



class Decoder(nn.Module):
    def __init__(self,
                 channels=(192, 96, 48, 24),
                 factors=(6, 4, 4, 5),
                 cond_channels=(192, 96, 48, 24),
                 content_channels=768,
                 sample_rate=24000,
                 frame_size=480,
                 loudness_frame_size=96,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.loudness_frame_size = loudness_frame_size
        self.loudness_n_fft = 768
        self.content_channels = content_channels

        self.source_downs = nn.ModuleList([])
        self.loudness_downs = nn.ModuleList([])
        condition_channels = list(reversed(cond_channels))
        condition_inputs = [1] + condition_channels[:-1]
        condition_factors = [1] + list(reversed(factors[1:]))
        for input_channels, output_channels, factor in zip(
                condition_inputs,
                condition_channels,
                condition_factors):
            self.source_downs.append(Downsample(input_channels, output_channels, factor))
            self.loudness_downs.append(Downsample(input_channels, output_channels, factor))

        self.content_in = nn.Conv1d(content_channels, channels[0], 1)

        # initialize upsample layers
        self.ups = nn.ModuleList([])
        up_inputs = [channels[0]] + list(channels[:-1])
        for input_channels, output_channels, condition_channels, factor in zip(
                up_inputs,
                channels,
                cond_channels,
                factors):
            self.ups.append(Upsample(
                input_channels,
                output_channels,
                condition_channels,
                factor,
            ))
        # output layer
        self.output_layer = DCC(channels[-1], 1, 3, 1)
        self.apply(initialize_wavegrad_convolution)
        for upsample in self.ups:
            for convolution in (*upsample.film.input_convs, *upsample.film.output_convs):
                nn.init.xavier_uniform_(convolution.conv.weight)
                nn.init.zeros_(convolution.conv.bias)

    def generate_source(self, p):
        initial_phase = torch.rand(p.shape[0], 1, 1, device=p.device, dtype=p.dtype)
        source_signal, _ = generate_excitation(
            p,
            phase=initial_phase,
            frame_size=self.frame_size,
            sample_rate=self.sample_rate,
        )
        return source_signal

    def forward(self, x, e, source_signals):
        expected_length = x.shape[2] * self.frame_size
        if not torch.jit.is_tracing() and source_signals.shape != (x.shape[0], 1, expected_length):
            raise ValueError("source excitation length differs from content timing")
        loudness_signal = F.interpolate(
            e,
            size=expected_length,
            mode="linear",
            align_corners=False,
        )

        source_skips = []
        loudness_skips = []
        source = source_signals
        loudness = loudness_signal
        for source_down, loudness_down in zip(self.source_downs, self.loudness_downs):
            source = source_down(source)
            loudness = loudness_down(loudness)
            source_skips.append(source)
            loudness_skips.append(loudness)

        x = self.content_in(x)

        for upsample, source_skip, loudness_skip in zip(
                self.ups,
                reversed(source_skips),
                reversed(loudness_skips)):
            x = upsample(x, source_skip, loudness_skip)

        x = self.output_layer(x)
        x = x.squeeze(1)
        return x

    def synthesize(self, x, p, e):
        source_signals = self.generate_source(p)
        out = self.forward(x, e, source_signals)
        return out
