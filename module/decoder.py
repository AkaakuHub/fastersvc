import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DCC
from .excitation import generate_excitation


class FiLM(nn.Module):
    def __init__(self, channels, cond_channels, condition_count=2):
        super().__init__()
        self.to_mu = nn.ModuleList([
            nn.Conv1d(cond_channels, channels, 1)
            for _ in range(condition_count)
        ])
        self.to_sigma = nn.ModuleList([
            nn.Conv1d(cond_channels, channels, 1)
            for _ in range(condition_count)
        ])

    def forward(self, x, *conditions):
        if len(conditions) != len(self.to_mu):
            raise ValueError("FiLM condition count differs from its configured inputs")
        mu = sum(layer(condition) for layer, condition in zip(self.to_mu, conditions))
        sigma = sum(layer(condition) for layer, condition in zip(self.to_sigma, conditions))
        x = x * mu + sigma
        return x


class Downsample(nn.Module):
    def __init__(self, input_channels, output_channels, factor=4):
        super().__init__()
        self.factor = factor

        self.pool = nn.AvgPool1d(factor)
        self.down_res = nn.Conv1d(input_channels, output_channels, 1)
        self.c1 = DCC(input_channels, input_channels, 3, 1)
        self.c2 = DCC(input_channels, input_channels, 3, 2)
        self.c3 = DCC(input_channels, output_channels, 3, 4)

    def forward(self, x):
        x = self.pool(x)
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

        self.c1 = DCC(input_channels, input_channels, 3, 1)
        self.c2 = DCC(input_channels, input_channels, 3, 3)
        self.film1 = FiLM(input_channels, cond_channels)
        self.c3 = DCC(input_channels, input_channels, 3, 9)
        self.c4 = DCC(input_channels, input_channels, 3, 27)
        self.film2 = FiLM(input_channels, cond_channels)
        self.c5 = DCC(input_channels, output_channels, 3, 1)

    def forward(self, x, source_condition, loudness_condition):
        source_condition = F.interpolate(
            source_condition,
            scale_factor=self.factor,
            mode='linear',
            align_corners=False,
        )
        loudness_condition = F.interpolate(
            loudness_condition,
            scale_factor=self.factor,
            mode='linear',
            align_corners=False,
        )
        x = F.interpolate(x, scale_factor=self.factor, mode='linear', align_corners=False)
        res = x
        x = F.leaky_relu(x, 0.2)
        x = self.c1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.c2(x)
        x = self.film1(x, source_condition, loudness_condition)
        x = x + res
        res = x
        x = F.leaky_relu(x, 0.2)
        x = self.c3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.c4(x)
        x = self.film2(x, source_condition, loudness_condition)
        x = x + res
        x = self.c5(x)
        return x



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

        self.source_down_input = nn.Conv1d(1, cond_channels[-1], 1)
        self.loudness_down_input = nn.Conv1d(1, cond_channels[-1], 1)
        self.source_downs = nn.ModuleList([])
        self.loudness_downs = nn.ModuleList([])
        cond = list(reversed(cond_channels))
        cond_next = cond[1:] + [cond[-1]]
        for c, c_n, f in zip(cond, cond_next, reversed(factors)):
            self.source_downs.append(Downsample(c, c_n, f))
            self.loudness_downs.append(Downsample(c, c_n, f))

        self.content_in = nn.Conv1d(content_channels, channels[0], 1)

        # initialize upsample layers
        self.ups = nn.ModuleList([])
        up = channels
        up_next = list(channels[1:]) + [channels[-1]]
        for u, u_n, c_n, f in zip(up, up_next, reversed(cond_next), factors):
            self.ups.append(Upsample(u, u_n, c_n, f))
        # output layer
        self.output_layer = DCC(channels[-1], 1, 3, 1)

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
        if source_signals.shape != (x.shape[0], 1, expected_length):
            raise ValueError("source excitation length differs from content timing")
        loudness_signal = F.interpolate(
            e,
            size=expected_length,
            mode="linear",
            align_corners=False,
        )

        source_skips = []
        loudness_skips = []
        source = self.source_down_input(source_signals)
        loudness = self.loudness_down_input(loudness_signal)
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
        x = torch.tanh(x).squeeze(1)
        return x

    def synthesize(self, x, p, e):
        source_signals = self.generate_source(p)
        out = self.forward(x, e, source_signals)
        return out
