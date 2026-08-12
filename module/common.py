import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.functional import resample
import numpy as np
import pyworld as pw


# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def spectrogram(wave, n_fft, hop_size):
    dtype = wave.dtype
    wave = wave.to(torch.float)
    window = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft, hop_size, return_complex=True, window=window).abs()
    spec = spec[:, :, 1:]
    spec = spec.to(dtype)
    return spec

# wave: [BatchSize, 1, Length]
# Output: [BatchSize, 1, Frames]
def energy(wave,
           frame_size=480):
    return F.max_pool1d((wave.abs()).unsqueeze(1), frame_size)


# Convert style based kNN.
# Warning: this method is not optimized.
# Do not give long sequence. computing complexy is quadratic.
# 
# source: [BatchSize, Channels, Length]
# reference: [BatchSize, Channels, Length]
# k: int
# alpha: float (0.0 ~ 1.0)
# metrics: one of ['IP', 'L2', 'cos'], 'IP' means innner product, 'L2' means euclid distance, 'cos' means cosine similarity
# Output: [BatchSize, Channels, Length]
def match_features(source, reference, k=4, alpha=0.0, metrics='cos'):
    if source.ndim != 3 or reference.ndim != 3:
        raise ValueError("source and reference must have shape [batch, channels, frames]")
    if source.shape[0] != reference.shape[0] or source.shape[1] != reference.shape[1]:
        raise ValueError("source and reference batch and channel dimensions must match")
    if not 1 <= k <= reference.shape[2]:
        raise ValueError("k must not exceed the number of reference frames")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if metrics not in {'IP', 'L2', 'cos'}:
        raise ValueError(f"unsupported feature metric: {metrics}")

    input_data = source

    source = source.transpose(1, 2)
    reference = reference.transpose(1, 2)
    if metrics == 'IP':
        sims = torch.bmm(source, reference.transpose(1, 2))
    elif metrics == 'L2':
        sims = -torch.cdist(source, reference)
    else:
        reference_norm = torch.norm(reference, dim=2, keepdim=True) + 1e-6
        source_norm = torch.norm(source, dim=2, keepdim=True) + 1e-6
        sims = torch.bmm(source / source_norm, (reference / reference_norm).transpose(1, 2))
    nearest_indices = torch.topk(sims, k, dim=2).indices
    batch_size, source_frames, _ = nearest_indices.shape
    channels = reference.shape[2]
    candidates = reference.unsqueeze(1).expand(-1, source_frames, -1, -1)
    gather_indices = nearest_indices.unsqueeze(3).expand(batch_size, source_frames, k, channels)
    result = torch.gather(candidates, 2, gather_indices).mean(dim=2)
    result = result.transpose(1, 2)
    return result * (1-alpha) + input_data * alpha


# Dlilated Causal Convolution
class DCC(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 dilation=1,
                 groups=1,
                 weight_norm=False
                 ):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation, groups=groups)
        self.pad_size = (kernel_size - 1) * dilation
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        x = F.pad(x, [self.pad_size, 0], mode='replicate')
        x = self.conv(x)
        return x


class ChannelNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mu) / sigma
        x = x * self.scale + self.shift
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, mlp_mul=1, norm=False, negative_slope=0.1):
        super().__init__()
        self.c1 = DCC(channels, channels, kernel_size, dilation)
        self.norm = ChannelNorm(channels) if norm else nn.Identity()
        self.c2 = nn.Conv1d(channels, channels * mlp_mul, 1)
        self.c3 = nn.Conv1d(channels * mlp_mul, channels, 1)
        self.negative_slope = negative_slope

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.leaky_relu(x, self.negative_slope)
        x = self.c3(x)
        return x + res


def compute_f0_dio(wf, sample_rate=24000, segment_size=480, f0_min=50, f0_max=1100):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        _f0, t = pw.dio(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
        f0 = pw.stonemask(signal, _f0, t, sample_rate)
        f0 = torch.from_numpy(f0).to(torch.float)
        f0 = f0.to(device)
        f0 = f0.unsqueeze(0).unsqueeze(0)
        f0 = F.interpolate(f0, wf.shape[0] // segment_size, mode='linear')
        f0 = f0.squeeze(0)
        return f0
    elif wf.ndim == 2:
        waves = wf.split(1, dim=0)
        pitchs = [compute_f0_dio(wave[0], sample_rate, segment_size, f0_min, f0_max) for wave in waves]
        pitchs = torch.stack(pitchs, dim=0)
        return pitchs


def compute_f0_harvest(wf, sample_rate=24000, segment_size=480, f0_min=50, f0_max=1100):
    if wf.ndim == 1:
        device = wf.device
        signal = wf.detach().cpu().numpy()
        signal = signal.astype(np.double)
        f0, t = pw.harvest(signal, sample_rate, f0_floor=f0_min, f0_ceil=f0_max)
        f0 = torch.from_numpy(f0).to(torch.float)
        f0 = f0.to(device)
        f0 = f0.unsqueeze(0).unsqueeze(0)
        f0 = F.interpolate(f0, wf.shape[0] // segment_size, mode='linear')
        f0 = f0.squeeze(0)
        return f0
    elif wf.ndim == 2:
        waves = wf.split(1, dim=0)
        pitchs = [compute_f0_harvest(wave[0], sample_rate, segment_size, f0_min, f0_max) for wave in waves]
        pitchs = torch.stack(pitchs, dim=0)
        return pitchs


def compute_f0(wf, sample_rate=24000, segment_size=480, algorithm='harvest'):
    l = wf.shape[1]
    wf = resample(wf, sample_rate, 16000)
    if algorithm == 'harvest':
        pitchs = compute_f0_harvest(wf, 16000)
    elif algorithm == 'dio':
        pitchs = compute_f0_dio(wf, 16000)
    else:
        raise ValueError(f"unsupported pitch estimation algorithm: {algorithm}")
    return F.interpolate(pitchs, l // segment_size, mode='linear')
