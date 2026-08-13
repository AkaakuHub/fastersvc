import math

import torch
import torch.nn.functional as F


def generate_excitation(
        f0,
        phase=0,
        frame_size=480,
        sample_rate=24000,
        voiced_threshold=10.0,
        sine_amplitude=0.1,
        voiced_noise_std=0.003,
        unvoiced_noise_std=0.3,
        noise=None,
        ):
    if f0.ndim != 3 or f0.shape[1] != 1:
        raise ValueError("f0 must have shape [batch, 1, frames]")

    waveform_length = f0.shape[2] * frame_size
    audio_rate_f0 = F.interpolate(
        f0,
        waveform_length,
        mode="linear",
        align_corners=False,
    )
    voiced = F.interpolate(
        (f0 >= voiced_threshold).to(f0.dtype),
        waveform_length,
        mode="nearest",
    )

    integrated_phase = torch.cumsum(audio_rate_f0 / sample_rate, dim=2)
    integrated_phase = integrated_phase - integrated_phase[:, :, :1]
    cycle_phase = (integrated_phase + phase) % 1
    sine = torch.sin(2 * math.pi * cycle_phase)

    if noise is None:
        noise = torch.randn_like(sine)
    if noise.shape != sine.shape:
        raise ValueError("noise must match the generated excitation shape")

    noise_scale = voiced * voiced_noise_std + (1 - voiced) * unvoiced_noise_std
    excitation = voiced * sine_amplitude * sine + noise_scale * noise
    next_phase = (cycle_phase[:, :, -1:] + audio_rate_f0[:, :, -1:] / sample_rate) % 1
    return excitation, next_phase
