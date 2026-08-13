import math

import torch


def interpolate_f0(f0, frame_size, previous_f0=None):
    if previous_f0 is None:
        previous_f0 = f0[:, :, :1]
    if previous_f0.shape != f0[:, :, :1].shape:
        raise ValueError("previous f0 must contain one frame per batch")

    frame_starts = torch.cat([previous_f0, f0[:, :, :-1]], dim=2)
    interpolation = torch.arange(
        1,
        frame_size + 1,
        device=f0.device,
        dtype=f0.dtype,
    ).view(1, 1, 1, frame_size) / frame_size
    audio_rate_f0 = (
        frame_starts.unsqueeze(3)
        + (f0 - frame_starts).unsqueeze(3) * interpolation
    )
    return audio_rate_f0.flatten(2)


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
        previous_f0=None,
        ):
    if f0.ndim != 3 or f0.shape[1] != 1:
        raise ValueError("f0 must have shape [batch, 1, frames]")

    audio_rate_f0 = interpolate_f0(f0, frame_size, previous_f0)
    voiced = (audio_rate_f0 >= voiced_threshold).to(f0.dtype)

    phase_increments = audio_rate_f0 / sample_rate
    integrated_phase = torch.cat([
        torch.zeros_like(phase_increments[:, :, :1]),
        torch.cumsum(phase_increments[:, :, :-1], dim=2),
    ], dim=2)
    cycle_phase = (integrated_phase + phase) % 1
    sine = torch.sin(2 * math.pi * cycle_phase)

    if noise is None:
        noise = torch.randn_like(sine)
    if noise.shape != sine.shape:
        raise ValueError("noise must match the generated excitation shape")

    noise_scale = voiced * voiced_noise_std + (1 - voiced) * unvoiced_noise_std
    excitation = voiced * sine_amplitude * sine + noise_scale * noise
    next_phase = (phase + phase_increments.sum(dim=2, keepdim=True)) % 1
    return excitation, next_phase
