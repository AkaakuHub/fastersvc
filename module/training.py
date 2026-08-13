import os

import torch
import torch.nn as nn


DEFAULT_DECODER_LEARNING_RATE = 1e-3
DECODER_OPTIMIZER_EPSILON = 1e-6
DECODER_GRADIENT_NORM = 10.0


def training_data_loader(dataset, batch_size, workers, device):
    if workers < 0:
        raise ValueError("data loader worker count must not be negative")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def step_scaled_optimizer(loss, optimizer, scaler, parameters, max_gradient_norm):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    scaler.step(optimizer)
    return gradient_norm


def atomic_save(value, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary_path = f"{path}.saving"
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def learning_rate_at_step(initial_learning_rate, step, interval=100000, decay=0.5):
    return initial_learning_rate * decay ** (step // interval)


def set_optimizer_learning_rate(optimizer, learning_rate):
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def crop_aligned_batch(waveforms, pitch, sample_count=24000, frame_size=480):
    if waveforms.ndim != 2 or pitch.ndim != 3 or pitch.shape[1] != 1:
        raise ValueError("waveforms and pitch must have batch-first waveform and frame shapes")
    if waveforms.shape[0] != pitch.shape[0]:
        raise ValueError("waveform and pitch batch sizes differ")
    if sample_count % frame_size != 0:
        raise ValueError("sample count must be a multiple of frame size")
    if waveforms.shape[1] != pitch.shape[2] * frame_size:
        raise ValueError("waveform and pitch frame counts differ")
    if waveforms.shape[1] < sample_count:
        raise ValueError("training waveform is shorter than the requested crop")

    output_frames = sample_count // frame_size
    maximum_start_frame = pitch.shape[2] - output_frames
    start_frames = torch.randint(
        maximum_start_frame + 1,
        (waveforms.shape[0], 1),
        device=waveforms.device,
    )
    waveform_indices = start_frames * frame_size + torch.arange(sample_count, device=waveforms.device)
    pitch_indices = start_frames.unsqueeze(1) + torch.arange(output_frames, device=pitch.device)
    return torch.gather(waveforms, 1, waveform_indices), torch.gather(pitch, 2, pitch_indices)
