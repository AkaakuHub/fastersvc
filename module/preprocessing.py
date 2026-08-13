import torch
import torch.nn.functional as F


def trim_silence(waveform, top_db=30.0, frame_size=2048, hop_size=512):
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, samples]")
    if top_db <= 0:
        raise ValueError("silence threshold must be positive")
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("silence analysis sizes must be positive")
    if waveform.shape[1] == 0:
        return waveform

    half_frame = frame_size // 2
    padding_mode = "reflect" if waveform.shape[1] > half_frame else "constant"
    padded = F.pad(waveform, (half_frame, half_frame), mode=padding_mode)
    frames = padded.unfold(1, frame_size, hop_size)
    frame_rms = frames.square().mean(dim=(0, 2)).sqrt()
    maximum_rms = frame_rms.max()
    if maximum_rms == 0:
        return waveform[:, :0]

    active_frames = torch.nonzero(
        frame_rms >= maximum_rms * 10.0 ** (-top_db / 20.0),
        as_tuple=False,
    ).flatten()
    start = max(0, int(active_frames[0]) * hop_size - half_frame)
    end = min(
        waveform.shape[1],
        int(active_frames[-1]) * hop_size + half_frame,
    )
    return waveform[:, start:end]


def split_complete_segments(waveform, segment_size):
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, samples]")
    if segment_size <= 0:
        raise ValueError("segment size must be positive")
    complete_length = waveform.shape[1] // segment_size * segment_size
    return waveform[:, :complete_length].split(segment_size, dim=1)
