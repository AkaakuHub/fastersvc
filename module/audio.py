import torch


def normalize_ssl_input(waveforms, epsilon=1e-7):
    if waveforms.ndim != 2:
        raise ValueError("waveforms must have shape [batch, samples]")
    mean = waveforms.mean(dim=1, keepdim=True)
    variance = waveforms.var(dim=1, keepdim=True, unbiased=False)
    return (waveforms - mean) / torch.sqrt(variance + epsilon)
