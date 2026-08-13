import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_ssl_input(waveforms, epsilon=1e-7):
    if waveforms.ndim != 2:
        raise ValueError("waveforms must have shape [batch, samples]")
    mean = waveforms.mean(dim=1, keepdim=True)
    variance = waveforms.var(dim=1, keepdim=True, unbiased=False)
    return (waveforms - mean) / torch.sqrt(variance + epsilon)


def a_weighting(frequencies, minimum_db=-80.0):
    squared_frequencies = frequencies.square()
    numerator = (12194.0 ** 2) * squared_frequencies.square()
    denominator = (
        (squared_frequencies + 20.6 ** 2)
        * torch.sqrt(
            (squared_frequencies + 107.7 ** 2)
            * (squared_frequencies + 737.9 ** 2)
        )
        * (squared_frequencies + 12194.0 ** 2)
    )
    response = numerator / denominator
    weighting_db = 20.0 * torch.log10(response) + 2.0
    return weighting_db.clamp_min(minimum_db)


class PerceptualLoudness(nn.Module):
    def __init__(
            self,
            sample_rate=24000,
            frame_size=96,
            n_fft=768,
            range_db=80.0):
        super().__init__()
        self.frame_size = frame_size
        self.n_fft = n_fft
        self.range_db = range_db
        frequencies = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer(
            "weighting",
            torch.pow(10.0, a_weighting(frequencies) / 10.0).view(1, -1, 1),
            persistent=False,
        )

    def forward(self, waveforms):
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        if waveforms.shape[1] % self.frame_size != 0:
            raise ValueError("waveform length must be a multiple of frame size")

        audio = F.pad(waveforms.to(torch.float32), (self.n_fft // 2, self.n_fft // 2))
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.frame_size,
            window=self.window,
            center=False,
            return_complex=True,
        )
        weighted_power = spectrum.abs().square() * self.weighting
        average_power = weighted_power.mean(dim=1)
        minimum_power = 10.0 ** (-self.range_db / 10.0)
        loudness_db = 10.0 * torch.log10(average_power.clamp_min(minimum_power))
        loudness_db = loudness_db.clamp_min(-self.range_db)
        expected_frames = waveforms.shape[1] // self.frame_size
        return loudness_db[:, 1:expected_frames + 1].unsqueeze(1).to(waveforms.dtype)


def perceptual_loudness(
        waveforms,
        sample_rate=24000,
        frame_size=96,
        n_fft=768,
        range_db=80.0):
    extractor = PerceptualLoudness(sample_rate, frame_size, n_fft, range_db).to(waveforms.device)
    return extractor(waveforms)
