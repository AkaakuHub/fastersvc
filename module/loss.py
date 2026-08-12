import torch
import torch.nn as nn


def safe_log(x):
    return torch.log(x.clamp_min(1e-6))


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
            self,
            fft_sizes=(64, 128, 256, 512, 1024, 2048),
            ):
        super().__init__()
        self.fft_sizes = fft_sizes
        for n_fft in fft_sizes:
            self.register_buffer(
                f"window_{n_fft}",
                torch.hann_window(n_fft),
                persistent=False,
            )

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        loss = x.new_tensor(0.0)
        for n_fft in self.fft_sizes:
            hop_length = n_fft // 4
            window = getattr(self, f"window_{n_fft}")
            x_spec = torch.stft(x, n_fft, hop_length, return_complex=True, window=window).abs()
            y_spec = torch.stft(y, n_fft, hop_length, return_complex=True, window=window).abs()
            spectral_convergence = torch.linalg.vector_norm(x_spec - y_spec) / torch.linalg.vector_norm(y_spec).clamp_min(1e-6)
            log_magnitude = (safe_log(x_spec) - safe_log(y_spec)).abs().mean()
            loss += spectral_convergence + log_magnitude
        return loss / len(self.fft_sizes)
