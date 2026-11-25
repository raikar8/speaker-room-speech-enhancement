import torch
from torch import nn


class STFT(nn.Module):
    """sqrt-Hann STFT / iSTFT (20 ms window, 10 ms hop at 16 kHz by default)."""

    def __init__(self, n_fft=320, hop=160):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft).sqrt(), persistent=False)

    def forward(self, wav):  # (B, L) -> (B, F, T) complex
        return torch.stft(wav.float(), self.n_fft, self.hop, window=self.window,
                          center=True, return_complex=True)

    def inverse(self, spec, length):
        return torch.istft(spec, self.n_fft, self.hop, window=self.window, center=True, length=length)


def cmag(x, eps=1e-8):
    """Magnitude of a complex tensor with a safe gradient at zero."""
    return torch.sqrt(x.real ** 2 + x.imag ** 2 + eps)
