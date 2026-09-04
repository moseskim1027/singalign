"""Trainable mel-to-waveform decoder for the generation sandbox."""

from __future__ import annotations

import torch
from torch import nn


class MelVocoder(nn.Module):
    """Small waveform decoder with an explicit mel-frame hop size.

    This is a trainable baseline vocoder, not a pretrained production vocoder.
    It makes waveform generation differentiable for future method comparisons.
    """

    def __init__(self, mel_bins: int = 80, hop_length: int = 160) -> None:
        super().__init__()
        if mel_bins < 1 or hop_length < 1:
            raise ValueError("mel_bins and hop_length must be positive")
        self.hop_length = hop_length
        self.frontend = nn.Sequential(
            nn.Conv1d(mel_bins, 128, 5, padding=2), nn.GELU(),
            nn.Conv1d(128, 64, 5, padding=2), nn.GELU(),
        )
        self.upsample = nn.ConvTranspose1d(
            64, 1, kernel_size=2 * hop_length, stride=hop_length,
            padding=hop_length // 2,
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Decode ``(batch, mel_bins, frames)`` to ``(batch, samples)``."""
        if mel.ndim != 3:
            raise ValueError("mel input must have shape (batch, mel_bins, frames)")
        return torch.tanh(self.upsample(self.frontend(mel)).squeeze(1))
