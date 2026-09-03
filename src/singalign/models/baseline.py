"""Compact mel-spectrogram reconstruction baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MelAutoencoder(nn.Module):
    """A small convolutional autoencoder suitable for Apple Silicon."""

    def __init__(self, latent_channels: int = 32) -> None:
        super().__init__()
        if latent_channels < 1:
            raise ValueError("latent_channels must be positive")
        hidden = max(latent_channels // 2, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, latent_channels, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, 1, 4, stride=2, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decoder(self.encoder(inputs))
        if reconstruction.shape[-2:] != inputs.shape[-2:]:
            reconstruction = F.interpolate(
                reconstruction, size=inputs.shape[-2:], mode="bilinear"
            )
        return reconstruction
