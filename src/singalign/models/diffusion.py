"""Architecture contract for a future singing voice diffusion model.

The module below is deliberately small and untrained.  It provides a stable
conditioning and tensor-shape contract for future GPU experiments; it is not a
voice-conversion model or an audio sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encode a diffusion step using the standard sinusoidal representation."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = torch.exp(
            -torch.log(torch.tensor(10000.0, device=timestep.device))
            * torch.arange(half, device=timestep.device)
            / max(half - 1, 1)
        )
        angles = timestep.float().unsqueeze(1) * scale.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.dimension % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


class ConditionalResidualBlock(nn.Module):
    """A compact 1-D residual block with frame-wise conditioning."""

    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.condition = nn.Conv1d(condition_channels, channels, kernel_size=1)
        self.output = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        update = self.input(hidden) + self.condition(condition)
        update = self.output(self.activation(update))
        return hidden + update


class ConditionalMelDiffusion(nn.Module):
    """Generic conditional denoiser for a singing-vocal mel spectrogram.

    Inputs use ``[batch, channels, frames]`` layout. ``condition`` is a
    frame-aligned representation combining content/phonemes, target F0,
    timing, and singer identity. The output is predicted diffusion noise.
    """

    def __init__(
        self,
        mel_bins: int = 80,
        condition_channels: int = 256,
        hidden_channels: int = 128,
        time_embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.mel_bins = mel_bins
        self.condition_channels = condition_channels
        self.input = nn.Conv1d(mel_bins, hidden_channels, kernel_size=1)
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
        )
        self.blocks = nn.ModuleList(
            [ConditionalResidualBlock(hidden_channels, condition_channels) for _ in range(4)]
        )
        self.output = nn.Sequential(nn.SiLU(), nn.Conv1d(hidden_channels, mel_bins, kernel_size=1))

    def forward(
        self,
        noisy_mel: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_mel.ndim != 3 or condition.ndim != 3:
            raise ValueError("noisy_mel and condition must be [batch, channels, frames]")
        if noisy_mel.shape[0] != condition.shape[0] or noisy_mel.shape[2] != condition.shape[2]:
            raise ValueError("noisy_mel and condition must share batch and frame dimensions")
        if condition.shape[1] != self.condition_channels:
            raise ValueError(f"condition must have {self.condition_channels} channels")
        hidden = self.input(noisy_mel)
        hidden = hidden + self.time(timestep).unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)


@dataclass(frozen=True)
class SingingVoiceDiffusionSpec:
    """Declare the conditioning contract without allocating a model."""

    mel_bins: int = 80
    conditioning_rate_hz: int = 100
    diffusion_steps: int = 1000
    content_condition: str = "phoneme-or-content-encoder"
    pitch_condition: str = "target-f0-contour"
    timing_condition: str = "score-frame-timing"
    timbre_condition: str = "singer-embedding"
    status: str = "architecture-only-not-trained"
