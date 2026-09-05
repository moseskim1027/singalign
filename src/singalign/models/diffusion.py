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


class ScoreConditionedMelDiffusion(nn.Module):
    """Study 1 scaffold: diffuse mel frames conditioned on a musical score.

    The discrete phoneme and MIDI streams, plus observed F0 and normalized
    frame timing, are projected into the frame-aligned conditioning expected by
    :class:`ConditionalMelDiffusion`. This defines the future model boundary;
    it does not include training, sampling, or a vocoder.
    """

    def __init__(
        self,
        mel_bins: int = 80,
        phoneme_vocab_size: int = 256,
        condition_channels: int = 256,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.phoneme = nn.Embedding(phoneme_vocab_size, 64, padding_idx=0)
        self.midi_pitch = nn.Embedding(129, 64, padding_idx=0)
        self.continuous = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64))
        self.condition = nn.Conv1d(192, condition_channels, kernel_size=1)
        self.denoiser = ConditionalMelDiffusion(
            mel_bins=mel_bins,
            condition_channels=condition_channels,
            hidden_channels=hidden_channels,
        )

    def forward(
        self,
        noisy_mel: torch.Tensor,
        timestep: torch.Tensor,
        phoneme_ids: torch.Tensor,
        midi_pitch: torch.Tensor,
        observed_f0: torch.Tensor,
        frame_position: torch.Tensor,
    ) -> torch.Tensor:
        streams = (phoneme_ids, midi_pitch, observed_f0, frame_position)
        if any(stream.ndim != 2 for stream in streams):
            raise ValueError("Study 1 conditioning streams must be [batch, frames]")
        if len({stream.shape for stream in streams}) != 1:
            raise ValueError("Study 1 conditioning streams must share shape")
        if noisy_mel.shape[0] != phoneme_ids.shape[0] or noisy_mel.shape[2] != phoneme_ids.shape[1]:
            raise ValueError("noisy_mel must align with Study 1 conditioning streams")
        discrete = torch.cat(
            (self.phoneme(phoneme_ids.long()), self.midi_pitch(midi_pitch.long().clamp(0, 128))),
            dim=-1,
        )
        continuous = self.continuous(torch.stack((observed_f0.float(), frame_position.float()), dim=-1))
        condition = self.condition(torch.cat((discrete, continuous), dim=-1).transpose(1, 2))
        return self.denoiser(noisy_mel, timestep, condition)


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
