"""Small symbolic-conditioned mel baseline for the next research stage."""

from __future__ import annotations

import torch
from torch import nn


class ScoreConditionedMelModel(nn.Module):
    """Map frame-aligned MIDI pitch and phoneme IDs to a mel spectrogram."""

    def __init__(self, mel_bins: int = 80, phoneme_vocab_size: int = 256) -> None:
        super().__init__()
        if mel_bins < 1 or phoneme_vocab_size < 2:
            raise ValueError("mel_bins must be positive and vocabulary >= 2")
        self.pitch = nn.Embedding(129, 32, padding_idx=0)
        self.phoneme = nn.Embedding(phoneme_vocab_size, 32, padding_idx=0)
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, mel_bins, 5, padding=2),
        )

    def forward(self, midi_pitch: torch.Tensor, phoneme_ids: torch.Tensor) -> torch.Tensor:
        if midi_pitch.shape != phoneme_ids.shape or midi_pitch.ndim != 2:
            raise ValueError("conditioning tensors must both have shape (batch, frames)")
        features = torch.cat(
            (self.pitch(midi_pitch.clamp(0, 128)), self.phoneme(phoneme_ids)), dim=-1
        )
        return self.decoder(features.transpose(1, 2))
