"""Dependency-light audio loading and log-mel feature extraction."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_audio(path: Path, target_sample_rate: int) -> torch.Tensor:
    """Load a mono WAV, normalize its integer encoding, and resample it."""

    sample_rate, samples = wavfile.read(path)
    if samples.ndim == 2:
        samples = samples.astype(np.float64).mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        limits = np.iinfo(samples.dtype)
        scale = float(max(abs(limits.min), limits.max))
        samples = samples.astype(np.float32) / scale
    else:
        samples = samples.astype(np.float32)
    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        samples = resample_poly(
            samples, target_sample_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(samples)).clamp(-1.0, 1.0)


def crop_or_pad(waveform: torch.Tensor, length: int, offset: int = 0) -> torch.Tensor:
    """Return an exact-length mono segment."""

    if length < 1:
        raise ValueError("length must be positive")
    if waveform.numel() >= length:
        start = min(max(offset, 0), waveform.numel() - length)
        return waveform[start : start + length]
    return torch.nn.functional.pad(waveform, (0, length - waveform.numel()))


def _hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mels / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, mel_bins: int) -> torch.Tensor:
    """Construct a triangular mel filter bank."""

    frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    mel_edges = torch.linspace(
        _hz_to_mel(torch.tensor(0.0)),
        _hz_to_mel(torch.tensor(sample_rate / 2)),
        mel_bins + 2,
    )
    hz_edges = _mel_to_hz(mel_edges)
    lower = hz_edges[:-2, None]
    center = hz_edges[1:-1, None]
    upper = hz_edges[2:, None]
    rising = (frequencies - lower) / (center - lower).clamp_min(1e-8)
    falling = (upper - frequencies) / (upper - center).clamp_min(1e-8)
    return torch.minimum(rising, falling).clamp_min(0.0)


def log_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    mel_bins: int,
) -> torch.Tensor:
    """Convert a mono waveform to a normalized log-mel spectrogram."""

    spectrum = (
        torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft, device=waveform.device),
            return_complex=True,
        )
        .abs()
        .square()
    )
    filters = mel_filterbank(sample_rate, n_fft, mel_bins).to(waveform.device)
    features = torch.log1p(filters @ spectrum)
    mean = features.mean()
    return (features - mean) / features.std().clamp_min(1e-6)
