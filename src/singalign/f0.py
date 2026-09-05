"""Deterministic frame-level fundamental-frequency extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class F0Frame:
    """One fixed-rate F0 observation; frequency is null when unvoiced."""

    time_seconds: float
    f0_hz: float | None
    voiced: bool
    confidence: float


def extract_f0(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_rate: float = 100.0,
    min_hz: float = 70.0,
    max_hz: float = 1000.0,
    voicing_threshold: float = 0.35,
) -> tuple[F0Frame, ...]:
    """Estimate F0 with normalized autocorrelation on a deterministic grid.

    This is an engineering baseline, not a claim of state-of-the-art pitch
    tracking. It exposes confidence and never interpolates unvoiced frames.
    """

    if waveform.ndim != 1 or sample_rate < 1 or frame_rate <= 0:
        raise ValueError("waveform must be mono and rates must be positive")
    if not 0 < min_hz < max_hz or not 0 <= voicing_threshold <= 1:
        raise ValueError("invalid pitch or voicing range")
    hop = max(1, round(sample_rate / frame_rate))
    window = max(hop * 2, round(sample_rate * 0.04))
    min_lag = max(1, round(sample_rate / max_hz))
    max_lag = min(window - 1, round(sample_rate / min_hz))
    padded = torch.nn.functional.pad(waveform.float(), (window // 2, window // 2))
    frames: list[F0Frame] = []
    for start in range(0, waveform.numel(), hop):
        frame = padded[start : start + window]
        if frame.numel() < window:
            frame = torch.nn.functional.pad(frame, (0, window - frame.numel()))
        frame = frame - frame.mean()
        energy = frame.square().mean()
        if float(energy) <= 1e-8 or max_lag <= min_lag:
            frames.append(F0Frame(start / sample_rate, None, False, 0.0))
            continue
        values = torch.stack(
            [(frame[:-lag] * frame[lag:]).mean() for lag in range(min_lag, max_lag + 1)]
        )
        confidence = float(values.max() / energy.clamp_min(1e-8))
        lag = min_lag + int(values.argmax())
        voiced = confidence >= voicing_threshold
        frames.append(
            F0Frame(
                start / sample_rate,
                sample_rate / lag if voiced else None,
                voiced,
                min(confidence, 1.0),
            )
        )
    return tuple(frames)


def f0_payload(frames: tuple[F0Frame, ...]) -> list[dict[str, object]]:
    """Serialize F0 frames without lossy interpolation."""

    return [asdict(frame) for frame in frames]
