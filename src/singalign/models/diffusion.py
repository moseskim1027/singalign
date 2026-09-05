"""Configuration-only contract for a future singing voice diffusion model."""

from __future__ import annotations

from dataclasses import dataclass


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
    status: str = "placeholder-not-trained"

