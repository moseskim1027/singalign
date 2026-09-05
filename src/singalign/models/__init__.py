"""SingAlign research models."""

from singalign.models.baseline import MelAutoencoder
from singalign.models.conditioned import ScoreConditionedMelModel
from singalign.models.diffusion import (
    ConditionalMelDiffusion,
    ScoreConditionedMelDiffusion,
    SingingVoiceDiffusionSpec,
)
from singalign.models.vocoder import MelVocoder

__all__ = [
    "MelAutoencoder",
    "MelVocoder",
    "ScoreConditionedMelModel",
    "ConditionalMelDiffusion",
    "ScoreConditionedMelDiffusion",
    "SingingVoiceDiffusionSpec",
]
