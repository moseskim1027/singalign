"""SingAlign research models."""

from singalign.models.baseline import MelAutoencoder
from singalign.models.conditioned import ScoreConditionedMelModel

__all__ = ["MelAutoencoder", "ScoreConditionedMelModel"]
