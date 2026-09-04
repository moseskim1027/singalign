"""Deterministic candidate generation and provenance for the sandbox."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from singalign.preferences import degrade


@dataclass(frozen=True)
class Candidate:
    """One generated mel candidate and the parameters that produced it."""

    method: str
    seed: int
    severity: float
    features: torch.Tensor

    def provenance(self) -> dict[str, str | int | float]:
        """Return JSON-compatible candidate metadata without tensor data."""
        metadata = asdict(self)
        metadata.pop("features")
        return metadata


def generate_candidates(
    reference: torch.Tensor,
    seed: int,
    severities: tuple[float, ...] = (0.0, 0.1, 0.2),
) -> list[Candidate]:
    """Generate one identity and deterministic perturbation per degradation family."""
    if reference.ndim < 2:
        raise ValueError("reference must contain frequency and time dimensions")
    if not severities or any(not 0.0 <= value <= 1.0 for value in severities):
        raise ValueError("severities must be non-empty and between zero and one")
    candidates = [Candidate("identity", seed, 0.0, reference.clone())]
    for index, family in enumerate(("noise", "time_mask", "frequency_mask")):
        severity = severities[min(index, len(severities) - 1)]
        generator = torch.Generator().manual_seed(seed + index)
        candidates.append(Candidate(family, seed + index, severity, degrade(reference, family, severity, generator)))
    return candidates
