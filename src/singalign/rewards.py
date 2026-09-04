"""Transparent proxy rewards for comparing generated music candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardBreakdown:
    """Named reward components and their weighted aggregate."""

    components: dict[str, float]
    weights: dict[str, float]
    total: float

    def provenance(self) -> dict[str, object]:
        return {"components": self.components, "weights": self.weights, "total": self.total}


def _error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes must match")
    denominator = reference.square().mean().clamp_min(1e-8)
    return float((torch.mean((candidate - reference) ** 2) / denominator).item())


def scalar_reward(reference: torch.Tensor, candidate: torch.Tensor) -> RewardBreakdown:
    """Return the initial single-objective reconstruction proxy reward."""
    error = _error(reference, candidate)
    return RewardBreakdown({"reconstruction": -error}, {"reconstruction": 1.0}, -error)


def multidimensional_reward(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> RewardBreakdown:
    """Combine reconstruction, smoothness, and amplitude proxy components."""
    chosen = weights or {"reconstruction": 0.7, "smoothness": 0.2, "amplitude": 0.1}
    if set(chosen) != {"reconstruction", "smoothness", "amplitude"}:
        raise ValueError("weights must contain reconstruction, smoothness, and amplitude")
    if any(value < 0 for value in chosen.values()) or sum(chosen.values()) <= 0:
        raise ValueError("weights must be non-negative with a positive total")
    reconstruction = _error(reference, candidate)
    smoothness = float(torch.mean(torch.abs(candidate[..., 1:] - candidate[..., :-1])).item())
    amplitude = float(torch.abs(candidate.abs().mean() - reference.abs().mean()).item())
    components = {"reconstruction": -reconstruction, "smoothness": -smoothness, "amplitude": -amplitude}
    total = sum(components[name] * chosen[name] for name in components) / sum(chosen.values())
    return RewardBreakdown(components, chosen, total)
