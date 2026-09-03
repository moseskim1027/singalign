"""Objective metrics and uncertainty estimates for SingAlign evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ConfidenceInterval:
    """A sample mean and percentile-bootstrap confidence interval."""

    mean: float
    lower: float
    upper: float
    confidence_level: float


def reconstruction_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Compute per-example errors on normalized log-mel spectrograms."""

    difference = prediction - target
    return {
        "log_mel_mse": F.mse_loss(prediction, target).item(),
        "log_mel_mae": F.l1_loss(prediction, target).item(),
        "spectral_convergence": (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(target).clamp_min(1e-8)
        ).item(),
    }


def bootstrap_mean_interval(
    values: list[float],
    seed: int,
    samples: int = 2000,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Estimate a deterministic percentile-bootstrap interval for a mean."""

    if not values:
        raise ValueError("values must not be empty")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    observed = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(observed), size=(samples, len(observed)))
    bootstrap_means = observed[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return ConfidenceInterval(
        mean=float(observed.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
    )


def paired_summary(
    baseline: list[float],
    aligned: list[float],
    seed: int,
    samples: int,
    confidence_level: float,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    """Summarize paired aligned-minus-baseline changes for lower-is-better metrics."""

    if len(baseline) != len(aligned) or not baseline:
        raise ValueError("paired values must have equal non-zero lengths")
    deltas = [new - old for old, new in zip(baseline, aligned, strict=True)]
    interval = bootstrap_mean_interval(deltas, seed, samples, confidence_level)
    return {
        "baseline_mean": float(np.mean(baseline)),
        "aligned_mean": float(np.mean(aligned)),
        "delta": {
            "mean": interval.mean,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence_level": interval.confidence_level,
        },
        "wins": sum(delta < -tolerance for delta in deltas),
        "ties": sum(abs(delta) <= tolerance for delta in deltas),
        "losses": sum(delta > tolerance for delta in deltas),
    }
