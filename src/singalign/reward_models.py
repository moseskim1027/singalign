"""Small learned reward models for exploratory preference simulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RewardModelFit:
    """Diagnostics from one deterministic pairwise training run."""

    losses: tuple[float, ...]


class RewardModel(nn.Module):
    """Predict a scalar reward from a fixed-shape feature tensor."""

    def __init__(self, feature_shape: tuple[int, ...], hidden_size: int = 64) -> None:
        super().__init__()
        if not feature_shape or any(size <= 0 for size in feature_shape):
            raise ValueError("feature_shape must contain positive dimensions")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.feature_shape = feature_shape
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.tensor(feature_shape).prod().item()), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if tuple(features.shape[-len(self.feature_shape) :]) != self.feature_shape:
            raise ValueError("features do not match the configured feature_shape")
        return self.network(features).squeeze(-1)


class MultiRewardModel(nn.Module):
    """Predict named reward components while retaining a weighted total."""

    def __init__(self, feature_shape: tuple[int, ...], components: tuple[str, ...], hidden_size: int = 64) -> None:
        super().__init__()
        if not components or len(set(components)) != len(components):
            raise ValueError("components must be non-empty and unique")
        self.components = components
        self.model = RewardModel(feature_shape, hidden_size)
        self.head = nn.Linear(1, len(components))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.model(features).unsqueeze(-1))

    def total(self, features: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        values = self(features)
        if weights is None:
            weights = torch.ones(values.shape[-1], device=values.device)
        if weights.shape != (values.shape[-1],) or torch.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("weights must be non-negative and match the component count")
        return (values * (weights / weights.sum())).sum(dim=-1)


def fit_pairwise_reward_model(
    model: RewardModel | MultiRewardModel,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weights: torch.Tensor | None = None,
) -> RewardModelFit:
    """Fit a reward model so chosen examples score above rejected examples."""

    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected shapes must match")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        chosen_score = model.total(chosen, weights) if isinstance(model, MultiRewardModel) else model(chosen)
        rejected_score = model.total(rejected, weights) if isinstance(model, MultiRewardModel) else model(rejected)
        loss = -torch.nn.functional.logsigmoid(chosen_score - rejected_score).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return RewardModelFit(tuple(losses))
