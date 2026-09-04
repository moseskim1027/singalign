"""Preference objectives for the simulation sandbox."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def pair_to_kto_batch(
    chosen_score: torch.Tensor, rejected_score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert chosen/rejected scores into KTO scores and binary labels."""
    if chosen_score.shape != rejected_score.shape:
        raise ValueError("chosen and rejected scores must have matching shapes")
    scores = torch.cat((chosen_score, rejected_score), dim=0)
    labels = torch.cat((torch.ones_like(chosen_score), torch.zeros_like(rejected_score))).bool()
    return scores, labels


def kto_proxy_loss(
    policy_score: torch.Tensor,
    reference_score: torch.Tensor,
    desirable: torch.Tensor,
    beta: float = 0.1,
    kl_baseline: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a compact KTO-style binary preference loss.

    Scores are proxy log-probabilities/energies. ``desirable`` is a boolean
    tensor; the returned second value is the mean signed advantage.
    """
    if policy_score.shape != reference_score.shape or policy_score.shape != desirable.shape:
        raise ValueError("scores and desirable labels must have matching shapes")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    advantage = beta * (policy_score - reference_score - kl_baseline)
    signed = torch.where(desirable.bool(), advantage, -advantage)
    return -F.logsigmoid(signed).mean(), advantage.mean()
