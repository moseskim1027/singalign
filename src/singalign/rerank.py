"""Proxy rewards and deterministic candidate reranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from singalign.candidates import Candidate


@dataclass(frozen=True)
class ScoredCandidate:
    """Candidate annotated with transparent diagnostic reward components."""

    candidate: Candidate
    reconstruction_error: float
    proxy_reward: float
    rank: int = 0

    def provenance(self) -> dict[str, str | int | float]:
        result = self.candidate.provenance()
        result.update({"reconstruction_error": self.reconstruction_error, "proxy_reward": self.proxy_reward, "rank": self.rank})
        return result


def rerank_candidates(reference: torch.Tensor, candidates: list[Candidate]) -> list[ScoredCandidate]:
    """Rank candidates by lower normalized MSE, breaking ties by input order."""
    if not candidates:
        raise ValueError("candidates must not be empty")
    denominator = reference.square().mean().clamp_min(1e-8)
    scored = []
    for candidate in candidates:
        error = torch.mean((candidate.features - reference) ** 2) / denominator
        value = float(error.item())
        scored.append(ScoredCandidate(candidate, value, -value))
    scored.sort(key=lambda item: (item.reconstruction_error, item.candidate.seed))
    return [ScoredCandidate(item.candidate, item.reconstruction_error, item.proxy_reward, rank)
            for rank, item in enumerate(scored, start=1)]
