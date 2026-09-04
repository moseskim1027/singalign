"""Serialize candidate/reranking results for inspection and later MLflow logging."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from singalign.candidates import generate_candidates
from singalign.rerank import rerank_candidates
from singalign.rewards import multidimensional_reward


def write_candidate_report(
    reference: torch.Tensor, output: Path, seed: int = 2026
) -> dict[str, object]:
    """Generate, score, and write one deterministic candidate report."""
    candidates = generate_candidates(reference, seed)
    ranked = rerank_candidates(reference, candidates)
    rows = []
    for item in ranked:
        reward = multidimensional_reward(reference, item.candidate.features)
        rows.append({**item.provenance(), "reward": reward.provenance()})
    report = {"seed": seed, "candidate_count": len(rows), "candidates": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
