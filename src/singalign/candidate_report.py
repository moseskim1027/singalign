"""Serialize candidate/reranking results for inspection and later MLflow logging."""

from __future__ import annotations

import json
import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-candidates")
    parser.add_argument("--input", type=Path, required=True, help="torch-saved mel tensor")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    reference = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(reference, torch.Tensor):
        raise ValueError("input must contain a torch Tensor")
    print(json.dumps(write_candidate_report(reference, args.output, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
