from __future__ import annotations

import unittest

import torch

from singalign.candidates import Candidate
from singalign.rerank import rerank_candidates, rerank_with_learned_model
from singalign.reward_models import RewardModel


class RerankTest(unittest.TestCase):
    def test_lower_error_is_ranked_first(self) -> None:
        reference = torch.ones(1, 4, 4)
        candidates = [
            Candidate("bad", 2, 1.0, torch.zeros_like(reference)),
            Candidate("good", 1, 0.0, reference.clone()),
        ]
        ranked = rerank_candidates(reference, candidates)
        self.assertEqual(ranked[0].candidate.method, "good")
        self.assertEqual(ranked[0].rank, 1)
        self.assertEqual(ranked[0].proxy_reward, 0.0)

    def test_empty_candidates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            rerank_candidates(torch.ones(1, 2, 2), [])

    def test_learned_model_ranking_preserves_candidate_provenance(self) -> None:
        model = RewardModel((2, 2), hidden_size=4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[-1].bias.fill_(1.0)
        candidates = [
            Candidate("noise", 2, 0.2, torch.zeros(1, 2, 2)),
            Candidate("noise", 1, 0.1, torch.ones(1, 2, 2)),
        ]
        ranked = rerank_with_learned_model(model, candidates)
        self.assertEqual([item.candidate.seed for item in ranked], [1, 2])
        self.assertEqual([item.rank for item in ranked], [1, 2])


if __name__ == "__main__":
    unittest.main()
