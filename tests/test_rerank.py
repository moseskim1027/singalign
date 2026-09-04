from __future__ import annotations

import unittest

import torch

from singalign.candidates import Candidate
from singalign.rerank import rerank_candidates


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


if __name__ == "__main__":
    unittest.main()
