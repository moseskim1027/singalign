from __future__ import annotations

import unittest

import torch

from singalign.candidates import generate_candidates


class CandidateTest(unittest.TestCase):
    def test_generation_is_deterministic_and_records_provenance(self) -> None:
        reference = torch.ones(1, 8, 12)
        first = generate_candidates(reference, seed=9)
        second = generate_candidates(reference, seed=9)
        self.assertEqual([item.method for item in first], ["identity", "noise", "time_mask", "frequency_mask"])
        self.assertTrue(all(left.features.equal(right.features) for left, right in zip(first, second, strict=True)))
        self.assertEqual(first[1].provenance()["seed"], 9)

    def test_generation_rejects_invalid_severity(self) -> None:
        with self.assertRaisesRegex(ValueError, "severities"):
            generate_candidates(torch.ones(8, 12), seed=1, severities=(1.1,))


if __name__ == "__main__":
    unittest.main()
