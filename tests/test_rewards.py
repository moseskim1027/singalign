from __future__ import annotations

import unittest

import torch

from singalign.rewards import multidimensional_reward, scalar_reward


class RewardTest(unittest.TestCase):
    def test_identity_has_maximum_scalar_reward(self) -> None:
        reference = torch.ones(1, 4, 4)
        result = scalar_reward(reference, reference)
        self.assertEqual(result.total, 0.0)
        self.assertEqual(result.components["reconstruction"], 0.0)

    def test_multidimensional_reward_preserves_components(self) -> None:
        reference = torch.ones(1, 4, 4)
        result = multidimensional_reward(reference, torch.zeros_like(reference))
        self.assertEqual(set(result.components), {"reconstruction", "smoothness", "amplitude"})
        self.assertEqual(result.provenance()["weights"], {"reconstruction": 0.7, "smoothness": 0.2, "amplitude": 0.1})

    def test_reward_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes"):
            scalar_reward(torch.zeros(2, 2), torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
