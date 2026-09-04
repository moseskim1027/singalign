from __future__ import annotations

import unittest

import torch

from singalign.reward_models import MultiRewardModel, RewardModel, fit_pairwise_reward_model


class RewardModelTest(unittest.TestCase):
    def test_scalar_model_fits_simple_preference(self) -> None:
        torch.manual_seed(4)
        chosen = torch.ones(8, 2, 2)
        rejected = torch.zeros_like(chosen)
        model = RewardModel((2, 2), hidden_size=8)
        fit = fit_pairwise_reward_model(model, chosen, rejected, epochs=30)
        self.assertEqual(len(fit.losses), 30)
        self.assertGreater(float(model(chosen).detach().mean()), float(model(rejected).detach().mean()))

    def test_multidimensional_model_returns_named_components(self) -> None:
        model = MultiRewardModel((2, 2), ("fidelity", "smoothness"), hidden_size=4)
        values = model(torch.zeros(3, 2, 2))
        self.assertEqual(values.shape, (3, 2))
        self.assertEqual(model.components, ("fidelity", "smoothness"))

    def test_pairwise_fit_rejects_shape_mismatch(self) -> None:
        model = RewardModel((2, 2))
        with self.assertRaisesRegex(ValueError, "shapes"):
            fit_pairwise_reward_model(model, torch.zeros(2, 2, 2), torch.zeros(3, 2, 2))


if __name__ == "__main__":
    unittest.main()
