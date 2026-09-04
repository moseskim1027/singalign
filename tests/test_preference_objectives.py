from __future__ import annotations

import unittest

import torch

from singalign.preference_objectives import kto_proxy_loss, pair_to_kto_batch


class PreferenceObjectiveTest(unittest.TestCase):
    def test_kto_prefers_positive_desirable_advantage(self) -> None:
        loss, advantage = kto_proxy_loss(
            torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor([True, False])
        )
        self.assertGreater(float(advantage), 0.0)
        self.assertLess(float(loss), 0.7)

    def test_kto_validates_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "matching"):
            kto_proxy_loss(torch.zeros(2), torch.zeros(1), torch.zeros(2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "positive"):
            kto_proxy_loss(torch.zeros(1), torch.zeros(1), torch.ones(1, dtype=torch.bool), beta=0)

    def test_pair_adapter_marks_chosen_examples_desirable(self) -> None:
        scores, labels = pair_to_kto_batch(torch.tensor([1.0]), torch.tensor([-1.0]))
        self.assertEqual(scores.tolist(), [1.0, -1.0])
        self.assertEqual(labels.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
