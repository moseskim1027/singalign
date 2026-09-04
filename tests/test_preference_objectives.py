from __future__ import annotations

import unittest

import torch

from singalign.preference_objectives import kto_proxy_loss


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


if __name__ == "__main__":
    unittest.main()
