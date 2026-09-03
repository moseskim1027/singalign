from __future__ import annotations

import unittest

import torch
from torch.utils.data import TensorDataset

from singalign.preferences import PreferencePairDataset, degrade


class PreferencesTest(unittest.TestCase):
    def test_degradations_are_deterministic(self) -> None:
        features = torch.ones(1, 20, 30)
        first = degrade(features, "noise", 0.1, torch.Generator().manual_seed(12))
        second = degrade(features, "noise", 0.1, torch.Generator().manual_seed(12))
        self.assertTrue(first.equal(second))

    def test_rejected_mask_is_more_severe(self) -> None:
        features = torch.ones(1, 20, 30)
        mild = degrade(features, "time_mask", 0.1, torch.Generator().manual_seed(1))
        strong = degrade(features, "time_mask", 0.4, torch.Generator().manual_seed(2))
        self.assertLess(torch.count_nonzero(strong), torch.count_nonzero(mild))

    def test_pair_dataset_validates_severity_order(self) -> None:
        features = torch.ones(2, 1, 8, 8)
        dataset = TensorDataset(features, features)
        with self.assertRaisesRegex(ValueError, "lower"):
            PreferencePairDataset(dataset, 1, 0.2, 0.1)

    def test_pair_dataset_returns_three_tensors(self) -> None:
        features = torch.ones(3, 1, 8, 8)
        dataset = PreferencePairDataset(TensorDataset(features, features), 3, 0.05, 0.2)
        inputs, chosen, rejected = dataset[0]
        self.assertEqual(inputs.shape, chosen.shape)
        self.assertEqual(chosen.shape, rejected.shape)
