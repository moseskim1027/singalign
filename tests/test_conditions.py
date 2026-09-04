from __future__ import annotations

import unittest
from pathlib import Path

import torch

from singalign.conditions import ConditionSpec, compare_condition_outputs, validate_conditions


class ConditionsTest(unittest.TestCase):
    def test_preserves_declared_order(self) -> None:
        values = [ConditionSpec("baseline", Path("a"), "supervised"), ConditionSpec("kto", Path("b"), "kto")]
        self.assertEqual(validate_conditions(values), values)

    def test_rejects_duplicate_names(self) -> None:
        values = [ConditionSpec("same", Path("a"), "one"), ConditionSpec("same", Path("b"), "two")]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_conditions(values)

    def test_compares_outputs_in_declared_order(self) -> None:
        conditions = [ConditionSpec("first", Path("a"), "one"), ConditionSpec("second", Path("b"), "two")]
        rows = compare_condition_outputs(torch.zeros(1, 2, 2), {"second": torch.ones(1, 2, 2), "first": torch.zeros(1, 2, 2)}, conditions)
        self.assertEqual([row["name"] for row in rows], ["first", "second"])
        self.assertEqual(rows[0]["metrics"]["log_mel_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
