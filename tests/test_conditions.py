from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from singalign.conditions import ConditionSpec, compare_condition_dataset, compare_condition_outputs, validate_conditions, write_condition_report


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

    def test_writes_condition_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions.json"
            report = write_condition_report(
                torch.zeros(1, 2, 2), {"first": torch.zeros(1, 2, 2)},
                [ConditionSpec("first", Path("a"), "one")], path,
            )
            self.assertEqual(report["condition_count"], 1)
            self.assertIn("conditions", path.read_text())

    def test_dataset_comparison_bootstraps_paired_effects(self) -> None:
        references = [torch.ones(1, 2, 2), torch.ones(1, 2, 2)]
        conditions = [
            ConditionSpec("baseline", Path("baseline.pt"), "supervised"),
            ConditionSpec("aligned", Path("aligned.pt"), "dpo"),
        ]
        report = compare_condition_dataset(
            references,
            {"baseline": [item * 0.9 for item in references], "aligned": references},
            conditions,
            samples=50,
        )
        self.assertEqual(report["anchor"], "baseline")
        self.assertEqual(report["example_count"], 2)
        self.assertLess(report["conditions"][1]["effect_vs_anchor"]["log_mel_mse"]["mean"], 0)
        self.assertIn("standardized_effect", report["conditions"][1]["effect_vs_anchor"]["log_mel_mse"])


if __name__ == "__main__":
    unittest.main()
