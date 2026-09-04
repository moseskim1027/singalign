from __future__ import annotations

import unittest
from pathlib import Path

from singalign.conditions import ConditionSpec, validate_conditions


class ConditionsTest(unittest.TestCase):
    def test_preserves_declared_order(self) -> None:
        values = [ConditionSpec("baseline", Path("a"), "supervised"), ConditionSpec("kto", Path("b"), "kto")]
        self.assertEqual(validate_conditions(values), values)

    def test_rejects_duplicate_names(self) -> None:
        values = [ConditionSpec("same", Path("a"), "one"), ConditionSpec("same", Path("b"), "two")]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_conditions(values)


if __name__ == "__main__":
    unittest.main()
