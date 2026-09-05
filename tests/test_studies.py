import unittest

from singalign.studies import validate_pair_manifest


class StudiesTest(unittest.TestCase):
    def test_accepts_declared_pair_manifest(self) -> None:
        validate_pair_manifest(
            {
                "schema_version": "study-2-pairs-v1",
                "dataset": "pjs",
                "dataset_version": "1.1",
                "pairs": [{"source_id": "a", "target_id": "b"}],
            }
        )

    def test_rejects_same_source_and_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_pair_manifest(
                {
                    "schema_version": "study-2-pairs-v1",
                    "dataset": "pjs",
                    "dataset_version": "1.1",
                    "pairs": [{"source_id": "a", "target_id": "a"}],
                }
            )
