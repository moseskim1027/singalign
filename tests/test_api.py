from __future__ import annotations

import unittest

from fastapi import HTTPException

from singalign.api import TrainingRequest, start_training


class TrainingApiTest(unittest.TestCase):
    def test_rejects_unknown_experiment_before_docker_access(self) -> None:
        with self.assertRaisesRegex(HTTPException, "unsupported experiment"):
            start_training(TrainingRequest(experiment="unknown"))

    def test_rejects_out_of_range_epochs_before_docker_access(self) -> None:
        with self.assertRaisesRegex(HTTPException, "out of range"):
            start_training(TrainingRequest(experiment="baseline", parameters={"epochs": 0}))

    def test_command_mapping_is_allowlisted(self) -> None:
        from singalign.api import EXPERIMENTS
        self.assertEqual(EXPERIMENTS["kto"], "singalign-kto-train")


if __name__ == "__main__":
    unittest.main()
