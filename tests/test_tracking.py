from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from singalign.tracking import (
    RunMetadata,
    flatten_parameters,
    git_revision,
    tracked_run,
)


class TrackingTests(unittest.TestCase):
    def test_metadata_rejects_invalid_run_kind(self) -> None:
        with self.assertRaises(ValueError):
            RunMetadata("exp", "run", "invalid", "pjs", "1.1", "split", 1)  # type: ignore[arg-type]

    def test_flatten_parameters(self) -> None:
        flattened = flatten_parameters(
            {"model": {"layers": 2, "dropout": 0.1}, "labels": ["a", "b"]}
        )
        self.assertEqual(flattened["model.layers"], 2)
        self.assertEqual(flattened["model.dropout"], 0.1)
        self.assertEqual(flattened["labels"], '["a", "b"]')

    def test_git_revision_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(git_revision(Path(directory)), ("unknown", False))

    def test_git_revision_uses_container_build_metadata(self) -> None:
        environment = {
            "SINGALIGN_GIT_REVISION": "abc123",
            "SINGALIGN_GIT_DIRTY": "true",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(git_revision(), ("abc123", True))

    @patch("singalign.tracking.git_revision", return_value=("abc123", False))
    @patch("singalign.tracking._load_mlflow")
    def test_tracked_run_logs_standard_metadata(
        self, load_mlflow: MagicMock, _: MagicMock
    ) -> None:
        mlflow = load_mlflow.return_value
        context = MagicMock()
        context.__enter__.return_value = MagicMock(info=MagicMock(run_id="run-id"))
        mlflow.start_run.return_value = context
        metadata = RunMetadata(
            "experiment",
            "pilot",
            "exploratory",
            "pjs",
            "1.1",
            "split-sha",
            2026,
        )

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}):
            with tracked_run(metadata, {"model": {"layers": 2}}) as run:
                self.assertEqual(run.info.run_id, "run-id")

        mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")
        mlflow.set_experiment.assert_called_once_with("experiment")
        tags = mlflow.start_run.call_args.kwargs["tags"]
        self.assertEqual(tags["research.run_kind"], "exploratory")
        self.assertEqual(tags["data.split_fingerprint"], "split-sha")
        self.assertEqual(tags["code.git_revision"], "abc123")
        mlflow.log_params.assert_called_once_with({"seed": 2026, "model.layers": 2})
        mlflow.log_metric.assert_called_once()


if __name__ == "__main__":
    unittest.main()
