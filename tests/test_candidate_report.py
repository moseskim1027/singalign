from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from singalign.candidate_report import write_candidate_report


class CandidateReportTest(unittest.TestCase):
    def test_report_is_reproducible_and_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report = write_candidate_report(torch.ones(1, 4, 8), path, seed=3)
            self.assertEqual(report["candidate_count"], 4)
            self.assertEqual(json.loads(path.read_text())["seed"], 3)
            self.assertEqual(report["candidates"][0]["rank"], 1)


if __name__ == "__main__":
    unittest.main()
