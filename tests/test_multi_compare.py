from __future__ import annotations

import tempfile
import json
from unittest.mock import patch
import unittest
from pathlib import Path

import torch

from singalign.multi_compare import _condition
from singalign.condition_analysis import main as condition_analysis_main


class MultiCompareTest(unittest.TestCase):
    def test_parses_condition_spec(self) -> None:
        spec, path = _condition("kto=checkpoints/kto.pt:kto")
        self.assertEqual(spec.name, "kto")
        self.assertEqual(spec.method, "kto")
        self.assertEqual(path, Path("checkpoints/kto.pt"))

    def test_saved_tensors_are_available_for_cli_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mel.pt"
            tensor = torch.zeros(1, 2, 2)
            torch.save(tensor, path)
            self.assertTrue(torch.load(path, weights_only=True).equal(tensor))

    def test_condition_analysis_cli_writes_aggregate_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.pt"
            baseline = root / "baseline.pt"
            aligned = root / "aligned.pt"
            torch.save(torch.ones(1, 2, 2), reference)
            torch.save(torch.zeros(1, 2, 2), baseline)
            torch.save(torch.ones(1, 2, 2), aligned)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "references": [str(reference)],
                "conditions": [
                    {"name": "baseline", "method": "supervised", "checkpoint": "baseline.pt", "outputs": [str(baseline)]},
                    {"name": "aligned", "method": "dpo", "checkpoint": "aligned.pt", "outputs": [str(aligned)]},
                ],
            }))
            output = root / "report.json"
            with patch("sys.argv", ["singalign-condition-analysis", "--manifest", str(manifest), "--output", str(output), "--samples", "10"]):
                self.assertEqual(condition_analysis_main(), 0)
            self.assertEqual(json.loads(output.read_text())["condition_count"], 2)


if __name__ == "__main__":
    unittest.main()
