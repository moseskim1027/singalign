from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from singalign.multi_compare import _condition


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


if __name__ == "__main__":
    unittest.main()
