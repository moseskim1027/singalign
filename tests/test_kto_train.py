from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from singalign.kto_train import load_kto_config


class KTOTrainTest(unittest.TestCase):
    def test_loads_kto_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kto.yaml"
            path.write_text("experiment: {}\nobjective: {beta: 0.1, temperature: 0.1}\ndata: {}\ntraining: {}\n")
            self.assertIn("objective", load_kto_config(path))

    def test_rejects_nonpositive_beta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kto.yaml"
            path.write_text("experiment: {}\nobjective: {beta: 0, temperature: 0.1}\ndata: {}\ntraining: {}\n")
            with self.assertRaisesRegex(ValueError, "positive"):
                load_kto_config(path)


if __name__ == "__main__":
    unittest.main()
