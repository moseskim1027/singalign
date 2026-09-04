from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from singalign.matrix_status import main


class MatrixStatusTest(unittest.TestCase):
    def test_missing_matrix_inputs_are_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "matrix.yaml"
            config.write_text(
                "name: test\nreference_root: refs\nconditions:\n"
                "  - name: baseline\n    method: supervised\n"
                "    checkpoint: baseline.pt\n    output_root: baseline\n"
            )
            with patch("sys.argv", ["singalign-matrix-status", "--config", str(config)]):
                self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
