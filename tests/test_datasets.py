from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from singalign.datasets import PJSMelDataset


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        audio_path = root / "song.wav"
        wavfile.write(audio_path, 16000, np.zeros(20000, dtype=np.int16))
        self.index = root / "index.jsonl"
        self.index.write_text(
            json.dumps({"id": "pjs001", "song_audio": str(audio_path)}) + "\n"
        )
        self.splits = root / "splits.json"
        self.splits.write_text(
            json.dumps(
                {
                    "train": ["pjs001"],
                    "validation": ["pjs001"],
                    "test": ["pjs001"],
                    "fingerprint_sha256": "fixture-fingerprint",
                }
            )
        )
        self.audio_config = {
            "sample_rate": 16000,
            "segment_seconds": 0.25,
            "n_fft": 256,
            "hop_length": 80,
            "mel_bins": 32,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dataset_is_deterministic(self) -> None:
        dataset = PJSMelDataset(
            self.index, self.splits, "train", self.audio_config, seed=7
        )
        first, target = dataset[0]
        second, _ = dataset[0]
        self.assertEqual(dataset.fingerprint, "fixture-fingerprint")
        self.assertEqual(first.shape[0:2], (1, 32))
        self.assertTrue(first.equal(second))
        self.assertTrue(first.equal(target))

    def test_test_split_cannot_be_loaded_for_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "train or validation"):
            PJSMelDataset(
                self.index,
                self.splits,
                "test",  # type: ignore[arg-type]
                self.audio_config,
                seed=7,
            )

    def test_test_split_requires_explicit_evaluation_opt_in(self) -> None:
        dataset = PJSMelDataset(
            self.index, self.splits, "test", self.audio_config, seed=7, allow_test=True
        )
        self.assertEqual(len(dataset), 1)
