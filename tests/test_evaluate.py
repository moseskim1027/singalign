from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from singalign.evaluate import evaluate_checkpoint, file_sha256
from singalign.models import MelAutoencoder


class EvaluateTest(unittest.TestCase):
    def test_evaluate_checkpoint_writes_reproducible_report_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index in range(2):
                item_id = f"pjs{index + 1:03d}"
                audio_path = root / f"{item_id}.wav"
                signal = np.sin(np.linspace(0, 30 + index, 5000))
                wavfile.write(audio_path, 8000, (signal * 16000).astype(np.int16))
                records.append({"id": item_id, "song_audio": str(audio_path)})
            index_path = root / "index.jsonl"
            index_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            splits_path = root / "splits.json"
            splits_path.write_text(
                json.dumps(
                    {
                        "train": [],
                        "validation": [],
                        "test": [record["id"] for record in records],
                        "fingerprint_sha256": "held-out-fixture",
                    }
                )
            )
            training_config = {
                "audio": {
                    "sample_rate": 8000,
                    "segment_seconds": 0.25,
                    "n_fft": 128,
                    "hop_length": 40,
                    "mel_bins": 16,
                },
                "model": {"latent_channels": 8},
            }
            model = MelAutoencoder(8)
            checkpoint_path = root / "best.pt"
            torch.save(
                {"config": training_config, "model_state_dict": model.state_dict()},
                checkpoint_path,
            )
            evaluation_config = {
                "experiment": {"name": "test"},
                "evaluation": {
                    "seed": 3,
                    "device": "cpu",
                    "bootstrap_samples": 100,
                    "confidence_level": 0.95,
                },
            }
            output_dir = root / "report"
            result = evaluate_checkpoint(
                checkpoint_path,
                index_path,
                splits_path,
                evaluation_config,
                output_dir,
            )

            self.assertEqual(result.summary["test_examples"], 2)
            self.assertEqual(result.summary["split_fingerprint"], "held-out-fixture")
            self.assertEqual(
                result.summary["checkpoint_sha256"], file_sha256(checkpoint_path)
            )
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertEqual(
                len((output_dir / "per-example.jsonl").read_text().splitlines()), 2
            )
            metadata = json.loads((output_dir / "evaluation-metadata.json").read_text())
            self.assertEqual(metadata["test_ids"], ["pjs001", "pjs002"])

    def test_file_sha256_returns_hex_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_bytes(b"content")
            self.assertEqual(len(file_sha256(path)), 64)
