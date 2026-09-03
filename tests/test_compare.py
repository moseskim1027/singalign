from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from singalign.compare import (
    compare_checkpoints,
    load_comparison_config,
    write_latest_pointer,
)
from singalign.models import MelAutoencoder


class CompareTest(unittest.TestCase):
    def test_comparison_writes_paired_report_and_audio_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "song.wav"
            signal = np.zeros(6000)
            signal[2000:6000] = np.sin(np.linspace(0, 20, 4000))
            wavfile.write(audio_path, 8000, (signal * 16000).astype(np.int16))
            index_path = root / "index.jsonl"
            index_path.write_text(
                json.dumps({"id": "pjs001", "song_audio": str(audio_path)}) + "\n"
            )
            splits_path = root / "splits.json"
            splits_path.write_text(
                json.dumps(
                    {
                        "validation": ["pjs001"],
                        "test": [],
                        "fingerprint_sha256": "paired-fixture",
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
            checkpoint = {
                "config": training_config,
                "model_state_dict": model.state_dict(),
            }
            baseline_path, aligned_path = root / "baseline.pt", root / "aligned.pt"
            torch.save(checkpoint, baseline_path)
            torch.save(checkpoint, aligned_path)
            config = {
                "comparison": {
                    "split": "validation",
                    "seed": 1,
                    "device": "cpu",
                    "bootstrap_samples": 20,
                    "confidence_level": 0.95,
                    "audio_examples": 1,
                    "audio_segment_seconds": 0.5,
                    "audio_selection_hop_seconds": 0.125,
                    "griffin_lim_iterations": 1,
                }
            }
            output_dir = root / "output"
            summary = compare_checkpoints(
                baseline_path,
                aligned_path,
                index_path,
                splits_path,
                config,
                output_dir,
            )
            self.assertEqual(summary["examples"], 1)
            self.assertEqual(summary["training_segment_seconds"], 0.25)
            self.assertEqual(summary["comparison_segment_seconds"], 0.5)
            self.assertEqual(
                summary["audio_selection_method"], "highest_rms_reference_window"
            )
            self.assertEqual(summary["metrics"]["log_mel_mse"]["ties"], 1)
            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(len(manifest["examples"]), 1)
            self.assertIn("trained on 0.25-second", manifest["duration_disclosure"])
            self.assertEqual(
                manifest["examples"][0]["selection_method"],
                "highest_rms_reference_window",
            )
            self.assertGreater(manifest["examples"][0]["offset_seconds"], 0)
            for filename in ("reference.wav", "baseline.wav", "aligned.wav"):
                audio_file = output_dir / "audio" / "pjs001" / filename
                self.assertTrue(audio_file.is_file())
                with wave.open(str(audio_file)) as stream:
                    self.assertEqual(stream.getframerate(), 8000)
                    self.assertEqual(stream.getsampwidth(), 2)
                    self.assertEqual(stream.getnframes(), 4000)

    def test_config_rejects_training_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.yaml"
            path.write_text("experiment: {}\ncomparison: {split: train}\n")
            with self.assertRaisesRegex(ValueError, "validation or test"):
                load_comparison_config(path)

    def test_latest_pointer_identifies_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_latest_pointer(root, "run-123")
            self.assertEqual(
                json.loads((root / "latest.json").read_text()), {"run_id": "run-123"}
            )

    def test_config_rejects_invalid_audio_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.yaml"
            path.write_text(
                "experiment: {}\n"
                "comparison: {split: validation, audio_segment_seconds: 0}\n"
            )
            with self.assertRaisesRegex(ValueError, "between 0 and 30"):
                load_comparison_config(path)
