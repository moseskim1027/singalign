from __future__ import annotations

import unittest
from copy import deepcopy

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from singalign.models import MelAutoencoder
from singalign.train import resolve_device, resolve_training_config, run_epoch


TRAINING_CONFIG = {
    "experiment": {},
    "audio": {"segment_seconds": 1.0},
    "model": {},
    "training": {},
}


class TrainTest(unittest.TestCase):
    def test_cpu_train_and_validation_epochs(self) -> None:
        inputs = torch.randn(4, 1, 16, 20)
        loader = DataLoader(TensorDataset(inputs, inputs), batch_size=2)
        model = MelAutoencoder(latent_channels=8)
        optimizer = Adam(model.parameters(), lr=0.001)
        train_loss = run_epoch(model, loader, torch.device("cpu"), optimizer)
        validation_loss = run_epoch(model, loader, torch.device("cpu"))
        self.assertGreaterEqual(train_loss, 0.0)
        self.assertGreaterEqual(validation_loss, 0.0)

    def test_auto_device_is_available(self) -> None:
        self.assertIn(resolve_device("auto").type, {"cpu", "mps", "cuda"})

    def test_unknown_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            resolve_device("tpu")

    def test_training_duration_override_is_resolved_without_mutation(self) -> None:
        original = deepcopy(TRAINING_CONFIG)
        resolved = resolve_training_config(original, 3)
        self.assertEqual(resolved["audio"]["segment_seconds"], 3.0)
        self.assertEqual(original["audio"]["segment_seconds"], 1.0)

    def test_training_duration_must_be_bounded(self) -> None:
        for duration in (0, 31, float("nan")):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "between 0 and 30"):
                    resolve_training_config(TRAINING_CONFIG, duration)
