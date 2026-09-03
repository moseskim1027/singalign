from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from singalign.align import (
    dpo_proxy_loss,
    energy_score,
    load_alignment_config,
    run_alignment_epoch,
)
from singalign.models import MelAutoencoder
from singalign.preferences import PreferencePairDataset


class AlignTest(unittest.TestCase):
    def test_energy_prefers_closer_candidate(self) -> None:
        output = torch.zeros(2, 1, 4, 4)
        close = torch.full_like(output, 0.1)
        far = torch.full_like(output, 0.5)
        close_score = energy_score(output, close, 0.1)
        far_score = energy_score(output, far, 0.1)
        self.assertTrue(torch.all(close_score > far_score))

    def test_identical_policy_and_reference_have_log_two_loss(self) -> None:
        output = torch.zeros(2, 1, 4, 4)
        chosen = torch.full_like(output, 0.1)
        rejected = torch.full_like(output, 0.5)
        loss, relative_margin, accuracy = dpo_proxy_loss(
            output, output, chosen, rejected, beta=0.1, temperature=0.1
        )
        self.assertAlmostEqual(loss.item(), 0.693147, places=5)
        self.assertEqual(relative_margin.item(), 0.0)
        self.assertEqual(accuracy.item(), 1.0)

    def test_alignment_epoch_reports_all_metrics(self) -> None:
        features = torch.randn(4, 1, 16, 20)
        pairs = PreferencePairDataset(TensorDataset(features, features), 4, 0.02, 0.1)
        loader = DataLoader(pairs, batch_size=2)
        policy = MelAutoencoder(8)
        reference = MelAutoencoder(8)
        reference.load_state_dict(policy.state_dict())
        optimizer = Adam(policy.parameters(), lr=0.001)
        metrics = run_alignment_epoch(
            policy,
            reference,
            loader,
            torch.device("cpu"),
            beta=0.1,
            temperature=0.1,
            anchor_weight=1.0,
            optimizer=optimizer,
        )
        self.assertEqual(
            set(metrics),
            {"loss", "dpo_loss", "anchor_loss", "margin", "accuracy"},
        )
        self.assertGreater(metrics["loss"], 0.0)

    def test_alignment_config_rejects_invalid_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.yaml"
            path.write_text(
                """experiment: {name: test}
preferences: {chosen_severity: 0.1, rejected_severity: 0.2}
alignment: {beta: 0.1, temperature: 0, anchor_weight: 1}
"""
            )
            with self.assertRaisesRegex(ValueError, "temperature"):
                load_alignment_config(path)
