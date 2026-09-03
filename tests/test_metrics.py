from __future__ import annotations

import unittest

import torch

from singalign.metrics import (
    bootstrap_mean_interval,
    paired_summary,
    reconstruction_metrics,
)


class MetricsTest(unittest.TestCase):
    def test_identical_spectrograms_have_zero_error(self) -> None:
        target = torch.ones(1, 1, 4, 5)
        metrics = reconstruction_metrics(target.clone(), target)
        self.assertEqual(metrics["log_mel_mse"], 0.0)
        self.assertEqual(metrics["log_mel_mae"], 0.0)
        self.assertEqual(metrics["spectral_convergence"], 0.0)

    def test_bootstrap_interval_is_deterministic(self) -> None:
        first = bootstrap_mean_interval([1.0, 2.0, 3.0], seed=11, samples=100)
        second = bootstrap_mean_interval([1.0, 2.0, 3.0], seed=11, samples=100)
        self.assertEqual(first, second)
        self.assertEqual(first.mean, 2.0)
        self.assertLessEqual(first.lower, first.mean)
        self.assertGreaterEqual(first.upper, first.mean)

    def test_bootstrap_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            bootstrap_mean_interval([], seed=1)

    def test_paired_summary_reports_improvements(self) -> None:
        summary = paired_summary([2.0, 3.0], [1.0, 2.0], 2, 100, 0.95)
        self.assertEqual(summary["wins"], 2)
        self.assertEqual(summary["losses"], 0)
        self.assertEqual(summary["delta"]["mean"], -1.0)  # type: ignore[index]
