import unittest

import torch

from singalign.models import ConditionalMelDiffusion, ScoreConditionedMelDiffusion


class DiffusionModelTest(unittest.TestCase):
    def test_conditional_denoiser_preserves_mel_shape(self) -> None:
        model = ConditionalMelDiffusion(mel_bins=16, condition_channels=12, hidden_channels=24)
        output = model(torch.randn(2, 16, 32), torch.tensor([1, 10]), torch.randn(2, 12, 32))
        self.assertEqual(output.shape, (2, 16, 32))

    def test_rejects_unaligned_conditioning(self) -> None:
        model = ConditionalMelDiffusion(mel_bins=16, condition_channels=12)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 16, 8), torch.tensor([1]), torch.randn(1, 12, 7))

    def test_score_conditioned_scaffold_preserves_mel_shape(self) -> None:
        model = ScoreConditionedMelDiffusion(mel_bins=16, condition_channels=12, hidden_channels=24)
        output = model(
            torch.randn(2, 16, 32),
            torch.tensor([1, 10]),
            torch.randint(0, 20, (2, 32)),
            torch.randint(0, 80, (2, 32)),
            torch.rand(2, 32),
            torch.linspace(0, 1, 32).repeat(2, 1),
        )
        self.assertEqual(output.shape, (2, 16, 32))
