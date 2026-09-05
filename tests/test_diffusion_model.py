import unittest

import torch

from singalign.models import ConditionalMelDiffusion


class DiffusionModelTest(unittest.TestCase):
    def test_conditional_denoiser_preserves_mel_shape(self) -> None:
        model = ConditionalMelDiffusion(mel_bins=16, condition_channels=12, hidden_channels=24)
        output = model(torch.randn(2, 16, 32), torch.tensor([1, 10]), torch.randn(2, 12, 32))
        self.assertEqual(output.shape, (2, 16, 32))

    def test_rejects_unaligned_conditioning(self) -> None:
        model = ConditionalMelDiffusion(mel_bins=16, condition_channels=12)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 16, 8), torch.tensor([1]), torch.randn(1, 12, 7))
