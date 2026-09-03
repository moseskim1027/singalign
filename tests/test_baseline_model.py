from __future__ import annotations

import unittest

import torch

from singalign.models import MelAutoencoder


class BaselineModelTest(unittest.TestCase):
    def test_model_preserves_input_shape_and_backpropagates(self) -> None:
        model = MelAutoencoder(latent_channels=8)
        inputs = torch.randn(2, 1, 31, 47)
        outputs = model(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        outputs.square().mean().backward()
        has_gradient = any(
            parameter.grad is not None for parameter in model.parameters()
        )
        self.assertTrue(has_gradient)

    def test_latent_channels_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            MelAutoencoder(latent_channels=0)
