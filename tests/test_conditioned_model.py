from __future__ import annotations

import unittest

import torch

from singalign.models import ScoreConditionedMelModel


class ConditionedModelTest(unittest.TestCase):
    def test_conditioned_model_returns_mel_frames(self) -> None:
        model = ScoreConditionedMelModel(mel_bins=80, phoneme_vocab_size=32)
        output = model(torch.tensor([[60, 60, 0, 62]]), torch.tensor([[3, 3, 4, 5]]))
        self.assertEqual(output.shape, (1, 80, 4))

    def test_conditioning_shapes_must_match(self) -> None:
        model = ScoreConditionedMelModel()
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
