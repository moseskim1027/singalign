from __future__ import annotations

import unittest

import torch

from singalign.models import MelVocoder


class VocoderTest(unittest.TestCase):
    def test_vocoder_returns_waveform_at_configured_hop_length(self) -> None:
        model = MelVocoder(mel_bins=80, hop_length=16)
        waveform = model(torch.randn(2, 80, 10))
        self.assertEqual(waveform.shape, (2, 160))
        self.assertLessEqual(float(waveform.detach().abs().max()), 1.0)

    def test_vocoder_rejects_non_sequence_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            MelVocoder()(torch.zeros(80, 10))


if __name__ == "__main__":
    unittest.main()
