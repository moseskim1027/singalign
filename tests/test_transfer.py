import unittest

import torch

from singalign.transfer import mix_vocal_and_instrument, tempo_align


class TransferTest(unittest.TestCase):
    def test_tempo_alignment_changes_duration(self) -> None:
        waveform = torch.ones(160)
        self.assertEqual(tempo_align(waveform, 2.0).numel(), 80)

    def test_mix_reports_and_limits_clipping(self) -> None:
        mixed, diagnostics = mix_vocal_and_instrument(torch.ones(4), torch.ones(2))
        self.assertEqual(mixed.tolist(), [1.0] * 4)
        self.assertEqual(diagnostics["clipped_samples"], 4)
