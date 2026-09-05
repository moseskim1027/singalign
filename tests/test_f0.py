import unittest

import torch

from singalign.f0 import extract_f0


class F0Test(unittest.TestCase):
    def test_tracks_deterministic_tone(self) -> None:
        rate = 16_000
        time = torch.arange(rate, dtype=torch.float32) / rate
        frames = extract_f0(torch.sin(2 * torch.pi * 220 * time), rate)
        voiced = [frame.f0_hz for frame in frames if frame.voiced]
        self.assertGreater(len(voiced), 50)
        self.assertAlmostEqual(sum(voiced) / len(voiced), 220, delta=8)

    def test_silence_is_not_interpolated(self) -> None:
        frames = extract_f0(torch.zeros(16_000), 16_000)
        self.assertTrue(all(not frame.voiced and frame.f0_hz is None for frame in frames))
