from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from singalign.audio import crop_or_pad, load_audio, log_mel_spectrogram


class AudioTest(unittest.TestCase):
    def test_load_resample_and_extract_log_mel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            samples = (np.sin(np.linspace(0, 20, 8000)) * 16000).astype(np.int16)
            wavfile.write(path, 8000, samples)
            waveform = load_audio(path, 16000)
            segment = crop_or_pad(waveform, 16000)
            features = log_mel_spectrogram(segment, 16000, 512, 160, 80)

        self.assertEqual(segment.shape, (16000,))
        self.assertEqual(features.shape[0], 80)
        self.assertTrue(torch.isfinite(features).all())

    def test_crop_or_pad_rejects_invalid_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            crop_or_pad(torch.zeros(4), 0)
