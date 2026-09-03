from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from singalign.audio import (
    crop_or_pad,
    invert_log_mel_spectrogram,
    load_audio,
    log_mel_spectrogram,
    raw_log_mel_spectrogram,
)


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

    def test_log_mel_inversion_returns_requested_length(self) -> None:
        waveform = torch.sin(torch.linspace(0, 30, 2048))
        raw = raw_log_mel_spectrogram(waveform, 8000, 128, 40, 16)
        mean, deviation = raw.mean(), raw.std()
        normalized = (raw - mean) / deviation
        reconstructed = invert_log_mel_spectrogram(
            normalized, mean, deviation, 8000, 128, 40, 16, 2048, 2, 7
        )
        self.assertEqual(reconstructed.shape, waveform.shape)
        self.assertTrue(torch.isfinite(reconstructed).all())
