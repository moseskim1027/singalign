"""Datasets for baseline SingAlign experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from singalign.audio import crop_or_pad, load_audio, log_mel_spectrogram

DevelopmentSplit = Literal["train", "validation"]


def read_index(path: Path) -> dict[str, dict[str, Any]]:
    """Read the PJS JSONL index keyed by record ID."""

    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {str(record["id"]): record for record in records}


class PJSMelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Deterministic short singing segments for reconstruction training."""

    def __init__(
        self,
        index_path: Path,
        splits_path: Path,
        split: DevelopmentSplit,
        audio_config: dict[str, Any],
        seed: int,
        max_items: int | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("training datasets may only use train or validation")
        records = read_index(index_path)
        split_data = json.loads(splits_path.read_text())
        self.fingerprint = str(split_data["fingerprint_sha256"])
        ids = list(split_data[split])
        if max_items is not None:
            ids = ids[:max_items]
        missing = [item_id for item_id in ids if item_id not in records]
        if missing:
            raise ValueError(f"split IDs missing from index: {missing[:3]}")
        self.records = [records[item_id] for item_id in ids]
        self.audio_config = audio_config
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        sample_rate = int(self.audio_config["sample_rate"])
        segment_seconds = float(self.audio_config["segment_seconds"])
        segment_length = round(sample_rate * segment_seconds)
        waveform = load_audio(Path(record["song_audio"]), sample_rate)
        available = max(waveform.numel() - segment_length, 0)
        token = f"{self.seed}:{record['id']}".encode()
        offset = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        offset = offset % (available + 1)
        segment = crop_or_pad(waveform, segment_length, offset)
        mel = log_mel_spectrogram(
            segment,
            sample_rate=sample_rate,
            n_fft=int(self.audio_config["n_fft"]),
            hop_length=int(self.audio_config["hop_length"]),
            mel_bins=int(self.audio_config["mel_bins"]),
        ).unsqueeze(0)
        return mel, mel
