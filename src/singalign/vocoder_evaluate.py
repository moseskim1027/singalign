"""Held-out diagnostic evaluation for the trainable mel vocoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from singalign.datasets import PJSVocoderDataset
from singalign.models import MelVocoder
from singalign.train import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-vocoder-evaluate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = resolve_device(str(config["training"].get("device", "cpu")))
    model = MelVocoder(int(config["audio"]["mel_bins"]), int(config["audio"]["hop_length"])).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = PJSVocoderDataset(args.index, args.splits, "test", config["audio"], int(config["training"]["seed"]))
    losses, peaks = [], []
    with torch.no_grad():
        for mel, target in dataset:
            prediction = model(mel.unsqueeze(0).to(device)).squeeze(0).cpu()
            target = torch.nn.functional.pad(target, (0, max(0, prediction.numel() - target.numel())))[:prediction.numel()]
            losses.append(torch.mean((prediction - target) ** 2).item())
            peaks.append(prediction.abs().max().item())
    result = {"split": "test", "split_fingerprint": dataset.fingerprint, "examples": len(dataset), "waveform_mse_mean": sum(losses) / len(losses), "generated_peak_mean": sum(peaks) / len(peaks)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
