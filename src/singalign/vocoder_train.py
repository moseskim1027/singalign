"""Exploratory training entry point for the trainable mel vocoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch
import yaml
from torch.utils.data import DataLoader

from singalign.datasets import PJSVocoderDataset
from singalign.models import MelVocoder
from singalign.tracking import RunMetadata, tracked_run
from singalign.train import resolve_device, seed_everything


def _match_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Crop or right-pad waveform targets to the decoder's sample length."""
    if waveform.shape[-1] >= length:
        return waveform[..., :length]
    return torch.nn.functional.pad(waveform, (0, length - waveform.shape[-1]))


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-vocoder-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    audio, training = config["audio"], config["training"]
    seed_everything(int(training["seed"]))
    device = resolve_device(str(training["device"]))
    train_data = PJSVocoderDataset(args.index, args.splits, "train", audio, int(training["seed"]))
    validation_data = PJSVocoderDataset(args.index, args.splits, "validation", audio, int(training["seed"]))
    model = MelVocoder(int(audio["mel_bins"]), int(audio["hop_length"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    metadata = RunMetadata(config["experiment"]["name"], config["experiment"]["run_name"], config["experiment"]["run_kind"], "pjs", "1.1", train_data.fingerprint, int(training["seed"]))
    with tracked_run(metadata, config):
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            total = 0.0
            train_loader = DataLoader(train_data, batch_size=int(training["batch_size"]), shuffle=True)
            for mel, target in train_loader:
                prediction = model(mel.to(device))
                loss = torch.nn.functional.mse_loss(
                    prediction, _match_length(target.to(device), prediction.shape[-1])
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()
            model.eval()
            with torch.no_grad():
                validation_total = 0.0
                validation_loader = DataLoader(validation_data, batch_size=int(training["batch_size"]))
                for mel, target in validation_loader:
                    prediction = model(mel.to(device))
                    validation_total += torch.nn.functional.mse_loss(
                        prediction, _match_length(target.to(device), prediction.shape[-1])
                    ).item()
                validation = validation_total / len(validation_loader)
            mlflow.log_metric("train.loss", total / len(train_loader), step=epoch)
            mlflow.log_metric("validation.loss", validation, step=epoch)
        checkpoint = Path(training["checkpoint_dir"]) / "last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
        mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
    print(json.dumps({"epochs": int(training["epochs"]), "checkpoint": str(checkpoint)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
