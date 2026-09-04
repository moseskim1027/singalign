"""Exploratory training entry point for the score-conditioned mel model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch
import yaml
from torch.utils.data import DataLoader

from singalign.datasets import PJSConditionedDataset
from singalign.models import ScoreConditionedMelModel
from singalign.tracking import RunMetadata, tracked_run
from singalign.train import resolve_device, seed_everything


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="singalign-conditioned-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = yaml.safe_load(args.config.read_text())
    audio, conditioning, training = config["audio"], config["conditioning"], config["training"]
    seed_everything(int(training["seed"]))
    device = resolve_device(str(training["device"]))
    train_data = PJSConditionedDataset(args.index, args.splits, "train", audio, conditioning, int(training["seed"]), args.max_items)
    model = ScoreConditionedMelModel(int(audio["mel_bins"]), int(conditioning["phoneme_vocab_size"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    metadata = RunMetadata(str(config["experiment"]["name"]), str(config["experiment"]["run_name"]), str(config["experiment"]["run_kind"]), "pjs", "1.1", train_data.fingerprint, int(training["seed"]))
    with tracked_run(metadata, config):
        model.train()
        loader = DataLoader(train_data, batch_size=int(training["batch_size"]), shuffle=True)
        for epoch in range(1, int(training["epochs"]) + 1):
            total = 0.0
            for pitch, phonemes, target in loader:
                prediction = model(pitch.to(device), phonemes.to(device))
                loss = torch.nn.functional.mse_loss(prediction, target.to(device).squeeze(1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()
            mlflow.log_metric("train.loss", total / len(loader), step=epoch)
        checkpoint = Path(training["checkpoint_dir"]) / "last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
        mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
    print(json.dumps({"epochs": int(training["epochs"]), "checkpoint": str(checkpoint)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
