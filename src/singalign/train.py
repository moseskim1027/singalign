"""Reproducible training entry point for the reconstruction baseline."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from singalign.datasets import PJSMelDataset
from singalign.models import MelAutoencoder
from singalign.tracking import RunMetadata, log_json_artifact, tracked_run


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate a training configuration."""

    config = yaml.safe_load(path.read_text())
    required = {"experiment", "audio", "model", "training"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain {sorted(required)}")
    return resolve_training_config(config)


def resolve_training_config(
    config: dict[str, Any], segment_seconds: float | None = None
) -> dict[str, Any]:
    """Return a validated copy containing the resolved segment duration."""

    resolved = copy.deepcopy(config)
    if segment_seconds is not None:
        resolved["audio"]["segment_seconds"] = segment_seconds
    duration = resolved["audio"]["segment_seconds"]
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 < duration <= 30
    ):
        raise ValueError("segment_seconds must be between 0 and 30")
    resolved["audio"]["segment_seconds"] = float(duration)
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto, MPS, CUDA, or CPU with explicit availability checks."""

    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported device: {requested}")
    return torch.device(requested)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: Adam | None = None,
) -> float:
    """Run one train or validation epoch and return mean MSE."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = nn.functional.mse_loss(model(inputs), targets)
        if training:
            loss.backward()
            optimizer.step()
        count = inputs.shape[0]
        total_loss += loss.item() * count
        total_items += count
    if not total_items:
        raise ValueError("data loader is empty")
    return total_loss / total_items


def fit(
    config: dict[str, Any],
    train_dataset: PJSMelDataset,
    validation_dataset: PJSMelDataset,
    epochs_override: int | None = None,
) -> tuple[MelAutoencoder, list[dict[str, float]]]:
    """Train the baseline and select its checkpoint using validation loss."""

    training = config["training"]
    seed_everything(int(training["seed"]))
    device = resolve_device(str(training["device"]))
    batch_size = int(training["batch_size"])
    generator = torch.Generator().manual_seed(int(training["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    model = MelAutoencoder(int(config["model"]["latent_channels"])).to(device)
    optimizer = Adam(model.parameters(), lr=float(training["learning_rate"]))
    checkpoint_dir = Path(training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs = epochs_override or int(training["epochs"])
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    import mlflow

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, device)
        metrics = {
            "epoch": float(epoch),
            "train.loss": train_loss,
            "validation.loss": validation_loss,
        }
        history.append(metrics)
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if key != "epoch"},
            step=epoch,
        )
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "config": config,
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(state, checkpoint_dir / "best.pt")
    mlflow.log_metric("validation.best_loss", best_loss)
    mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")
    return model, history


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="singalign-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--segment-seconds", type=float)
    parser.add_argument("--max-train-items", type=int)
    parser.add_argument("--max-validation-items", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = resolve_training_config(load_config(args.config), args.segment_seconds)
    training = config["training"]
    seed = int(training["seed"])
    train_dataset = PJSMelDataset(
        args.index,
        args.splits,
        "train",
        config["audio"],
        seed,
        args.max_train_items,
    )
    validation_dataset = PJSMelDataset(
        args.index,
        args.splits,
        "validation",
        config["audio"],
        seed,
        args.max_validation_items,
    )
    metadata = RunMetadata(
        experiment_name=str(config["experiment"]["name"]),
        run_name=str(config["experiment"]["run_name"]),
        run_kind=str(config["experiment"]["run_kind"]),  # type: ignore[arg-type]
        dataset="pjs",
        dataset_version="1.1",
        split_fingerprint=train_dataset.fingerprint,
        seed=seed,
    )
    with tracked_run(metadata, config):
        _, history = fit(config, train_dataset, validation_dataset, args.epochs)
        log_json_artifact({"epochs": history}, "training-history.json")
    print(json.dumps(history[-1], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
