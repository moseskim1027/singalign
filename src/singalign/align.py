"""DPO-style energy-based post-training for the reconstruction baseline."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from singalign.datasets import PJSMelDataset
from singalign.evaluate import file_sha256, load_checkpoint
from singalign.preferences import PreferencePairDataset, preference_parameters
from singalign.tracking import RunMetadata, log_json_artifact, tracked_run
from singalign.train import resolve_device, seed_everything


def load_alignment_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate an alignment configuration."""

    config = yaml.safe_load(path.read_text())
    required = {"experiment", "preferences", "alignment"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain {sorted(required)}")
    preference_parameters(config["preferences"])
    settings = config["alignment"]
    if float(settings["beta"]) <= 0.0:
        raise ValueError("beta must be positive")
    if float(settings["temperature"]) <= 0.0:
        raise ValueError("temperature must be positive")
    if float(settings["anchor_weight"]) < 0.0:
        raise ValueError("anchor_weight must be non-negative")
    return config


def energy_score(
    reconstruction: torch.Tensor, candidate: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Return a Gaussian-energy proxy for conditional log probability."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    dimensions = tuple(range(1, reconstruction.ndim))
    energy = torch.mean((reconstruction - candidate).square(), dim=dimensions)
    return -energy / temperature


def dpo_proxy_loss(
    policy_output: torch.Tensor,
    reference_output: torch.Tensor,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    beta: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO-style loss, policy margin, and preference accuracy."""

    policy_margin = energy_score(policy_output, chosen, temperature) - energy_score(
        policy_output, rejected, temperature
    )
    reference_margin = energy_score(
        reference_output, chosen, temperature
    ) - energy_score(reference_output, rejected, temperature)
    relative_margin = policy_margin - reference_margin
    loss = -F.logsigmoid(beta * relative_margin).mean()
    accuracy = (policy_margin > 0.0).float().mean()
    return loss, relative_margin.mean(), accuracy


def run_alignment_epoch(
    policy: nn.Module,
    reference: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    beta: float,
    temperature: float,
    anchor_weight: float,
    optimizer: Adam | None = None,
) -> dict[str, float]:
    """Run one post-training or validation epoch."""

    training = optimizer is not None
    policy.train(training)
    totals = {
        "loss": 0.0,
        "dpo_loss": 0.0,
        "anchor_loss": 0.0,
        "margin": 0.0,
        "accuracy": 0.0,
    }
    count = 0
    for inputs, chosen, rejected in loader:
        inputs = inputs.to(device)
        chosen = chosen.to(device)
        rejected = rejected.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            reference_output = reference(inputs)
        with torch.set_grad_enabled(training):
            policy_output = policy(inputs)
            dpo_loss, margin, accuracy = dpo_proxy_loss(
                policy_output,
                reference_output,
                chosen,
                rejected,
                beta,
                temperature,
            )
            anchor_loss = F.mse_loss(policy_output, inputs)
            loss = dpo_loss + anchor_weight * anchor_loss
        if training:
            loss.backward()
            optimizer.step()
        batch = inputs.shape[0]
        values = {
            "loss": loss.item(),
            "dpo_loss": dpo_loss.item(),
            "anchor_loss": anchor_loss.item(),
            "margin": margin.item(),
            "accuracy": accuracy.item(),
        }
        for name, value in values.items():
            totals[name] += value * batch
        count += batch
    if not count:
        raise ValueError("data loader is empty")
    return {name: value / count for name, value in totals.items()}


def align(
    config: dict[str, Any],
    baseline_checkpoint: Path,
    train_dataset: PJSMelDataset,
    validation_dataset: PJSMelDataset,
    epochs_override: int | None = None,
) -> list[dict[str, float]]:
    """Post-train a baseline relative to a frozen reference model."""

    settings = config["alignment"]
    seed = int(settings["seed"])
    seed_everything(seed)
    device = resolve_device(str(settings["device"]))
    policy, training_config = load_checkpoint(baseline_checkpoint, device)
    baseline_hash = file_sha256(baseline_checkpoint)
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    chosen_severity, rejected_severity = preference_parameters(config["preferences"])
    train_pairs = PreferencePairDataset(
        train_dataset, seed, chosen_severity, rejected_severity
    )
    validation_pairs = PreferencePairDataset(
        validation_dataset, seed + 1_000_000, chosen_severity, rejected_severity
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_pairs,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(settings["num_workers"]),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_pairs, batch_size=int(settings["batch_size"])
    )
    optimizer = Adam(policy.parameters(), lr=float(settings["learning_rate"]))
    checkpoint_dir = Path(settings["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs = epochs_override or int(settings["epochs"])
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_alignment_epoch(
            policy,
            reference,
            train_loader,
            device,
            float(settings["beta"]),
            float(settings["temperature"]),
            float(settings["anchor_weight"]),
            optimizer,
        )
        validation_metrics = run_alignment_epoch(
            policy,
            reference,
            validation_loader,
            device,
            float(settings["beta"]),
            float(settings["temperature"]),
            float(settings["anchor_weight"]),
        )
        metrics = {
            "epoch": float(epoch),
            **{f"train.{key}": value for key, value in train_metrics.items()},
            **{f"validation.{key}": value for key, value in validation_metrics.items()},
        }
        history.append(metrics)
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if key != "epoch"}, step=epoch
        )
        state = {
            "epoch": epoch,
            "model_state_dict": policy.state_dict(),
            "config": training_config,
            "alignment_config": config,
            "baseline_checkpoint_sha256": baseline_hash,
            "validation_loss": validation_metrics["loss"],
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            torch.save(state, checkpoint_dir / "best.pt")
    mlflow.log_metric("validation.best_loss", best_loss)
    mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")
    return history


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="singalign-align")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-items", type=int)
    parser.add_argument("--max-validation-items", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_alignment_config(args.config)
    settings = config["alignment"]
    seed = int(settings["seed"])
    baseline_hash = file_sha256(args.checkpoint)
    _, baseline_config = load_checkpoint(args.checkpoint, torch.device("cpu"))
    audio_config = baseline_config["audio"]
    train_dataset = PJSMelDataset(
        args.index,
        args.splits,
        "train",
        audio_config,
        seed,
        args.max_train_items,
    )
    validation_dataset = PJSMelDataset(
        args.index,
        args.splits,
        "validation",
        audio_config,
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
    with tracked_run(metadata, {**config, "baseline_sha256": baseline_hash}):
        history = align(
            config,
            args.checkpoint,
            train_dataset,
            validation_dataset,
            args.epochs,
        )
        log_json_artifact({"epochs": history}, "alignment-history.json")
    print(json.dumps(history[-1], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
