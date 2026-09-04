"""Exploratory KTO-style training on synthetic mel preference pairs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mlflow
import torch
import yaml
from torch.utils.data import DataLoader

from singalign.align import energy_score
from singalign.datasets import PJSMelDataset
from singalign.evaluate import load_checkpoint
from singalign.preference_objectives import kto_proxy_loss
from singalign.preferences import PreferencePairDataset, preference_parameters
from singalign.tracking import RunMetadata, tracked_run
from singalign.train import resolve_device, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-kto-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    preference_parameters(config["preferences"] if "preferences" in config else {"chosen_severity": 0.03, "rejected_severity": 0.12})
    settings, objective = config["training"], config["objective"]
    seed_everything(int(config["data"]["seed"]))
    device = resolve_device(str(settings["device"]))
    policy, train_config = load_checkpoint(args.checkpoint, device)
    reference = copy.deepcopy(policy).eval()
    dataset = PJSMelDataset(args.index, args.splits, "train", train_config["audio"], int(config["data"]["seed"]))
    pairs = PreferencePairDataset(dataset, int(config["data"]["seed"]), 0.03, 0.12)
    loader = DataLoader(pairs, batch_size=int(settings["batch_size"]), shuffle=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(settings["learning_rate"]))
    metadata = RunMetadata(config["experiment"]["name"], config["experiment"]["run_name"], config["experiment"]["run_kind"], "pjs", "1.1", dataset.fingerprint, int(config["data"]["seed"]))
    with tracked_run(metadata, config):
        for epoch in range(1, int(settings["epochs"]) + 1):
            total = 0.0
            for inputs, chosen, rejected in loader:
                inputs, chosen, rejected = inputs.to(device), chosen.to(device), rejected.to(device)
                with torch.no_grad():
                    reference_output = reference(inputs)
                policy_output = policy(inputs)
                chosen_score = energy_score(policy_output, chosen, float(objective["temperature"]))
                rejected_score = energy_score(policy_output, rejected, float(objective["temperature"]))
                reference_chosen = energy_score(reference_output, chosen, float(objective["temperature"]))
                reference_rejected = energy_score(reference_output, rejected, float(objective["temperature"]))
                scores = torch.cat((chosen_score - reference_chosen, rejected_score - reference_rejected))
                labels = torch.cat((torch.ones_like(chosen_score), torch.zeros_like(rejected_score))).bool()
                loss, advantage = kto_proxy_loss(scores, torch.zeros_like(scores), labels, float(objective["beta"]), float(objective["kl_baseline"]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()
            mlflow.log_metric("train.loss", total / len(loader), step=epoch)
            mlflow.log_metric("train.advantage", advantage.item(), step=epoch)
        checkpoint = Path(settings["checkpoint_dir"]) / "last.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": policy.state_dict(), "config": train_config}, checkpoint)
        mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
    print(json.dumps({"epochs": int(settings["epochs"]), "checkpoint": str(checkpoint)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
