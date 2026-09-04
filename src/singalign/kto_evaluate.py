"""Held-out diagnostics for the exploratory KTO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from singalign.align import energy_score
from singalign.datasets import PJSMelDataset
from singalign.evaluate import load_checkpoint
from singalign.preferences import PreferencePairDataset
from singalign.train import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-kto-evaluate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = resolve_device(str(config["training"].get("device", "cpu")))
    policy, train_config = load_checkpoint(args.checkpoint, device)
    base = PJSMelDataset(args.index, args.splits, "test", train_config["audio"], int(config["data"]["seed"]), allow_test=True)
    pairs = PreferencePairDataset(base, int(config["data"]["seed"]), 0.03, 0.12)
    losses, correct = [], []
    policy.eval()
    with torch.no_grad():
        for inputs, chosen, rejected in DataLoader(pairs, batch_size=4):
            output = policy(inputs.to(device))
            chosen_score = energy_score(output, chosen.to(device), 0.1)
            rejected_score = energy_score(output, rejected.to(device), 0.1)
            margin = chosen_score - rejected_score
            losses.append(float((-torch.nn.functional.logsigmoid(margin)).mean().item()))
            correct.extend((margin > 0).tolist())
    print(json.dumps({"split": "test", "examples": len(base), "preference_loss_mean": sum(losses) / len(losses), "preference_accuracy": sum(correct) / len(correct), "split_fingerprint": base.fingerprint}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
