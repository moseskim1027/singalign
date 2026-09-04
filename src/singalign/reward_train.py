"""Train and track exploratory learned reward models from preference tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import mlflow

from singalign.reward_models import MultiRewardModel, RewardModel, fit_pairwise_reward_model
from singalign.tracking import RunMetadata, tracked_run


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-reward-train")
    parser.add_argument("--chosen", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("scalar", "multidimensional"), default="scalar")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mlflow-experiment", default="singalign-learned-reward")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    chosen = torch.load(args.chosen, map_location="cpu", weights_only=True)
    rejected = torch.load(args.rejected, map_location="cpu", weights_only=True)
    if not isinstance(chosen, torch.Tensor) or not isinstance(rejected, torch.Tensor):
        raise ValueError("chosen and rejected files must contain tensors")
    model = (MultiRewardModel(tuple(chosen.shape[1:]), ("fidelity", "smoothness", "amplitude"), args.hidden_size)
             if args.kind == "multidimensional" else RewardModel(tuple(chosen.shape[1:]), args.hidden_size))
    metadata = RunMetadata(args.mlflow_experiment, "reward-model-training", "exploratory", "saved-preference-tensors", "unknown", "not-applicable", args.seed)
    with tracked_run(metadata, {"kind": args.kind, "epochs": args.epochs, "learning_rate": args.learning_rate, "hidden_size": args.hidden_size}):
        fit = fit_pairwise_reward_model(model, chosen, rejected, args.epochs, args.learning_rate)
        with torch.no_grad():
            chosen_score = model.total(chosen) if isinstance(model, MultiRewardModel) else model(chosen)
            rejected_score = model.total(rejected) if isinstance(model, MultiRewardModel) else model(rejected)
        mlflow.log_metric("reward.final_loss", fit.losses[-1])
        mlflow.log_metric("reward.pairwise_accuracy", float((chosen_score > rejected_score).float().mean()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "feature_shape": model.feature_shape if isinstance(model, RewardModel) else model.model.feature_shape, "kind": args.kind}, args.output)
        mlflow.log_artifact(str(args.output), artifact_path="checkpoints")
    print(f"Wrote reward model checkpoint to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
