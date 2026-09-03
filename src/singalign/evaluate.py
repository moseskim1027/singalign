"""Held-out evaluation for the SingAlign reconstruction baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml

from singalign.audio import crop_or_pad, load_audio, log_mel_spectrogram
from singalign.datasets import read_index
from singalign.metrics import bootstrap_mean_interval, reconstruction_metrics
from singalign.models import MelAutoencoder
from singalign.tracking import RunMetadata, tracked_run
from singalign.train import resolve_device, seed_everything


@dataclass(frozen=True)
class EvaluationResult:
    """Paths and summary produced by one held-out evaluation."""

    summary: dict[str, Any]
    output_dir: Path


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluation_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate an evaluation configuration."""

    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not {"experiment", "evaluation"}.issubset(
        config
    ):
        raise ValueError("configuration must contain experiment and evaluation")
    return config


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[MelAutoencoder, dict[str, Any]]:
    """Restore a baseline checkpoint and its training configuration."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or "audio" not in config or "model" not in config:
        raise ValueError("checkpoint does not contain a valid training configuration")
    model = MelAutoencoder(int(config["model"]["latent_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _features(record: dict[str, Any], audio: dict[str, Any]) -> torch.Tensor:
    sample_rate = int(audio["sample_rate"])
    length = round(sample_rate * float(audio["segment_seconds"]))
    waveform = crop_or_pad(load_audio(Path(record["song_audio"]), sample_rate), length)
    return (
        log_mel_spectrogram(
            waveform,
            sample_rate,
            int(audio["n_fft"]),
            int(audio["hop_length"]),
            int(audio["mel_bins"]),
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    index_path: Path,
    splits_path: Path,
    evaluation_config: dict[str, Any],
    output_dir: Path,
    max_items: int | None = None,
) -> EvaluationResult:
    """Evaluate a selected checkpoint once on the immutable test partition."""

    settings = evaluation_config["evaluation"]
    seed = int(settings["seed"])
    seed_everything(seed)
    device = resolve_device(str(settings["device"]))
    model, training_config = load_checkpoint(checkpoint_path, device)
    records = read_index(index_path)
    split_data = json.loads(splits_path.read_text())
    test_ids = list(split_data["test"])
    if max_items is not None:
        test_ids = test_ids[:max_items]
    if not test_ids:
        raise ValueError("test split is empty")
    missing = [item_id for item_id in test_ids if item_id not in records]
    if missing:
        raise ValueError(f"test IDs missing from index: {missing[:3]}")

    per_example: list[dict[str, Any]] = []
    with torch.inference_mode():
        for item_id in test_ids:
            target = _features(records[item_id], training_config["audio"]).to(device)
            _synchronize(device)
            started = time.perf_counter()
            prediction = model(target)
            _synchronize(device)
            latency = time.perf_counter() - started
            per_example.append(
                {
                    "id": item_id,
                    **reconstruction_metrics(prediction, target),
                    "latency_seconds": latency,
                }
            )

    metric_names = ["log_mel_mse", "log_mel_mae", "spectral_convergence"]
    intervals = {
        name: asdict(
            bootstrap_mean_interval(
                [float(row[name]) for row in per_example],
                seed=seed,
                samples=int(settings["bootstrap_samples"]),
                confidence_level=float(settings["confidence_level"]),
            )
        )
        for name in metric_names
    }
    latencies = [float(row["latency_seconds"]) for row in per_example]
    summary: dict[str, Any] = {
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "split_fingerprint": str(split_data["fingerprint_sha256"]),
        "test_examples": len(per_example),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "metrics": intervals,
        "latency_seconds": {
            "mean": sum(latencies) / len(latencies),
            "median": float(torch.tensor(latencies).median().item()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "per-example.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_example)
    )
    metadata = {
        "metric_domain": "normalized log-mel spectrogram",
        "test_ids": test_ids,
        "evaluation_config": evaluation_config,
        "training_config": training_config,
    }
    (output_dir / "evaluation-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return EvaluationResult(summary, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="singalign-evaluate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_evaluation_config(args.config)
    settings = config["evaluation"]
    split_data = json.loads(args.splits.read_text())
    checkpoint_hash = file_sha256(args.checkpoint)
    metadata = RunMetadata(
        experiment_name=str(config["experiment"]["name"]),
        run_name=str(config["experiment"]["run_name"]),
        run_kind=str(config["experiment"]["run_kind"]),  # type: ignore[arg-type]
        dataset="pjs",
        dataset_version="1.1",
        split_fingerprint=str(split_data["fingerprint_sha256"]),
        seed=int(settings["seed"]),
    )
    parameters = {**config, "checkpoint_sha256": checkpoint_hash}
    with tracked_run(metadata, parameters) as run:
        destination = Path(settings["output_dir"]) / run.info.run_id
        result = evaluate_checkpoint(
            args.checkpoint,
            args.index,
            args.splits,
            config,
            destination,
            args.max_items,
        )
        for name, interval in result.summary["metrics"].items():
            mlflow.log_metric(f"test.{name}", interval["mean"])
        mlflow.log_metric(
            "test.latency_seconds", result.summary["latency_seconds"]["mean"]
        )
        mlflow.log_artifacts(str(result.output_dir), artifact_path="evaluation")
    print(json.dumps(result.summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
