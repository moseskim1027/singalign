"""Paired comparison and listening artifacts for baseline and aligned models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from scipy.io import wavfile

from singalign.audio import (
    crop_or_pad,
    invert_log_mel_spectrogram,
    load_audio,
    raw_log_mel_spectrogram,
)
from singalign.datasets import read_index
from singalign.evaluate import file_sha256, load_checkpoint
from singalign.metrics import paired_summary, reconstruction_metrics
from singalign.tracking import RunMetadata, tracked_run
from singalign.train import resolve_device, seed_everything


def load_comparison_config(path: Path) -> dict[str, Any]:
    """Load and validate the comparison configuration."""

    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not {"experiment", "comparison"}.issubset(
        config
    ):
        raise ValueError("configuration must contain experiment and comparison")
    split = config["comparison"]["split"]
    if split not in {"validation", "test"}:
        raise ValueError("comparison split must be validation or test")
    return config


def _prepare(record: dict[str, Any], audio: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    sample_rate = int(audio["sample_rate"])
    length = round(sample_rate * float(audio["segment_seconds"]))
    waveform = crop_or_pad(load_audio(Path(record["song_audio"]), sample_rate), length)
    raw = raw_log_mel_spectrogram(
        waveform,
        sample_rate,
        int(audio["n_fft"]),
        int(audio["hop_length"]),
        int(audio["mel_bins"]),
    )
    mean, standard_deviation = raw.mean(), raw.std().clamp_min(1e-6)
    normalized = (raw - mean) / standard_deviation
    return waveform, normalized.unsqueeze(0).unsqueeze(0), mean, standard_deviation


def _write_wave(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = waveform.detach().cpu().clamp(-1.0, 1.0).numpy()
    pcm = np.round(samples * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def compare_checkpoints(
    baseline_path: Path,
    aligned_path: Path,
    index_path: Path,
    splits_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Compare two checkpoints on identical validation or test examples."""

    settings = config["comparison"]
    seed = int(settings["seed"])
    seed_everything(seed)
    device = resolve_device(str(settings["device"]))
    baseline, baseline_config = load_checkpoint(baseline_path, device)
    aligned, aligned_config = load_checkpoint(aligned_path, device)
    if baseline_config["audio"] != aligned_config["audio"]:
        raise ValueError("checkpoint audio configurations do not match")
    audio = baseline_config["audio"]
    split_data = json.loads(splits_path.read_text())
    split_name = str(settings["split"])
    item_ids = list(split_data[split_name])
    if max_items is not None:
        item_ids = item_ids[:max_items]
    if not item_ids:
        raise ValueError("comparison split is empty")
    records = read_index(index_path)
    missing = [item_id for item_id in item_ids if item_id not in records]
    if missing:
        raise ValueError(f"split IDs missing from index: {missing[:3]}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    manifest_examples: list[dict[str, Any]] = []
    audio_limit = min(int(settings["audio_examples"]), len(item_ids))
    sample_rate = int(audio["sample_rate"])
    segment_length = round(sample_rate * float(audio["segment_seconds"]))

    with torch.inference_mode():
        for position, item_id in enumerate(item_ids):
            waveform, features, mean, standard_deviation = _prepare(
                records[item_id], audio
            )
            features = features.to(device)
            _synchronize(device)
            started = time.perf_counter()
            baseline_output = baseline(features)
            _synchronize(device)
            baseline_latency = time.perf_counter() - started
            _synchronize(device)
            started = time.perf_counter()
            aligned_output = aligned(features)
            _synchronize(device)
            aligned_latency = time.perf_counter() - started
            baseline_metrics = reconstruction_metrics(baseline_output, features)
            aligned_metrics = reconstruction_metrics(aligned_output, features)
            rows.append(
                {
                    "id": item_id,
                    "baseline": baseline_metrics,
                    "aligned": aligned_metrics,
                    "delta": {
                        name: aligned_metrics[name] - baseline_metrics[name]
                        for name in baseline_metrics
                    },
                    "latency_seconds": {
                        "baseline": baseline_latency,
                        "aligned": aligned_latency,
                    },
                }
            )
            if position < audio_limit:
                relative = Path("audio") / item_id
                _write_wave(
                    output_dir / relative / "reference.wav", waveform, sample_rate
                )
                inversion = {
                    "sample_rate": sample_rate,
                    "n_fft": int(audio["n_fft"]),
                    "hop_length": int(audio["hop_length"]),
                    "mel_bins": int(audio["mel_bins"]),
                    "length": segment_length,
                    "iterations": int(settings["griffin_lim_iterations"]),
                }
                for label, prediction in (
                    ("baseline", baseline_output),
                    ("aligned", aligned_output),
                ):
                    reconstructed = invert_log_mel_spectrogram(
                        prediction.squeeze().cpu(),
                        mean,
                        standard_deviation,
                        **inversion,
                        seed=seed + position,
                    )
                    _write_wave(
                        output_dir / relative / f"{label}.wav",
                        reconstructed,
                        sample_rate,
                    )
                manifest_examples.append(
                    {
                        "id": item_id,
                        "reference": str(relative / "reference.wav"),
                        "baseline": str(relative / "baseline.wav"),
                        "aligned": str(relative / "aligned.wav"),
                        "disclosure": (
                            "Approximate synthetic Griffin-Lim reconstruction"
                        ),
                    }
                )

    metric_names = list(rows[0]["baseline"])
    comparisons = {
        name: paired_summary(
            [row["baseline"][name] for row in rows],
            [row["aligned"][name] for row in rows],
            seed,
            int(settings["bootstrap_samples"]),
            float(settings["confidence_level"]),
        )
        for name in metric_names
    }
    summary = {
        "split": split_name,
        "split_fingerprint": split_data["fingerprint_sha256"],
        "examples": len(rows),
        "baseline_checkpoint_sha256": file_sha256(baseline_path),
        "aligned_checkpoint_sha256": file_sha256(aligned_path),
        "metrics": comparisons,
        "metric_direction": "lower is better; delta is aligned minus baseline",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "per-example.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    manifest = {
        "title": "SingAlign baseline versus proxy-aligned comparison",
        "split": split_name,
        "examples": manifest_examples,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="singalign-compare")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--aligned-checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--confirm-test-use", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_comparison_config(args.config)
    settings = config["comparison"]
    if settings["split"] == "test" and not args.confirm_test_use:
        raise SystemExit("test comparison requires --confirm-test-use")
    split_data = json.loads(args.splits.read_text())
    metadata = RunMetadata(
        experiment_name=str(config["experiment"]["name"]),
        run_name=str(config["experiment"]["run_name"]),
        run_kind=str(config["experiment"]["run_kind"]),  # type: ignore[arg-type]
        dataset="pjs",
        dataset_version="1.1",
        split_fingerprint=str(split_data["fingerprint_sha256"]),
        seed=int(settings["seed"]),
    )
    parameters = {
        **config,
        "baseline_sha256": file_sha256(args.baseline_checkpoint),
        "aligned_sha256": file_sha256(args.aligned_checkpoint),
    }
    with tracked_run(metadata, parameters) as run:
        output_dir = Path(settings["output_dir"]) / run.info.run_id
        summary = compare_checkpoints(
            args.baseline_checkpoint,
            args.aligned_checkpoint,
            args.index,
            args.splits,
            config,
            output_dir,
            args.max_items,
        )
        for name, result in summary["metrics"].items():
            mlflow.log_metric(f"comparison.{name}.delta", result["delta"]["mean"])
        mlflow.log_artifacts(str(output_dir), artifact_path="comparison")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
