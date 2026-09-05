"""Objective evaluation for the Study 2 content-and-melody transfer control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from singalign.audio import load_audio
from singalign.tracking import RunMetadata, log_json_artifact, tracked_run


def frame_pitch(waveform: torch.Tensor, sample_rate: int, frame_size: int = 1024, hop: int = 256) -> torch.Tensor:
    """Estimate one dominant frequency per frame with deterministic autocorrelation."""
    if waveform.ndim != 1 or sample_rate < 1 or frame_size < 2 or hop < 1:
        raise ValueError("invalid waveform or pitch-analysis settings")
    if waveform.numel() < frame_size:
        waveform = torch.nn.functional.pad(waveform, (0, frame_size - waveform.numel()))
    window = torch.hann_window(frame_size, dtype=waveform.dtype)
    frames = waveform.unfold(0, frame_size, hop)
    pitches: list[float] = []
    min_lag = max(1, sample_rate // 1000)
    max_lag = min(frame_size - 1, sample_rate // 70)
    for frame in frames:
        signal = frame * window
        if float(signal.square().mean()) < 1e-6:
            pitches.append(0.0)
            continue
        correlation = torch.nn.functional.conv1d(
            signal.view(1, 1, -1), signal.flip(0).view(1, 1, -1), padding=frame_size - 1
        ).flatten()[frame_size - 1 :]
        lag = min_lag + int(torch.argmax(correlation[min_lag : max_lag + 1]))
        pitches.append(float(sample_rate / lag))
    return torch.tensor(pitches)


def evaluate_transfer(reference: torch.Tensor, transferred: torch.Tensor, sample_rate: int) -> dict[str, float]:
    """Compare source vocal content/melody with the transferred output."""
    if reference.numel() == 0 or transferred.numel() == 0:
        raise ValueError("reference and transferred audio must not be empty")
    length = min(reference.numel(), transferred.numel())
    source = reference[:length]
    output = transferred[:length]
    source_pitch = frame_pitch(source, sample_rate)
    output_pitch = frame_pitch(output, sample_rate)
    voiced = (source_pitch > 0) & (output_pitch > 0)
    pitch_difference_hz = float((source_pitch[voiced] - output_pitch[voiced]).abs().mean()) if voiced.any() else 0.0
    source_envelope = source.abs().unfold(0, min(1024, length), min(1024, length)).mean(1)
    output_envelope = output.abs().unfold(0, min(1024, length), min(1024, length)).mean(1)
    if source_envelope.numel() > 1 and float(source_envelope.std()) > 0 and float(output_envelope.std()) > 0:
        content_similarity = float(torch.corrcoef(torch.stack((source_envelope, output_envelope)))[0, 1].clamp(-1, 1))
    else:
        content_similarity = 0.0
    return {
        "pitch_difference_hz_mean": pitch_difference_hz,
        "pitch_voiced_frame_rate": float(voiced.float().mean()),
        "content_envelope_correlation": content_similarity,
        "duration_error_seconds": abs(reference.numel() - transferred.numel()) / sample_rate,
        "output_peak": float(transferred.abs().max()),
    }


def run_evaluation(args: argparse.Namespace) -> int:
    reference = load_audio(args.reference, args.sample_rate)
    transferred = load_audio(args.transferred, args.sample_rate)
    metrics = evaluate_transfer(reference, transferred, args.sample_rate)
    metadata = RunMetadata("singalign-study-2-evaluation", args.run_name, "exploratory", "pjs", args.dataset_version, args.split_fingerprint, args.seed)
    with tracked_run(metadata, {"parent_transfer_run_id": args.parent_run_id or "not-provided", "sample_rate": args.sample_rate}) as run:
        mlflow = __import__("mlflow")
        if args.parent_run_id:
            mlflow.set_tag("lineage.parent_transfer_run_id", args.parent_run_id)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        log_json_artifact(metrics, "study-2-evaluation.json")
        report = {"mlflow_run_id": run.info.run_id, "parent_transfer_run_id": args.parent_run_id, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-transfer-evaluate")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transferred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-run-id")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--dataset-version", default="1.1")
    parser.add_argument("--split-fingerprint", default="not-provided")
    parser.add_argument("--run-name", default="study-2-evaluation")
    parser.add_argument("--seed", type=int, default=2026)
    return run_evaluation(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
