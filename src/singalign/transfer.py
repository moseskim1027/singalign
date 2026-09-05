"""Deterministic audio controls for Study 2 content-and-melody transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from singalign.audio import load_audio
from singalign.tracking import RunMetadata, log_json_artifact, tracked_run


def render_note_events(
    notes: list[tuple[float, float, int | None]],
    bpm: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Render score note events to a deterministic additive-sine instrument."""

    if bpm <= 0 or sample_rate < 1 or not notes:
        raise ValueError("bpm and sample_rate must be positive and notes required")
    seconds_per_quarter = 60.0 / bpm
    duration = max(onset + length for onset, length, _ in notes) * seconds_per_quarter
    output = torch.zeros(max(1, round(duration * sample_rate)))
    for onset, length, midi in notes:
        if midi is None:
            continue
        start = round(onset * seconds_per_quarter * sample_rate)
        count = min(
            round(length * seconds_per_quarter * sample_rate),
            output.numel() - start,
        )
        if count <= 0:
            continue
        time = torch.arange(count, dtype=torch.float32) / sample_rate
        frequency = 440.0 * 2 ** ((midi - 69) / 12)
        envelope = torch.minimum(
            torch.ones(count), torch.arange(count) / max(1, round(sample_rate * 0.01))
        )
        output[start : start + count] += 0.25 * envelope * torch.sin(
            2 * torch.pi * frequency * time
        )
    return output.clamp(-1.0, 1.0)


def tempo_align(waveform: torch.Tensor, tempo_scale: float) -> torch.Tensor:
    """Change duration by a declared tempo scale using deterministic resampling."""

    if waveform.ndim != 1 or tempo_scale <= 0:
        raise ValueError("waveform must be mono and tempo_scale must be positive")
    denominator = 10_000
    numerator = max(1, round(tempo_scale * denominator))
    return torch.from_numpy(
        resample_poly(waveform.numpy(), denominator, numerator).astype("float32")
    )


def mix_vocal_and_instrument(
    vocal: torch.Tensor,
    instrumental: torch.Tensor,
    vocal_gain: float = 1.0,
    instrument_gain: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mix two mono stems with fixed gains and report clipping diagnostics."""

    if vocal_gain < 0 or instrument_gain < 0:
        raise ValueError("gains must be non-negative")
    length = max(vocal.numel(), instrumental.numel())
    vocal = torch.nn.functional.pad(vocal, (0, length - vocal.numel()))
    instrumental = torch.nn.functional.pad(
        instrumental, (0, length - instrumental.numel())
    )
    mixed = vocal * vocal_gain + instrumental * instrument_gain
    peak = float(mixed.abs().max()) if mixed.numel() else 0.0
    clipped_samples = int((mixed.abs() > 1.0).sum())
    return mixed.clamp(-1.0, 1.0), {
        "peak_before_clamp": peak,
        "clipped_samples": clipped_samples,
        "sample_count": length,
    }


def run_transfer(args: argparse.Namespace) -> int:
    vocal = load_audio(args.source, args.sample_rate)
    instrumental = load_audio(args.target, args.sample_rate)
    aligned = tempo_align(vocal, args.tempo_scale)
    mixed, diagnostics = mix_vocal_and_instrument(
        aligned, instrumental, args.vocal_gain, args.instrument_gain
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(
        args.output, args.sample_rate, (mixed.numpy() * 32767).astype("int16")
    )
    if args.aligned_output:
        args.aligned_output.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(
            args.aligned_output,
            args.sample_rate,
            (aligned.numpy() * 32767).astype("int16"),
        )
    metadata = RunMetadata(
        experiment_name="singalign-study-2",
        run_name=args.run_name,
        run_kind="exploratory",
        dataset="pjs",
        dataset_version=args.dataset_version,
        split_fingerprint=args.split_fingerprint,
        seed=args.seed,
    )
    with tracked_run(
        metadata,
        {
            "source_id": args.source_id,
            "target_id": args.target_id,
            "tempo_scale": args.tempo_scale,
            "transpose_semitones": args.transpose_semitones,
            "sample_rate": args.sample_rate,
            "vocal_gain": args.vocal_gain,
            "instrument_gain": args.instrument_gain,
        },
    ) as run:
        mlflow = __import__("mlflow")
        for name, value in diagnostics.items():
            mlflow.log_metric(f"transfer.{name}", value)
        log_json_artifact(
            {
                "diagnostics": diagnostics,
                "source": str(args.source),
                "target": str(args.target),
            },
            "transfer-metadata.json",
        )
        run_id = run.info.run_id
    print(json.dumps({"output": str(args.output), "mlflow_run_id": run_id}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-transfer")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligned-output", type=Path)
    parser.add_argument("--source-id", default="source")
    parser.add_argument("--target-id", default="target")
    parser.add_argument("--tempo-scale", type=float, default=1.0)
    parser.add_argument("--transpose-semitones", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--vocal-gain", type=float, default=1.0)
    parser.add_argument("--instrument-gain", type=float, default=1.0)
    parser.add_argument("--dataset-version", default="1.1")
    parser.add_argument("--split-fingerprint", default="not-provided")
    parser.add_argument("--run-name", default="transfer-control")
    parser.add_argument("--seed", type=int, default=2026)
    return run_transfer(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
