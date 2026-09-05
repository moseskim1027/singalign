"""Render a fixed, score-driven Study 2 instrumental."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.io import wavfile

from singalign.conditioning import read_musicxml_notes
from singalign.tracking import RunMetadata, log_json_artifact, tracked_run
from singalign.transfer import render_note_events


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(args: argparse.Namespace) -> int:
    notes = read_musicxml_notes(args.score)
    audio = render_note_events(
        [(note.onset, note.duration, note.midi) for note in notes],
        args.bpm,
        args.sample_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(
        args.output, args.sample_rate, (audio.numpy() * 32767).astype("int16")
    )
    metadata = {
        "score": str(args.score),
        "score_sha256": sha256(args.score),
        "output": str(args.output),
        "sample_rate": args.sample_rate,
        "bpm": args.bpm,
        "renderer": "additive-sine-v1",
        "attack_seconds": 0.01,
        "note_count": len(notes),
    }
    run_metadata = RunMetadata(
        "singalign-study-2-rendering",
        args.run_name,
        "exploratory",
        "pjs",
        "1.1",
        "not-provided",
        args.seed,
    )
    with tracked_run(run_metadata, metadata) as run:
        log_json_artifact(metadata, "instrumental-render-metadata.json")
        run_id = run.info.run_id
    result = {"output": str(args.output), "mlflow_run_id": run_id, **metadata}
    print(json.dumps(result, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-render-instrumental")
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--run-name", default="fixed-midi-render")
    parser.add_argument("--seed", type=int, default=2026)
    return render(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
