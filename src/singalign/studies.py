"""Manifest-driven entry points for the two PJS study workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from singalign.tracking import RunMetadata, log_json_artifact, tracked_run


def file_fingerprint(path: Path) -> str:
    """Return a stable SHA-256 fingerprint for a manifest or split file."""

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_pair_manifest(payload: dict[str, Any]) -> None:
    """Validate required Study 2 pair fields and controlled values."""

    if payload.get("schema_version") != "study-2-pairs-v1":
        raise ValueError("unsupported Study 2 pair-manifest schema")
    if payload.get("dataset") != "pjs" or not payload.get("dataset_version"):
        raise ValueError("pair manifest must declare the PJS dataset and version")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pair manifest must contain at least one pair")
    seen: set[tuple[str, str]] = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("each pair must be an object")
        source, target = pair.get("source_id"), pair.get("target_id")
        if not isinstance(source, str) or not isinstance(target, str) or source == target:
            raise ValueError("pairs require distinct source_id and target_id")
        if (source, target) in seen:
            raise ValueError("pair manifest contains duplicates")
        seen.add((source, target))
        scale = pair.get("tempo_scale", 1.0)
        transpose = pair.get("transpose_semitones", 0)
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError("tempo_scale must be positive")
        if not isinstance(transpose, int) or not -24 <= transpose <= 24:
            raise ValueError("transpose_semitones must be an integer from -24 to 24")


def run_manifest(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_pair_manifest(manifest)
    split_fingerprint = file_fingerprint(args.split) if args.split else "not-provided"
    metadata = RunMetadata(
        experiment_name="singalign-study-2",
        run_name=f"study-2-{args.run_name}",
        run_kind=args.run_kind,
        dataset="pjs",
        dataset_version=str(manifest["dataset_version"]),
        split_fingerprint=split_fingerprint,
        seed=args.seed,
    )
    parameters = {
        "study": "content-and-melody-transfer",
        "manifest_schema": manifest["schema_version"],
        "manifest_fingerprint": file_fingerprint(args.manifest),
        "pair_count": len(manifest["pairs"]),
        "split": manifest.get("split", "unspecified"),
    }
    with tracked_run(metadata, parameters):
        log_json_artifact(manifest, "study-2-pair-manifest.json")
        print(f"Started MLflow study run with {len(manifest['pairs'])} pair(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-study")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--run-name", default="manifest-validation")
    parser.add_argument("--run-kind", choices=("exploratory", "confirmatory"), default="exploratory")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    return run_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
