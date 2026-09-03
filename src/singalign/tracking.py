"""Standardized MLflow tracking for SingAlign experiments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

RunKind = Literal["exploratory", "confirmatory"]


@dataclass(frozen=True)
class RunMetadata:
    """Required identity and reproducibility metadata for a tracked run."""

    experiment_name: str
    run_name: str
    run_kind: RunKind
    dataset: str
    dataset_version: str
    split_fingerprint: str
    seed: int

    def __post_init__(self) -> None:
        if self.run_kind not in ("exploratory", "confirmatory"):
            raise ValueError("run_kind must be exploratory or confirmatory")
        for field in (
            "experiment_name",
            "run_name",
            "dataset",
            "dataset_version",
            "split_fingerprint",
        ):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must not be empty")


def git_revision(repository: Path | None = None) -> tuple[str, bool]:
    """Return the current Git revision and whether tracked files are dirty."""

    revision_override = os.environ.get("SINGALIGN_GIT_REVISION")
    dirty_override = os.environ.get("SINGALIGN_GIT_DIRTY")
    if revision_override and revision_override != "unknown":
        return revision_override, dirty_override == "true"

    cwd = repository or Path.cwd()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def flatten_parameters(
    values: Mapping[str, Any], prefix: str = ""
) -> dict[str, str | int | float | bool]:
    """Flatten nested mappings into MLflow-compatible parameter values."""

    flattened: dict[str, str | int | float | bool] = {}
    for key in sorted(values):
        value = values[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_parameters(value, name))
        elif isinstance(value, str | int | float | bool):
            flattened[name] = value
        elif value is None:
            flattened[name] = "null"
        else:
            flattened[name] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return flattened


def _load_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow is not installed; run `uv sync` or use Docker Compose"
        ) from error
    return mlflow


def _tags(metadata: RunMetadata, repository: Path | None) -> dict[str, str]:
    revision, dirty = git_revision(repository)
    return {
        "research.run_kind": metadata.run_kind,
        "data.dataset": metadata.dataset,
        "data.version": metadata.dataset_version,
        "data.split_fingerprint": metadata.split_fingerprint,
        "code.git_revision": revision,
        "code.git_dirty": str(dirty).lower(),
        "runtime.python": platform.python_version(),
        "runtime.platform": platform.platform(),
    }


@contextmanager
def tracked_run(
    metadata: RunMetadata,
    parameters: Mapping[str, Any] | None = None,
    repository: Path | None = None,
) -> Iterator[Any]:
    """Start an MLflow run with the repository's required metadata contract."""

    mlflow = _load_mlflow()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(metadata.experiment_name)
    started = time.monotonic()
    with mlflow.start_run(
        run_name=metadata.run_name, tags=_tags(metadata, repository)
    ) as run:
        mlflow.log_params(
            {
                "seed": metadata.seed,
                **flatten_parameters(parameters or {}),
            }
        )
        try:
            yield run
        finally:
            mlflow.log_metric("runtime.elapsed_seconds", time.monotonic() - started)


def log_json_artifact(payload: Mapping[str, Any], filename: str) -> None:
    """Log a mapping as a formatted JSON artifact in the active run."""

    mlflow = _load_mlflow()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(path), artifact_path="metadata")


def _smoke(args: argparse.Namespace) -> int:
    metadata = RunMetadata(
        experiment_name=args.experiment,
        run_name="tracking-smoke",
        run_kind="exploratory",
        dataset="pjs",
        dataset_version="1.1",
        split_fingerprint="smoke-test",
        seed=2026,
    )
    mlflow = _load_mlflow()
    with tracked_run(metadata, {"command": "smoke"}):
        mlflow.log_metric("smoke.success", 1.0)
        log_json_artifact(asdict(metadata), "run-metadata.json")
    print(f"MLflow smoke run completed for experiment {args.experiment!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-track")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="log a minimal MLflow run")
    smoke.add_argument("--experiment", default="singalign-smoke")
    args = parser.parse_args()
    if args.command == "smoke":
        return _smoke(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
