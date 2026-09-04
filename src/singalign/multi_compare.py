"""CLI for comparing saved mel outputs from multiple sandbox conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import mlflow

from singalign.conditions import ConditionSpec, write_condition_report
from singalign.tracking import RunMetadata, tracked_run


def _condition(value: str) -> tuple[ConditionSpec, Path]:
    """Parse ``name=output.pt:method`` while retaining the output path."""
    try:
        identity, method = value.rsplit(":", 1)
        name, path = identity.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("condition must be name=output.pt:method") from error
    return ConditionSpec(name, Path(path), method), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-multi-compare")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", type=_condition, action="append", required=True)
    parser.add_argument("--mlflow-experiment")
    args = parser.parse_args()
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    if not isinstance(reference, torch.Tensor):
        raise ValueError("reference must contain a torch Tensor")
    conditions = [item[0] for item in args.condition]
    outputs = {condition.name: torch.load(path, map_location="cpu", weights_only=True)
               for condition, path in args.condition}
    if any(not isinstance(output, torch.Tensor) for output in outputs.values()):
        raise ValueError("condition files must contain torch Tensors")
    if args.mlflow_experiment:
        metadata = RunMetadata(args.mlflow_experiment, "multi-condition-comparison", "exploratory", "local-mel-input", "unknown", "not-applicable", 0)
        with tracked_run(metadata, {"condition_count": len(conditions)}):
            report = write_condition_report(reference, outputs, conditions, args.output)
            mlflow.log_artifact(str(args.output), artifact_path="conditions")
    else:
        report = write_condition_report(reference, outputs, conditions, args.output)
    print({"output": str(args.output), "condition_count": report["condition_count"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
