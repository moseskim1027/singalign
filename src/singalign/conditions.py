"""Shared condition registry for multi-method comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import mlflow

from singalign.metrics import reconstruction_metrics
from singalign.metrics import bootstrap_mean_interval


@dataclass(frozen=True)
class ConditionSpec:
    """Human-readable condition identity and checkpoint location."""

    name: str
    checkpoint: Path
    method: str


def validate_conditions(conditions: list[ConditionSpec]) -> list[ConditionSpec]:
    """Validate and return conditions in declared order."""
    if not conditions:
        raise ValueError("at least one condition is required")
    names = [condition.name for condition in conditions]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("condition names must be non-empty and unique")
    if any(not condition.method.strip() for condition in conditions):
        raise ValueError("condition methods must be non-empty")
    return conditions


def compare_condition_outputs(
    reference: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    conditions: list[ConditionSpec],
) -> list[dict[str, object]]:
    """Compute identical reconstruction diagnostics for declared conditions."""
    validate_conditions(conditions)
    if set(outputs) != {condition.name for condition in conditions}:
        raise ValueError("outputs must match the declared condition names")
    return [
        {"name": condition.name, "method": condition.method,
         "metrics": reconstruction_metrics(outputs[condition.name], reference)}
        for condition in conditions
    ]


def write_condition_report(
    reference: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    conditions: list[ConditionSpec],
    output: Path,
) -> dict[str, object]:
    """Write a stable JSON report for one multi-condition example."""
    rows = compare_condition_outputs(reference, outputs, conditions)
    report = {"condition_count": len(rows), "conditions": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def log_condition_report(report_path: Path, artifact_path: str = "conditions") -> None:
    """Attach a serialized condition report to the active MLflow run."""
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    mlflow.log_artifact(str(report_path), artifact_path=artifact_path)


def compare_condition_dataset(
    references: list[torch.Tensor],
    outputs: dict[str, list[torch.Tensor]],
    conditions: list[ConditionSpec],
    seed: int = 2026,
    samples: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Aggregate paired metrics and effects across a shared example set.

    The first declared condition is the comparison anchor. Every condition is
    evaluated against the same reference at each index, and bootstrap samples
    resample example indices rather than mixing conditions independently.
    """
    validate_conditions(conditions)
    if not references:
        raise ValueError("references must not be empty")
    if any(len(outputs.get(condition.name, [])) != len(references) for condition in conditions):
        raise ValueError("each condition must provide one output per reference")
    values: dict[str, dict[str, list[float]]] = {}
    metric_names = ("log_mel_mse", "log_mel_mae", "spectral_convergence")
    for condition in conditions:
        values[condition.name] = {name: [] for name in metric_names}
        for reference, output in zip(references, outputs[condition.name], strict=True):
            metrics = reconstruction_metrics(output, reference)
            for name in metric_names:
                values[condition.name][name].append(metrics[name])
    anchor = conditions[0].name
    rows = []
    for condition_index, condition in enumerate(conditions):
        metrics: dict[str, object] = {}
        for metric_index, name in enumerate(metric_names):
            interval = bootstrap_mean_interval(values[condition.name][name], seed + metric_index, samples, confidence_level)
            metrics[name] = {"mean": interval.mean, "lower": interval.lower, "upper": interval.upper}
        effects: dict[str, object] = {}
        if condition_index:
            for metric_index, name in enumerate(metric_names):
                deltas = [new - old for old, new in zip(values[anchor][name], values[condition.name][name], strict=True)]
                interval = bootstrap_mean_interval(deltas, seed + 100 + metric_index, samples, confidence_level)
                spread = math.sqrt(sum((delta - interval.mean) ** 2 for delta in deltas) / max(len(deltas) - 1, 1))
                effects[name] = {
                    "mean": interval.mean,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "standardized_effect": interval.mean / spread if spread > 1e-12 else 0.0,
                }
        rows.append({"name": condition.name, "method": condition.method, "metrics": metrics, "effect_vs_anchor": effects})
    return {"condition_count": len(rows), "example_count": len(references), "anchor": anchor, "conditions": rows}
