"""Shared condition registry for multi-method comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from singalign.metrics import reconstruction_metrics


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
