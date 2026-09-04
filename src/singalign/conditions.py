"""Shared condition registry for multi-method comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
