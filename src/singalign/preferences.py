"""Deterministic synthetic preference pairs for proxy alignment studies."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


def degrade(
    features: torch.Tensor,
    family: str,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply a controlled degradation to one log-mel feature tensor."""

    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be between zero and one")
    output = features.clone()
    if family == "noise":
        noise = torch.randn(output.shape, generator=generator, dtype=output.dtype)
        return output + severity * noise
    if family == "time_mask":
        width = max(1, round(output.shape[-1] * severity))
        maximum = max(output.shape[-1] - width, 0)
        start = int(torch.randint(maximum + 1, (1,), generator=generator).item())
        output[..., start : start + width] = 0.0
        return output
    if family == "frequency_mask":
        width = max(1, round(output.shape[-2] * severity))
        maximum = max(output.shape[-2] - width, 0)
        start = int(torch.randint(maximum + 1, (1,), generator=generator).item())
        output[..., start : start + width, :] = 0.0
        return output
    raise ValueError(f"unknown degradation family: {family}")


class PreferencePairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Wrap reconstruction examples with deterministic chosen/rejected pairs."""

    families = ("noise", "time_mask", "frequency_mask")

    def __init__(
        self,
        dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
        seed: int,
        chosen_severity: float,
        rejected_severity: float,
    ) -> None:
        if chosen_severity >= rejected_severity:
            raise ValueError("chosen severity must be lower than rejected severity")
        self.dataset = dataset
        self.seed = seed
        self.chosen_severity = chosen_severity
        self.rejected_severity = rejected_severity

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        inputs, target = self.dataset[index]
        family = self.families[index % len(self.families)]
        chosen_generator = torch.Generator().manual_seed(self.seed + index * 2)
        rejected_generator = torch.Generator().manual_seed(self.seed + index * 2 + 1)
        chosen = degrade(target, family, self.chosen_severity, chosen_generator)
        rejected = degrade(target, family, self.rejected_severity, rejected_generator)
        return inputs, chosen, rejected


def preference_parameters(config: dict[str, Any]) -> tuple[float, float]:
    """Extract and validate configured degradation severities."""

    chosen = float(config["chosen_severity"])
    rejected = float(config["rejected_severity"])
    if not 0.0 <= chosen < rejected <= 1.0:
        raise ValueError(
            "preference severities must satisfy 0 <= chosen < rejected <= 1"
        )
    return chosen, rejected
