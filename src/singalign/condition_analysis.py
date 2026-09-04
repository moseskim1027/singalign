"""CLI for aggregate paired analysis across saved condition outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from singalign.conditions import ConditionSpec, compare_condition_dataset


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-condition-analysis")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    references = [torch.load(path, map_location="cpu", weights_only=True) for path in manifest["references"]]
    conditions = [ConditionSpec(item["name"], Path(item["checkpoint"]), item["method"]) for item in manifest["conditions"]]
    outputs = {
        item["name"]: [torch.load(path, map_location="cpu", weights_only=True) for path in item["outputs"]]
        for item in manifest["conditions"]
    }
    tensors = references + [tensor for items in outputs.values() for tensor in items]
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("manifest tensors must contain torch Tensors")
    report = compare_condition_dataset(references, outputs, conditions, args.seed, args.samples, args.confidence_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote condition analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
