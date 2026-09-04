"""Report readiness of the declared simulation comparison matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(prog="singalign-matrix-status")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference_root = Path(config["reference_root"])
    references = sorted(reference_root.glob("*.pt"))
    print(f"matrix={config['name']} references={len(references)}")
    ready = True
    for condition in config["conditions"]:
        checkpoint = Path(condition["checkpoint"])
        outputs = sorted(Path(condition["output_root"]).glob("*.pt"))
        complete = bool(references) and checkpoint.is_file() and len(outputs) == len(references)
        ready &= complete
        state = "ready" if complete else "pending"
        print(f"{condition['name']}: {state} references={len(references)} outputs={len(outputs)} checkpoint={checkpoint}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
