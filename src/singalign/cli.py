"""Command-line interface for SingAlign research data workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from singalign.conditioning import load_conditioning
from singalign.data.pjs import (
    build_index,
    create_splits,
    format_issues,
    validate_corpus,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="singalign-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a PJS corpus")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--expected-count", type=int, default=100)

    index = subparsers.add_parser("index", help="build a PJS JSONL index")
    index.add_argument("--root", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--expected-count", type=int, default=100)

    split = subparsers.add_parser("split", help="create deterministic data splits")
    split.add_argument("--index", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=2026)
    split.add_argument("--train-count", type=int, default=80)
    split.add_argument("--validation-count", type=int, default=10)

    conditioning = subparsers.add_parser(
        "conditioning", help="export deterministic score/phoneme conditioning"
    )
    conditioning.add_argument("--musicxml", type=Path, required=True)
    conditioning.add_argument("--labels", type=Path, required=True)
    conditioning.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        report = validate_corpus(args.root, expected_count=args.expected_count)
        if report.valid:
            print(f"PJS validation passed: {report.examples} examples")
            return 0
        print(format_issues(report.issues))
        print(f"PJS validation failed: {len(report.issues)} issue(s)")
        return 1
    if args.command == "index":
        records = build_index(
            args.root, args.output, expected_count=args.expected_count
        )
        print(f"Wrote {len(records)} records to {args.output}")
        return 0
    if args.command == "split":
        splits = create_splits(
            args.index,
            args.output,
            seed=args.seed,
            train_count=args.train_count,
            validation_count=args.validation_count,
        )
        print(
            "Wrote splits to "
            f"{args.output}: train={len(splits['train'])}, "
            f"validation={len(splits['validation'])}, test={len(splits['test'])}"
        )
        return 0
    if args.command == "conditioning":
        record = load_conditioning(args.musicxml, args.labels)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "notes": [note.__dict__ for note in record.notes],
                    "phonemes": record.phonemes,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote conditioning record to {args.output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
