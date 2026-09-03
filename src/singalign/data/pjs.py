"""Validation, indexing, and deterministic splitting for PJS version 1.1."""

from __future__ import annotations

import hashlib
import json
import random
import re
import tempfile
import wave
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

PJS_ID_PATTERN = re.compile(r"pjs\d{3}")
EXPECTED_AUDIO = {"channels": 1, "sample_rate": 48_000, "sample_width_bytes": 3}


@dataclass(frozen=True)
class ValidationIssue:
    """A corpus validation error tied to a path when available."""

    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a PJS validation pass."""

    root: str
    examples: int
    issues: tuple[ValidationIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class PJSRecord:
    """Metadata for one paired PJS song and speech example."""

    id: str
    song_audio: str
    speech_audio: str
    label: str
    midi: str
    musicxml: str
    metadata: str
    song_duration_seconds: float
    speech_duration_seconds: float
    sample_rate: int
    channels: int
    sample_width_bits: int
    phoneme_count: int
    bpm: int | None
    key: str | None
    genre: str | None
    scale: str | None


def _example_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and PJS_ID_PATTERN.fullmatch(path.name)
    )


def _paths(example: Path) -> dict[str, Path]:
    item_id = example.name
    return {
        "song_audio": example / f"{item_id}_song.wav",
        "speech_audio": example / f"{item_id}_speech.wav",
        "label": example / f"{item_id}.lab",
        "midi": example / f"{item_id}.mid",
        "musicxml": example / f"{item_id}.musicxml",
        "metadata": example / f"{item_id}.txt",
    }


def _validate_wave(path: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        with wave.open(str(path), "rb") as audio:
            observed = {
                "channels": audio.getnchannels(),
                "sample_rate": audio.getframerate(),
                "sample_width_bytes": audio.getsampwidth(),
            }
    except (EOFError, wave.Error) as error:
        return [ValidationIssue("invalid_wav", str(error), str(path))]

    for field, expected in EXPECTED_AUDIO.items():
        if observed[field] != expected:
            issues.append(
                ValidationIssue(
                    "invalid_audio_format",
                    f"expected {field}={expected}, observed {observed[field]}",
                    str(path),
                )
            )
    return issues


def _read_labels(path: Path) -> list[tuple[int, int, str]]:
    labels: list[tuple[int, int, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw_line.split()
        if len(fields) != 3:
            raise ValueError(f"line {line_number}: expected start, end, phoneme")
        start, end = int(fields[0]), int(fields[1])
        if start < 0 or end <= start:
            raise ValueError(f"line {line_number}: invalid interval {start} {end}")
        if labels and start < labels[-1][1]:
            raise ValueError(f"line {line_number}: interval overlaps previous label")
        labels.append((start, end, fields[2]))
    if not labels:
        raise ValueError("label file is empty")
    return labels


def _parse_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        if ":" not in raw_line:
            raise ValueError(f"metadata line has no colon: {raw_line!r}")
        key, value = raw_line.split(":", 1)
        metadata[key.strip()] = value.strip()
    if "BPM" in metadata:
        int(metadata["BPM"])
    return metadata


def _validate_example(example: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    paths = _paths(example)
    for kind, path in paths.items():
        if not path.is_file():
            issues.append(
                ValidationIssue("missing_file", f"missing required {kind}", str(path))
            )
    if issues:
        return issues

    for audio_key in ("song_audio", "speech_audio"):
        issues.extend(_validate_wave(paths[audio_key]))

    try:
        _read_labels(paths["label"])
    except (OSError, UnicodeError, ValueError) as error:
        issues.append(
            ValidationIssue("invalid_labels", str(error), str(paths["label"]))
        )

    try:
        ET.parse(paths["musicxml"])
    except (OSError, ET.ParseError) as error:
        issues.append(
            ValidationIssue("invalid_musicxml", str(error), str(paths["musicxml"]))
        )

    try:
        if paths["midi"].read_bytes()[:4] != b"MThd":
            raise ValueError("missing MIDI header")
    except (OSError, ValueError) as error:
        issues.append(ValidationIssue("invalid_midi", str(error), str(paths["midi"])))

    try:
        _parse_metadata(paths["metadata"])
    except (OSError, UnicodeError, ValueError) as error:
        issues.append(
            ValidationIssue("invalid_metadata", str(error), str(paths["metadata"]))
        )
    return issues


def validate_corpus(root: Path, expected_count: int = 100) -> ValidationReport:
    """Validate the structure and parseable contents of a PJS corpus directory."""

    root = root.resolve()
    issues: list[ValidationIssue] = []
    examples = _example_directories(root)
    if not root.is_dir():
        issues.append(
            ValidationIssue("missing_root", "corpus root does not exist", str(root))
        )
    if len(examples) != expected_count:
        issues.append(
            ValidationIssue(
                "unexpected_example_count",
                f"expected {expected_count} examples, observed {len(examples)}",
                str(root),
            )
        )
    for example in examples:
        issues.extend(_validate_example(example))
    return ValidationReport(str(root), len(examples), tuple(issues))


def _duration(path: Path) -> tuple[float, int, int, int]:
    with wave.open(str(path), "rb") as audio:
        return (
            audio.getnframes() / audio.getframerate(),
            audio.getframerate(),
            audio.getnchannels(),
            audio.getsampwidth() * 8,
        )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _record(example: Path) -> PJSRecord:
    paths = _paths(example)
    song_duration, sample_rate, channels, sample_width_bits = _duration(
        paths["song_audio"]
    )
    speech_duration, _, _, _ = _duration(paths["speech_audio"])
    labels = _read_labels(paths["label"])
    metadata = _parse_metadata(paths["metadata"])
    bpm = int(metadata["BPM"]) if "BPM" in metadata else None
    return PJSRecord(
        id=example.name,
        song_audio=_display_path(paths["song_audio"]),
        speech_audio=_display_path(paths["speech_audio"]),
        label=_display_path(paths["label"]),
        midi=_display_path(paths["midi"]),
        musicxml=_display_path(paths["musicxml"]),
        metadata=_display_path(paths["metadata"]),
        song_duration_seconds=round(song_duration, 6),
        speech_duration_seconds=round(speech_duration, 6),
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bits=sample_width_bits,
        phoneme_count=len(labels),
        bpm=bpm,
        key=metadata.get("key"),
        genre=metadata.get("ジャンル"),
        scale=metadata.get("スケール"),
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def build_index(root: Path, output: Path, expected_count: int = 100) -> list[PJSRecord]:
    """Validate PJS and write a JSON Lines metadata index."""

    report = validate_corpus(root, expected_count=expected_count)
    if not report.valid:
        messages = "; ".join(issue.message for issue in report.issues[:5])
        raise ValueError(f"corpus validation failed: {messages}")
    records = [_record(example) for example in _example_directories(root.resolve())]
    content = "".join(
        json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    _atomic_write(output, content)
    return records


def _read_index(path: Path) -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("index contains duplicate IDs")
    return records


def create_splits(
    index: Path,
    output: Path,
    seed: int = 2026,
    train_count: int = 80,
    validation_count: int = 10,
) -> dict[str, object]:
    """Create deterministic, disjoint song-level splits from a PJS index."""

    ids = [str(record["id"]) for record in _read_index(index)]
    if train_count < 1 or validation_count < 1:
        raise ValueError("train and validation counts must be positive")
    if train_count + validation_count >= len(ids):
        raise ValueError("split counts leave no test examples")
    random.Random(seed).shuffle(ids)
    splits: dict[str, object] = {
        "dataset": "pjs",
        "dataset_version": "1.1",
        "seed": seed,
        "strategy": "deterministic song-disjoint shuffle",
        "train": sorted(ids[:train_count]),
        "validation": sorted(ids[train_count : train_count + validation_count]),
        "test": sorted(ids[train_count + validation_count :]),
    }
    canonical = json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    splits["fingerprint_sha256"] = fingerprint
    _atomic_write(
        output,
        json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return splits


def format_issues(issues: Iterable[ValidationIssue]) -> str:
    """Format issues for terminal output."""

    return "\n".join(
        f"[{issue.code}] {issue.message}" + (f" ({issue.path})" if issue.path else "")
        for issue in issues
    )
