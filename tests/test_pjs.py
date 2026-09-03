from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from singalign.data.pjs import build_index, create_splits, validate_corpus


def _write_wave(path: Path, *, channels: int = 1, rate: int = 48_000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(3)
        audio.setframerate(rate)
        audio.writeframes(b"\x00\x00\x00" * channels * 480)


def _make_example(root: Path, item_id: str = "pjs001") -> Path:
    example = root / item_id
    example.mkdir(parents=True)
    _write_wave(example / f"{item_id}_song.wav")
    _write_wave(example / f"{item_id}_speech.wav")
    (example / f"{item_id}.lab").write_text(
        "0 100000 pau\n100000 200000 a\n", encoding="utf-8"
    )
    (example / f"{item_id}.mid").write_bytes(
        b"MThd" + struct.pack(">I", 6) + b"\x00" * 6
    )
    (example / f"{item_id}.musicxml").write_text(
        '<?xml version="1.0"?><score-partwise version="3.1"/>', encoding="utf-8"
    )
    (example / f"{item_id}.txt").write_text(
        "key:C maj\nBPM:120\nジャンル:テスト\nスケール:メジャー\n",
        encoding="utf-8",
    )
    return example


class PJSValidationTests(unittest.TestCase):
    def test_valid_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_example(root)
            report = validate_corpus(root, expected_count=1)
            self.assertTrue(report.valid, report.issues)

    def test_missing_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = _make_example(root)
            (example / "pjs001.mid").unlink()
            report = validate_corpus(root, expected_count=1)
            self.assertIn("missing_file", {issue.code for issue in report.issues})

    def test_invalid_audio_format_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = _make_example(root)
            _write_wave(example / "pjs001_song.wav", rate=24_000)
            report = validate_corpus(root, expected_count=1)
            self.assertIn(
                "invalid_audio_format", {issue.code for issue in report.issues}
            )

    def test_overlapping_labels_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = _make_example(root)
            (example / "pjs001.lab").write_text(
                "0 200000 a\n100000 300000 i\n", encoding="utf-8"
            )
            report = validate_corpus(root, expected_count=1)
            self.assertIn("invalid_labels", {issue.code for issue in report.issues})

    def test_invalid_score_files_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            example = _make_example(root)
            (example / "pjs001.mid").write_bytes(b"nope")
            (example / "pjs001.musicxml").write_text("<broken>", encoding="utf-8")
            report = validate_corpus(root, expected_count=1)
            codes = {issue.code for issue in report.issues}
            self.assertIn("invalid_midi", codes)
            self.assertIn("invalid_musicxml", codes)

    def test_unexpected_count_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_example(root)
            report = validate_corpus(root, expected_count=100)
            self.assertIn(
                "unexpected_example_count", {issue.code for issue in report.issues}
            )

    def test_index_and_splits_are_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            for number in range(1, 7):
                _make_example(root, f"pjs{number:03d}")
            index = Path(directory) / "index.jsonl"
            records = build_index(root, index, expected_count=6)
            self.assertEqual(len(records), 6)
            indexed = [json.loads(line) for line in index.read_text().splitlines()]
            self.assertEqual(indexed[0]["bpm"], 120)

            first = Path(directory) / "splits-a.json"
            second = Path(directory) / "splits-b.json"
            split_a = create_splits(
                index, first, seed=7, train_count=3, validation_count=1
            )
            split_b = create_splits(
                index, second, seed=7, train_count=3, validation_count=1
            )
            self.assertEqual(split_a, split_b)
            groups = [set(split_a[name]) for name in ("train", "validation", "test")]
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])


if __name__ == "__main__":
    unittest.main()
