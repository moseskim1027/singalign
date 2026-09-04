"""Deterministic symbolic conditioning records for score-based experiments."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NoteEvent:
    """A pitched or rest event expressed in quarter-note units."""

    onset: float
    duration: float
    midi: int | None


@dataclass(frozen=True)
class ConditioningRecord:
    """Score and phoneme conditioning aligned to one corpus example."""

    notes: tuple[NoteEvent, ...]
    phonemes: tuple[tuple[int, int, str], ...]

    @property
    def pitch_metadata(self) -> dict[str, float | int | None]:
        """Return deterministic summary metadata for the score pitches."""

        pitches = [note.midi for note in self.notes if note.midi is not None]
        return {
            "note_count": len(self.notes),
            "voiced_note_count": len(pitches),
            "rest_count": len(self.notes) - len(pitches),
            "midi_pitch_min": min(pitches) if pitches else None,
            "midi_pitch_max": max(pitches) if pitches else None,
            "midi_pitch_mean": sum(pitches) / len(pitches) if pitches else None,
        }


def read_phoneme_labels(path: Path) -> tuple[tuple[int, int, str], ...]:
    """Read validated PJS phoneme intervals without exposing audio content."""

    labels: list[tuple[int, int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        start, end, phoneme = line.split()
        labels.append((int(start), int(end), phoneme))
    if not labels:
        raise ValueError("phoneme label file is empty")
    return tuple(labels)


def _midi_number(step: str, octave: int, alter: int = 0) -> int:
    semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    return 12 * (octave + 1) + semitones[step] + alter


def read_musicxml_notes(path: Path) -> tuple[NoteEvent, ...]:
    """Extract deterministic note events from a MusicXML partwise score."""

    root = ET.parse(path).getroot()
    divisions = 1.0
    onset = 0.0
    events: list[NoteEvent] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "divisions" and element.text:
            divisions = float(element.text)
        elif tag == "note":
            duration_node = element.find("{*}duration")
            if duration_node is None or not duration_node.text:
                continue
            duration = float(duration_node.text) / divisions
            pitch = element.find("{*}pitch")
            midi = None
            if pitch is not None:
                step = pitch.findtext("{*}step")
                octave = pitch.findtext("{*}octave")
                alter = int(pitch.findtext("{*}alter", "0"))
                if step is not None and octave is not None:
                    midi = _midi_number(step, int(octave), alter)
            events.append(NoteEvent(onset, duration, midi))
            onset += duration
    if not events:
        raise ValueError("MusicXML contains no note events")
    return tuple(events)


def load_conditioning(musicxml: Path, labels: Path) -> ConditioningRecord:
    """Load symbolic score and phoneme conditioning for one example."""

    return ConditioningRecord(
        read_musicxml_notes(musicxml), read_phoneme_labels(labels)
    )
