from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from singalign.conditioning import frame_conditioning, load_conditioning


class ConditioningTest(unittest.TestCase):
    def test_loads_musicxml_notes_and_phonemes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "score.musicxml").write_text(
                """<score-partwise><part><measure><attributes><divisions>2</divisions></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration></note><note><rest/><duration>4</duration></note></measure></part></score-partwise>"""
            )
            (root / "voice.lab").write_text("0 100 aa\n100 200 pau\n")
            record = load_conditioning(root / "score.musicxml", root / "voice.lab")
            self.assertEqual(record.notes[0].midi, 60)
            self.assertEqual(record.notes[0].duration, 1.0)
            self.assertIsNone(record.notes[1].midi)
            self.assertEqual(record.phonemes[1][2], "pau")
            self.assertEqual(record.pitch_metadata["midi_pitch_min"], 60)
            self.assertEqual(record.pitch_metadata["midi_pitch_max"], 60)
            self.assertEqual(record.pitch_metadata["rest_count"], 1)
            frames = frame_conditioning(record, 2, 2, 1)
            self.assertEqual(len(frames), 4)
            self.assertEqual(frames[0]["midi_pitch"], 60)
            self.assertEqual(frames[0]["voiced"], 1)
            self.assertIsNone(frames[2]["midi_pitch"])


if __name__ == "__main__":
    unittest.main()
