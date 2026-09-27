"""M12 — importing existing tablature files (no re-arrangement).

A tab somebody already made is ground truth, so the importer must preserve
``string`` / ``fret`` and the source tuning, and must merge tied continuation
beats back into the note they belong to instead of creating extra plucks.

The round-trip here is the real test: export our own arrangement, import it
back, and check we get the same notes out.
"""

import os
import sys
import tempfile
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlacedNote
from gp_export import to_gp5
from importers import is_tab_file, load_gp

_STANDARD = [64, 59, 55, 50, 45, 40]


def pn(pitch, onset, dur, string=1, fret=0):
    return PlacedNote(pitch=pitch, onset=onset, duration=dur, string=string,
                      fret=fret, finger=0, pluck="", role="melody")


class TestIsTabFile(unittest.TestCase):
    def test_recognises_gp_versions(self):
        for name in ("a.gp5", "a.GP4", "a.gp3", "a.gtp", "x/y/b.GP5"):
            self.assertTrue(is_tab_file(name), name)

    def test_rejects_everything_else(self):
        for name in ("a.mp3", "a.wav", "a.mid", "a.pdf", "a.gpx", "a.gp"):
            self.assertFalse(is_tab_file(name), name)

    def test_gp7_container_not_supported_yet(self):
        # .gp (GP7/8) is a ZIP+XML container pyguitarpro cannot read
        self.assertFalse(is_tab_file("song.gp"))
        self.assertFalse(is_tab_file("song.gpx"))


class TestImportRoundTrip(unittest.TestCase):
    """Export -> import -> same notes, same fingering, same tuning."""

    @staticmethod
    def _roundtrip(placed, tempo=120.0, **kw):
        fd, path = tempfile.mkstemp(suffix=".gp5")
        os.close(fd)
        try:
            to_gp5(placed, path, tempo=tempo, **kw)
            return load_gp(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_notes_and_fingering_survive(self):
        # pitch must agree with string + fret: open strings are
        # 1..6 = 64,59,55,50,45,40 (standard EADGBE)
        placed = [pn(60, 0.0, 0.5, 3, 5),      # 55 + 5
                  pn(62, 0.5, 0.5, 2, 3),      # 59 + 3
                  pn(71, 1.0, 0.5, 1, 7)]      # 64 + 7
        got = self._roundtrip(placed)
        self.assertEqual(len(got.placed), len(placed))
        for src, dst in zip(placed, got.placed):
            self.assertEqual((dst.string, dst.fret), (src.string, src.fret))
            self.assertEqual(dst.pitch, src.pitch)

    def test_ties_collapse_into_one_note(self):
        """A note held over 4 beats must import as ONE note, not four."""
        placed = [pn(64, 0.0, 2.0, 6, 0)]       # 2 s at 120 BPM = 4 beats
        got = self._roundtrip(placed)
        self.assertEqual(len(got.placed), 1,
                         "tied continuation beats must not become extra notes")
        self.assertAlmostEqual(got.placed[0].duration, 2.0, places=2)

    def test_tuning_is_preserved(self):
        got = self._roundtrip([pn(64, 0.0, 0.5, 6, 0)])
        self.assertEqual(got.tuning, _STANDARD)

    def test_tempo_and_measure_count(self):
        placed = [pn(64, 0.0, 0.5, 1, 0), pn(64, 1.0, 0.5, 1, 0),
                  pn(64, 2.0, 0.5, 1, 0)]
        got = self._roundtrip(placed, tempo=100.0)
        self.assertAlmostEqual(got.tempo, 100.0, places=1)
        self.assertGreaterEqual(got.measures, 1)

    def test_timeline_starts_at_zero(self):
        """pyguitarpro hands measure 0 a non-zero start tick; anchor it."""
        got = self._roundtrip([pn(64, 0.0, 0.5, 1, 0), pn(67, 0.5, 0.5, 1, 3)])
        self.assertAlmostEqual(min(p.onset for p in got.placed), 0.0, places=3)

    def test_no_same_string_collision_after_import(self):
        placed = [pn(64, 0.0, 0.5, 3, 5), pn(67, 0.0, 0.5, 2, 3),
                  pn(71, 0.0, 0.5, 1, 7)]
        got = self._roundtrip(placed)
        by_onset = Counter(round(p.onset, 4) for p in got.placed)
        for onset, count in by_onset.items():
            strings = [p.string for p in got.placed
                       if round(p.onset, 4) == onset]
            self.assertEqual(len(strings), len(set(strings)),
                             f"two notes on one string at {onset}")


if __name__ == "__main__":
    unittest.main()
