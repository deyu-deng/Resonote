"""M14 — playability: a left hand spans 4-5 frets."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import PlacedNote
from arrange import enforce_playability


def pn(onset, dur, fret, role="harmony", string=3):
    return PlacedNote(pitch=60 + fret, onset=onset, duration=dur,
                      string=string, fret=fret, finger=0, pluck="", role=role)


class TestEnforcePlayability(unittest.TestCase):
    def test_wide_voicing_loses_harmony_not_melody(self):
        # melody fret 2 + harmony fret 14 sounding together -> span 12
        placed = [pn(0, 1.0, 2, "melody", 1), pn(0, 1.0, 14, "harmony", 5)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertEqual([p.role for p in out], ["melody"])

    def test_sustained_bass_counts_even_after_its_onset(self):
        # bass rings 2 beats; melody jumps at t=1 -> still one hand
        placed = [pn(0, 2.0, 1, "bass", 6), pn(1.0, 0.5, 15, "melody", 1),
                  pn(1.0, 0.5, 14, "harmony", 4)]
        out = enforce_playability(placed, tempo=120.0)
        roles = sorted(p.role for p in out)
        self.assertEqual(roles, ["bass", "melody"])

    def test_comfortable_voicing_untouched(self):
        placed = [pn(0, 1.0, 1, "bass", 6), pn(0, 1.0, 3, "melody", 2),
                  pn(0, 1.0, 4, "harmony", 3)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertEqual(len(out), 3)

    def test_nearest_to_median_dropped_first(self):
        placed = [pn(0, 1.0, 0, "melody", 1), pn(0, 1.0, 2, "harmony", 3),
                  pn(0, 1.0, 16, "harmony", 5)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertEqual(sorted(p.fret for p in out), [0, 2])

    def test_no_tempo_is_a_noop(self):
        placed = [pn(0, 1.0, 0, "melody"), pn(0, 1.0, 18, "harmony")]
        self.assertEqual(len(enforce_playability(placed, tempo=0)), 2)

    def test_empty(self):
        self.assertEqual(enforce_playability([], tempo=120.0), [])


if __name__ == "__main__":
    unittest.main()
