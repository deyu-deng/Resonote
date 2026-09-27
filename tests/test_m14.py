"""M14 — playability: a left hand spans 4-5 frets.

Contract of :func:`arrange.enforce_playability`:
  * melody and bass are never dropped;
  * bass/harmony may be RE-VOICED (same pitch, another string/fret) -- the
    pitch is preserved, so ``pitch == OPEN_MIDI[string] + fret`` still holds;
  * harmony is dropped only when no re-voice can bring the span in;
  * the span is judged over *sounding* notes on a sixteenth grid, because a
    bass held for two beats still occupies the hand while the melody moves;
  * open strings cost no finger and do not constrain the hand position.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlacedNote
from arrange import OPEN_MIDI, enforce_playability


def pn(onset, dur, fret, role="harmony", string=3):
    """A physically consistent note: pitch follows from string + fret."""
    return PlacedNote(pitch=OPEN_MIDI[string] + fret, onset=onset,
                      duration=dur, string=string, fret=fret,
                      finger=0, pluck="", role=role)


def _span(notes):
    f = [p.fret for p in notes if p.fret > 0]
    return (max(f) - min(f)) if len(f) > 1 else 0


class TestEnforcePlayability(unittest.TestCase):
    def test_comfortable_voicing_untouched(self):
        placed = [pn(0, 1.0, 1, "bass", 6), pn(0, 1.0, 3, "melody", 2),
                  pn(0, 1.0, 4, "harmony", 3)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertEqual(len(out), 3)
        self.assertEqual(_span(out), 3)

    def test_open_strings_do_not_count_as_a_stretch(self):
        # open bass under a fret-12 melody is a normal fingerstyle figure
        placed = [pn(0, 2.0, 0, "bass", 6), pn(0, 2.0, 12, "melody", 1)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertEqual(len(out), 2)

    def test_sustained_bass_counts_even_after_its_onset(self):
        # the bass rang two beats ago and is still held: the hand is still on
        # it, so a sweep over *sounding* notes sees a 14-fret stretch that a
        # sweep over simultaneous onsets would miss entirely.
        # Pitch 41 (E2) exists only at string 6 fret 1 and the melody is at
        # fret 15, so no re-voice closes the gap -- and dropping the harmony
        # would not fix it either (the stretch is bass-vs-melody). When
        # nothing works the embellishment is kept: subtracting it would be
        # destruction without a payoff. Re-fingering the melody is the real
        # fix and belongs to the fingering DP, not to this pass.
        placed = [pn(0, 2.0, 1, "bass", 6), pn(1.0, 0.5, 15, "melody", 1),
                  pn(1.0, 0.5, 14, "harmony", 4)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertEqual(sorted(p.role for p in out),
                         ["bass", "harmony", "melody"])

    def test_revoice_is_preferred_over_dropping(self):
        # bass an octave below the melody: same pitch exists higher up the
        # neck, so the harmony survives instead of being sacrificed
        placed = [pn(0, 2.0, 1, "bass", 6), pn(0, 2.0, 3, "melody", 2),
                  pn(0, 2.0, 10, "harmony", 4)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertLessEqual(_span(out), 5)
        self.assertIn("harmony", {p.role for p in out})
        self.assertIn("bass", {p.role for p in out})

    def test_harmony_dropped_only_when_nothing_else_works(self):
        # a high harmony whose only low position is the melody's own string
        # cannot be re-voiced -> it is the thing to subtract
        placed = [pn(0, 1.0, 2, "melody", 1), pn(0, 1.0, 18, "harmony", 4)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertEqual([p.role for p in out], ["melody"])

    def test_refit_lands_in_a_low_position_when_one_exists(self):
        # two harmonies an octave apart: the high one can come down to fret 3
        # on string 2, so nothing is dropped and the hand stays low
        placed = [pn(0, 1.0, 0, "melody", 1), pn(0, 1.0, 2, "harmony", 3),
                  pn(0, 1.0, 16, "harmony", 5)]
        out = enforce_playability(placed, tempo=120.0)
        self.assertEqual(len(out), 3)
        self.assertLessEqual(_span(out), 5)
        fretted = [p.fret for p in out if p.fret > 0]
        self.assertLessEqual(max(fretted), 7)              # low position

    def test_no_tempo_is_a_noop(self):
        placed = [pn(0, 1.0, 0, "melody"), pn(0, 1.0, 18, "harmony")]
        self.assertEqual(len(enforce_playability(placed, tempo=0)), 2)

    def test_empty(self):
        self.assertEqual(enforce_playability([], tempo=120.0), [])


if __name__ == "__main__":
    unittest.main()
