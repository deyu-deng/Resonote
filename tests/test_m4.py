"""M4 — Quantization (L4.5) + corrected multi-voice GP5 export.

Two things are verified here, both with substance (not just "it runs"):

  * Quantization snaps timing onto the detected beat grid (anchor = downbeat).
  * The GP5 exporter no longer DROPS notes: a parsed-back file must contain
    every placed note, chords land on one beat, long notes use ties, and every
    measure tiles to exactly 4/4.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Note, PlacedNote
from analysis import AnalysisResult, Chord, Section
from quantize import (grid_step, anchor_of, quantize_notes,
                      quantize_placed)
from collections import Counter
from arrange import arrange
from gp_export import to_gp5, SLOTS_PER_MEASURE, GRID, QUARTER


def _sample_notes():
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60]
    return [Note(p, i * 0.5, 0.5, 100) for i, p in enumerate(pitches)]


def _sample_analysis():
    # monophonic -> note-based backend yields tempo 120, beats on the 0.5 grid
    from analysis import analyze
    return analyze(_sample_notes())


def _count_gp5_notes(path):
    import guitarpro as gp
    song = gp.parse(path)
    total = 0
    measures = 0
    for m in song.tracks[0].measures:
        measures += 1
        for v in m.voices:
            for b in v.beats:
                total += len(b.notes)
    return total, measures, song.tempo


class TestQuantize(unittest.TestCase):
    def test_snaps_off_grid_onset(self):
        notes = [Note(60, 0.13, 0.37, 100)]         # off-grid
        a = AnalysisResult(tempo=120.0, beats=[0.0])  # eighth grid = 0.25s
        q = quantize_notes(notes, a, subdivision=2)
        # 0.13 -> nearest 0.25 multiple from anchor 0 = 0.25
        self.assertAlmostEqual(q[0].onset, 0.25, places=4)
        # end 0.50 -> nearest 0.25 = 0.50 ; duration = 0.25
        self.assertAlmostEqual(q[0].duration, 0.25, places=4)

    def test_anchor_uses_first_beat(self):
        a = AnalysisResult(tempo=120.0, beats=[0.5])
        self.assertAlmostEqual(anchor_of(a), 0.5)

    def test_idempotent(self):
        notes = [Note(60, 0.13, 0.37, 100), Note(64, 0.62, 0.4, 100)]
        a = AnalysisResult(tempo=120.0, beats=[0.0])
        once = quantize_notes(notes, a, 2)
        twice = quantize_notes(once, a, 2)
        self.assertEqual([(n.onset, n.duration) for n in once],
                         [(n.onset, n.duration) for n in twice])

    def test_does_not_mutate_input(self):
        n = Note(60, 0.13, 0.37, 100)
        a = AnalysisResult(tempo=120.0, beats=[0.0])
        quantize_notes([n], a, 2)
        self.assertAlmostEqual(n.onset, 0.13, places=4)


class TestExportNoDrop(unittest.TestCase):
    def _attack_pitch_string(self, path):
        import guitarpro as gp
        song = gp.parse(path)
        out = []
        for m in song.tracks[0].measures:
            for v in m.voices:
                for b in v.beats:
                    for nn in b.notes:
                        if nn.type.name == "normal":   # attack, not a tie continuation
                            # pitch = open-string MIDI + fret
                            open_midi = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
                            out.append((open_midi[nn.string] + nn.value, nn.string))
        return out

    def test_all_placed_notes_survive_roundtrip(self):
        notes = _sample_notes()
        a = _sample_analysis()
        placed = arrange(quantize_notes(notes, a, 2), analysis=a)
        placed = quantize_placed(placed, a, 2)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.gp5")
            to_gp5(placed, out, tempo=a.tempo, anchor=a.beats[0])
            attacks = self._attack_pitch_string(out)
            total, measures, tempo = _count_gp5_notes(out)
        # every placed note attacks exactly once (a 'normal' note); ties just
        # continue it, so the attack multiset must equal the placed multiset.
        placed_ms = Counter((p.pitch, p.string) for p in placed)
        self.assertEqual(Counter(attacks), placed_ms,
                         "GP5 dropped or duplicated a note vs the placed list")
        self.assertGreater(len(placed), 15, "expected full fingerstyle, not melody-only")
        self.assertGreater(total, len(placed),
                           "expected ties to extend sustained notes across beats")

    def test_real_tempo_written(self):
        notes = _sample_notes()
        a = _sample_analysis()
        placed = arrange(quantize_notes(notes, a, 2), analysis=a)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.gp5")
            to_gp5(placed, out, tempo=a.tempo, anchor=a.beats[0])
            _, _, tempo = _count_gp5_notes(out)
        self.assertAlmostEqual(tempo, a.tempo, places=1)

    def test_ties_used_for_sustained_notes(self):
        notes = _sample_notes()
        a = _sample_analysis()
        placed = arrange(quantize_notes(notes, a, 2), analysis=a)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.gp5")
            to_gp5(placed, out, tempo=a.tempo, anchor=a.beats[0])
            import guitarpro as gp
            song = gp.parse(out)
            types = []
            for m in song.tracks[0].measures:
                for v in m.voices:
                    for b in v.beats:
                        for nn in b.notes:
                            types.append(nn.type)
        # a whole-bar bass note must produce tie continuations
        self.assertTrue(any(t.name == "tie" for t in types),
                        "sustained notes should use NoteType.tie")

    def test_measures_tile_to_4_4(self):
        notes = _sample_notes()
        a = _sample_analysis()
        placed = arrange(quantize_notes(notes, a, 2), analysis=a)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t.gp5")
            to_gp5(placed, out, tempo=a.tempo, anchor=a.beats[0])
            import guitarpro as gp
            song = gp.parse(out)
            for mi, m in enumerate(song.tracks[0].measures):
                slots = 0
                for v in m.voices:
                    for b in v.beats:
                        # value 8->1 slot, 4->2, 2->4, 1->8
                        slots += 8 // b.duration.value
                self.assertEqual(slots, SLOTS_PER_MEASURE,
                                 f"measure {mi} does not sum to 4/4 (got {slots})")


class TestStringCollision(unittest.TestCase):
    def test_no_same_string_overlap(self):
        notes = _sample_notes()
        a = _sample_analysis()
        placed = arrange(quantize_notes(notes, a, 2), analysis=a)
        # O(n^2): no two notes on the same string with overlapping time
        bad = 0
        pl = sorted(placed, key=lambda p: p.onset)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                if pl[j].onset >= pl[i].onset + pl[i].duration:
                    break
                if pl[i].string == pl[j].string and pl[i].pitch != pl[j].pitch:
                    # same string, overlapping, different pitch -> GP conflict
                    if abs(pl[i].onset - pl[j].onset) < 1e-3 or \
                       (pl[j].onset < pl[i].onset + pl[i].duration - 1e-3):
                        bad += 1
        self.assertEqual(bad, 0, "found same-string pitch collisions")


if __name__ == "__main__":
    unittest.main(verbosity=2)
