"""M9 — real-audio sanity: beat grid / chord timeline robustness.

Regression cover for the failure found when running an actual song through
the pipeline: the beat detector folded the tempo into range but never
recomputed the beat interval, so the grid stayed at the 20 ms histogram
floor. That produced ~375 "bars" and 127 chord segments over 30 s, and every
chord segment then spawned a full triad — a tab that is structurally valid
but musically unplayable (5-6 strings re-plucked every eighth note).
"""

import os
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import (BEAT_MAX_BPM, BEAT_MIN_BPM, Chord,
                      _enforce_min_duration, detect_beats)
from arrangement import (Arrangement, Role, RoleNote, all_notes,
                         cap_concurrent_plucks)
from models import Note


def rn(pitch, onset, role=Role.MELODY, dur=0.5, vel=100):
    return RoleNote(pitch=pitch, onset=onset, duration=dur,
                    velocity=vel, role=role)


def n(pitch: float, onset: float, dur: float = 0.2, vel: int = 80,
      inst: str = "") -> Note:
    return Note(pitch=pitch, onset=onset, duration=dur, velocity=vel,
                instrument=inst)


class TestBeatGridConsistency(unittest.TestCase):
    """The invariant the old code broke: grid spacing MUST equal 60/tempo."""

    def test_clean_120bpm_grid(self):
        notes = [n(60, i * 0.5) for i in range(16)]
        beats, tempo = detect_beats(notes)
        self.assertAlmostEqual(tempo, 120.0, places=0)
        self.assertGreater(len(beats), 4)
        spacing = beats[1] - beats[0]
        self.assertAlmostEqual(spacing, 60.0 / tempo, places=3)

    def test_grid_matches_tempo_on_dense_onsets(self):
        """Dense AMT output: a spurious ~30 ms pulse on top of a real pulse.

        The old estimator returned tempo=187 while keeping a 20 ms grid.
        Here we only assert the invariant + a musically sane tempo, since the
        "true" pulse of synthetic noise is not well defined.
        """
        noise = [round(i * 0.03, 4) for i in range(1000)]
        pulse = [round(i * 0.5, 4) for i in range(60)]
        onsets = sorted(set(noise) | set(pulse))
        notes = [n(60 + (i % 7), o) for i, o in enumerate(onsets)]

        beats, tempo = detect_beats(notes)
        self.assertGreaterEqual(tempo, BEAT_MIN_BPM)
        self.assertLessEqual(tempo, BEAT_MAX_BPM)

        spacing = beats[1] - beats[0]
        self.assertAlmostEqual(spacing, 60.0 / tempo, places=3,
                               msg="beat grid drifted away from reported tempo")

        # 30 s of audio can never legitimately contain 375 bars / 1468 beats
        self.assertLess(len(beats), 30.0 / (60.0 / BEAT_MAX_BPM) + 2)

    def test_tempo_folded_into_musical_range(self):
        """A 20 ms modal IOI must not leak out as a 3000 BPM tempo."""
        notes = [n(60, round(i * 0.02, 4)) for i in range(500)]
        _, tempo = detect_beats(notes)
        self.assertGreaterEqual(tempo, BEAT_MIN_BPM)
        self.assertLessEqual(tempo, BEAT_MAX_BPM)

    def test_two_notes_still_works(self):
        beats, tempo = detect_beats([n(60, 0.0), n(67, 2.3)])
        self.assertIsInstance(beats, list)
        self.assertGreater(tempo, 0)


class TestChordMinDuration(unittest.TestCase):
    def test_short_segments_absorbed(self):
        chords = [Chord("C", 0, "", 0.0, 0.2),
                  Chord("G", 7, "", 0.2, 0.4),
                  Chord("C", 0, "", 0.4, 4.0)]
        out = _enforce_min_duration(chords, 1.0)
        # the two flickering segments fold into the real one
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].label, "C")
        self.assertAlmostEqual(out[0].start, 0.0)
        self.assertAlmostEqual(out[0].end, 4.0)

    def test_long_segments_untouched(self):
        chords = [Chord("C", 0, "", 0.0, 2.0), Chord("G", 7, "", 2.0, 4.0)]
        out = _enforce_min_duration(chords, 1.0)
        self.assertEqual([c.label for c in out], ["C", "G"])

    def test_short_tail_folds_into_previous(self):
        chords = [Chord("C", 0, "", 0.0, 3.0), Chord("Am", 9, "m", 3.0, 3.2)]
        out = _enforce_min_duration(chords, 1.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].label, "C")
        self.assertAlmostEqual(out[0].end, 3.2)

    def test_single_chord_untouched(self):
        chords = [Chord("C", 0, "", 0.0, 0.1)]
        self.assertEqual(len(_enforce_min_duration(chords, 1.0)), 1)


class TestConcurrentPluckCap(unittest.TestCase):
    def test_six_string_wall_is_thinned(self):
        arr = Arrangement(style="fingerstyle")
        arr.melody = [rn(72, 1.0)]
        arr.harmony = [rn(60 + i, 1.0, Role.HARMONY) for i in range(6)]
        arr = cap_concurrent_plucks(arr)
        # 1 melody + up to 3 chord voices = the fingerstyle budget of 4
        self.assertEqual(len(all_notes(arr)), 4)

    def test_melody_survives_the_cap(self):
        arr = Arrangement(style="fingerstyle")
        arr.melody = [rn(72, 1.0, vel=40)]          # quiet melody
        arr.harmony = [rn(60 + i, 1.0, Role.HARMONY, vel=127) for i in range(5)]
        arr = cap_concurrent_plucks(arr)
        kept = all_notes(arr)
        self.assertEqual(len(kept), 4)
        # the melody is the line you must hear -> it wins over loud harmony
        self.assertIn(72, [x.pitch for x in kept])

    def test_near_simultaneous_notes_cluster(self):
        arr = Arrangement(style="fingerstyle")
        arr.harmony = [rn(60 + i, 1.0 + i * 0.01, Role.HARMONY) for i in range(5)]
        arr = cap_concurrent_plucks(arr)
        self.assertEqual(len(all_notes(arr)), 4)

    def test_cluster_width_is_bounded(self):
        """Notes 80 ms apart are two plucks, not one chord."""
        arr = Arrangement(style="fingerstyle")
        arr.harmony = [rn(60 + i, 1.0 + i * 0.08, Role.HARMONY) for i in range(5)]
        arr = cap_concurrent_plucks(arr)
        self.assertGreater(len(all_notes(arr)), 3)

    def test_separate_onsets_untouched(self):
        arr = Arrangement(style="fingerstyle")
        arr.melody = [rn(72, 0.5 + i) for i in range(5)]   # 1 s apart
        arr = cap_concurrent_plucks(arr)
        self.assertEqual(len(all_notes(arr)), 5)

    def test_dense_melody_does_not_starve_accompaniment(self):
        """The bug a flat global cap caused: 8 spurious melody onsets in one
        50 ms cluster ate the whole budget and deleted the bass + chord."""
        arr = Arrangement(style="fingerstyle")
        arr.melody = [rn(70 + (i % 5), 1.0 + i * 0.001, vel=120) for i in range(8)]
        arr.bass = [rn(45, 1.0, Role.BASS)]
        arr.harmony = [rn(60 + i, 1.0, Role.HARMONY) for i in range(3)]
        arr = cap_concurrent_plucks(arr)
        roles = Counter(x.role.value for x in all_notes(arr))
        self.assertEqual(roles.get("melody"), 1, "melody must be monophonic")
        self.assertGreaterEqual(roles.get("bass", 0), 1)
        self.assertGreaterEqual(roles.get("harmony", 0), 1)

    def test_strum_style_allows_more(self):
        arr = Arrangement(style="strum")
        arr.harmony = [rn(60 + i, 1.0, Role.HARMONY) for i in range(6)]
        arr = cap_concurrent_plucks(arr)
        self.assertEqual(len(all_notes(arr)), 6)


class TestTempoPrior(unittest.TestCase):
    def test_octave_ambiguity_resolved_toward_common_pulse(self):
        """Uniform dense onsets fit both 94 and 188 BPM equally well."""
        notes = [n(60, round(i * 0.02, 4)) for i in range(500)]
        _, tempo = detect_beats(notes)
        self.assertGreaterEqual(tempo, BEAT_MIN_BPM)
        self.assertLessEqual(tempo, BEAT_MAX_BPM)
        # must pick the musically plausible side, not the 2x one
        self.assertLess(tempo, 140.0)


if __name__ == "__main__":
    unittest.main()
