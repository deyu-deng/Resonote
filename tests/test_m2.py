"""M2 — L4 music-theory analysis layer tests.

Run:  python tests/test_m2.py
These exercise the dependency-free rule engine so they pass in a thin env.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Note
from analysis import (
    analyze, detect_key, detect_beats, estimate_chord,
    madmom_available, _detect_sections, _build_bars,
)


def n(pitch, onset, dur=0.5, vel=100):
    return Note(pitch=pitch, onset=onset, duration=dur, velocity=vel)


class TestChordEstimation(unittest.TestCase):
    def test_major_triad(self):
        label, root, quality, conf = estimate_chord({0, 4, 7}, bass_pc=0)
        self.assertEqual(label, "C")
        self.assertEqual(root, 0)
        self.assertAlmostEqual(conf, 1.0, places=1)

    def test_minor_triad(self):
        # A minor: A=9, C=0, E=4
        label, root, quality, conf = estimate_chord({9, 0, 4}, bass_pc=9)
        self.assertEqual(label, "Am")
        self.assertEqual(quality, "m")

    def test_seventh_chord(self):
        # G7: G=7, B=11, D=2, F=5
        label, root, quality, conf = estimate_chord({7, 11, 2, 5}, bass_pc=7)
        self.assertEqual(label, "G7")

    def test_implied_tone_low_confidence(self):
        # single note -> implied tone, not a confirmed chord
        label, root, quality, conf = estimate_chord({0}, bass_pc=0)
        self.assertEqual(label, "C")
        self.assertEqual(quality, "")
        self.assertLess(conf, 0.5)

    def test_empty_is_no_chord(self):
        self.assertEqual(estimate_chord(set()), ("N", 0, "", 0.0))


class TestKeyDetection(unittest.TestCase):
    def test_c_major(self):
        notes = [n(p, o) for o, p in enumerate(
            [60, 64, 67, 62, 65, 69, 64, 67, 72, 60, 64, 67])]
        self.assertEqual(detect_key(notes), "C major")

    def test_a_minor(self):
        # A C E + scale tones, no strong C major pull
        notes = [n(p, o * 0.5) for o, p in enumerate(
            [69, 72, 76, 71, 74, 77, 69, 72, 76])]
        key = detect_key(notes)
        self.assertTrue(key.endswith("minor"), f"expected minor key, got {key}")


class TestBeatDetection(unittest.TestCase):
    def test_regular_grid_120bpm(self):
        onsets = [i * 0.5 for i in range(8)]
        notes = [n(60, o) for o in onsets]
        beats, tempo = detect_beats(notes)
        self.assertAlmostEqual(tempo, 120.0, places=0)
        self.assertGreater(len(beats), 4)

    def test_sparse_returns_something(self):
        notes = [n(60, 0.0), n(67, 2.3)]
        beats, tempo = detect_beats(notes)
        self.assertIsInstance(beats, list)
        self.assertGreater(tempo, 0)


class TestSectionDetection(unittest.TestCase):
    def test_split_by_rest(self):
        # bar0 = C, bar1 = SILENT, bar2 = G  -> two distinct sections
        # (4 beats/bar @ 120bpm => each bar spans 2.0s)
        notes = [n(60, 0.0), n(64, 0.5), n(67, 1.0),   # C region, bar0 [0,2)
                 n(67, 4.5), n(71, 5.0)]               # G region, bar2 [4,6)
        beats, _ = detect_beats(notes)
        bars = _build_bars(beats, 6.0)
        # rebuild per-bar labels like analyze() does
        from analysis import estimate_chord, _note_pcs, _pc
        bar_labels = []
        for b0, b1 in bars:
            bn = [x for x in notes if b0 - 1e-6 <= x.onset < b1]
            bar_labels.append(estimate_chord(_note_pcs(bn),
                              _pc(min(bn, key=lambda x: x.pitch).pitch))[0]
                              if bn else None)
        sections = _detect_sections(bar_labels, bars)
        self.assertGreaterEqual(len(sections), 2)
        labels = [s.label for s in sections]
        self.assertIn("A", labels)
        self.assertIn("B", labels)


class TestMonophonicCap(unittest.TestCase):
    def test_monophonic_arpeggio_caps_extension(self):
        # sequential notes outlining Dm9 (D F A C E) in one line
        notes = [n(62, 0.0), n(65, 0.5), n(69, 1.0), n(72, 1.5), n(76, 2.0)]
        res = analyze(notes)
        for c in res.chords:
            self.assertNotIn("9", c.label)   # no 9th chords from a single line
        self.assertTrue(any(c.label.startswith("Dm") or c.label == "Dm"
                            for c in res.chords))

    def test_polyphonic_stack_keeps_seventh(self):
        # simultaneous G7 stack -> the 7th must survive
        notes = [n(67, 0.0, 2.0), n(71, 0.0, 2.0),
                 n(74, 0.0, 2.0), n(77, 0.0, 2.0)]
        res = analyze(notes)
        self.assertTrue(any(c.label == "G7" for c in res.chords))


class TestAnalyzeE2E(unittest.TestCase):
    def test_demo_notes(self):
        from tests.make_sample import sample_notes
        notes = sample_notes()
        res = analyze(notes)
        self.assertEqual(res.backend, "note-based")
        self.assertGreater(res.tempo, 0)
        self.assertTrue(res.key)                      # non-empty key guess
        self.assertGreaterEqual(len(res.chords), 1)   # at least one segment
        self.assertGreaterEqual(len(res.sections), 1)

    def test_madmom_path_requires_package(self):
        notes = [n(60, 0.0), n(64, 0.5)]
        if madmom_available():
            print("SKIP: madmom installed — cannot assert missing-package error")
            return
        with self.assertRaises(RuntimeError):
            analyze(notes, audio=[0.0, 0.0], sr=22050, backend="madmom")


if __name__ == "__main__":
    unittest.main(verbosity=2)
