"""M3 — L5 fingerstyle arrangement engine tests.

Run:  python tests/test_m3.py
Exercises role assignment, voicing, the judgment layer (rules + LLM adapter),
and the end-to-end arrange() with real placements. No heavy deps required.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Note, PlacedNote
from analysis import AnalysisResult, Chord, Section
from arrangement import (
    Role, RoleNote, Arrangement, build_arrangement, assign_roles,
    voice, judge, Edit, LLMJudgmentLayer,
)
from arrange import arrange, OPEN_MIDI


def n(pitch, onset, dur=0.5, vel=100, inst=""):
    return Note(pitch=pitch, onset=onset, duration=dur, velocity=vel, instrument=inst)


def _sample_analysis() -> AnalysisResult:
    """A small I-V-IV-I analysis like the demo produces."""
    chords = [
        Chord("C", 0, "", 0.0, 2.0, 1.0),
        Chord("G", 7, "", 2.0, 4.0, 1.0),
        Chord("F", 5, "", 4.0, 6.0, 1.0),
        Chord("C", 0, "", 6.0, 8.0, 1.0),
    ]
    beats = [i * 0.5 for i in range(17)]
    return AnalysisResult(chords=chords, beats=beats, tempo=120.0,
                          key="C major", sections=[Section("A", 0.0, 8.0, 8)],
                          backend="note-based")


class TestRoleAssignment(unittest.TestCase):
    def test_monophonic_derives_bass_harmony(self):
        # a bare melody -> bass + harmony derived from the chord timeline
        notes = [n(60, 0.0), n(64, 0.5), n(67, 1.0), n(65, 2.0)]
        ana = _sample_analysis()
        arr = assign_roles(notes, ana)
        self.assertEqual(len(arr.melody), 4)
        self.assertGreater(len(arr.bass), 0, "bass should be derived from chords")
        self.assertGreater(len(arr.harmony), 0, "harmony should be derived from chords")
        self.assertTrue(any("derived:" in h.source for h in arr.harmony))

    def test_polyphonic_uses_stems_directly(self):
        # bass + other stems present -> use them, do NOT over-derive
        notes = [
            n(40, 0.0, 1.0, 100, "bass"),
            n(64, 0.0, 0.5, 100, "other"),
            n(67, 0.5, 0.5, 100, "other"),
            n(72, 1.0, 1.0, 100, "vocals"),
        ]
        ana = _sample_analysis()
        arr = assign_roles(notes, ana)
        self.assertEqual(len(arr.bass), 1)
        self.assertEqual(len(arr.harmony), 2)
        self.assertEqual(len(arr.melody), 1)
        # no derived notes when stems already carry the roles
        self.assertFalse(any("derived:" in h.source for h in arr.harmony))

    def test_no_analysis_no_derivation(self):
        notes = [n(60, i * 0.5) for i in range(5)]
        arr = assign_roles(notes, None)
        self.assertEqual(len(arr.melody), 5)
        self.assertEqual(len(arr.bass), 0)
        self.assertEqual(len(arr.harmony), 0)


class TestVoicing(unittest.TestCase):
    def test_voice_leading_produces_valid_pitches(self):
        ana = _sample_analysis()
        notes = [n(60, i * 0.5) for i in range(8)]
        arr = assign_roles(notes, ana)
        arr = voice(arr)
        # every harmony note must sit in the mid register we targeted
        for h in arr.harmony:
            self.assertGreaterEqual(h.pitch, 50)
            self.assertLessEqual(h.pitch, 79)


class TestJudgment(unittest.TestCase):
    def test_rules_light_density_thins_harmony(self):
        ana = _sample_analysis()
        notes = [n(60, i * 0.5) for i in range(8)]
        arr = build_arrangement(notes, ana, instructions="make it simpler",
                                judge_backend="rules")
        # light density should have dropped some harmony chords
        self.assertEqual(arr.density, "light")
        # after jud_apply, harmony count reduced vs full
        full = build_arrangement(notes, ana, instructions="", judge_backend="rules")
        self.assertLessEqual(len(arr.harmony), len(full.harmony))

    def test_rules_style_parsing(self):
        ana = _sample_analysis()
        notes = [n(60, i * 0.5) for i in range(4)]
        jazz = build_arrangement(notes, ana, instructions="jazz voicing",
                                 judge_backend="rules")
        self.assertEqual(jazz.style, "jazz")
        folk = build_arrangement(notes, ana, instructions="民谣",
                                 judge_backend="rules")
        self.assertEqual(folk.style, "folk")

    def test_llm_backend_requires_client(self):
        ana = _sample_analysis()
        notes = [n(60, i * 0.5) for i in range(4)]
        arr = build_arrangement(notes, ana, judge_backend="rules")
        layer = LLMJudgmentLayer()  # no llm_fn injected
        with self.assertRaises(RuntimeError):
            layer.review(arr, "make it pretty")
        # auto falls back to rules (no error)
        arr2 = judge(arr, "make it pretty", backend="auto")
        self.assertIsInstance(arr2, Arrangement)

    def test_llm_adapter_with_injected_fn(self):
        ana = _sample_analysis()
        notes = [n(60, i * 0.5) for i in range(4)]
        arr = build_arrangement(notes, ana, judge_backend="rules")
        captured = {}

        def fake_llm(prompt: str) -> list:
            captured["prompt"] = prompt
            return [Edit("set_style", "all", "jazz", "llm says jazz")]

        layer = LLMJudgmentLayer(llm_fn=fake_llm)
        edits = layer.review(arr, "jazz it up")
        self.assertTrue(edits)
        self.assertIn("jazz", captured["prompt"])


class TestArrangeEndToEnd(unittest.TestCase):
    def test_full_arrangement_places_all_roles(self):
        from tests.make_sample import sample_notes
        notes = sample_notes()
        ana = _sample_analysis()
        placed = arrange(notes, analysis=ana, instructions="", style="fingerstyle")

        # every placed note must sound its intended pitch on a valid position
        for p in placed:
            self.assertIn(p.string, range(1, 7))
            self.assertIn(p.fret, range(0, 20))
            self.assertEqual(OPEN_MIDI[p.string] + p.fret, p.pitch,
                             f"position mismatch: {p}")
            self.assertIn(p.role, ("melody", "bass", "harmony"))

        roles = {p.role for p in placed}
        self.assertIn("melody", roles)
        self.assertIn("bass", roles)
        self.assertIn("harmony", roles)

    def test_bass_sits_low(self):
        notes = [n(60, i * 0.5) for i in range(4)]
        ana = _sample_analysis()
        placed = arrange(notes, analysis=ana)
        bass = [p for p in placed if p.role == "bass"]
        self.assertTrue(bass)
        for b in bass:
            self.assertGreaterEqual(b.string, 4, "bass should use low strings")

    def test_backward_compat_no_analysis(self):
        # old path: arrange(notes) with no analysis -> melody only, no loss
        notes = [n(p, i * 0.5) for i, p in enumerate(
            [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65, 64, 62, 60])]
        placed = arrange(notes)
        self.assertEqual(len(placed), len(notes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
