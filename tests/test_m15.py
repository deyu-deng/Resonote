"""M15 — MusicXML export (standard interchange, not a self-invented format).

The exporter hands the arrangement to real engravers (MuseScore, Finale,
Guitar Pro 7, alphaTab, OSMD). Syntax follows the MusicXML tablature
tutorial: fret BEFORE string inside notations/technical, tunings in
attributes/staff-details with line 1 = lowest string, clef sign TAB.
These tests pin the parts that break interop if wrong.
"""

import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PlacedNote
from arrange import OPEN_MIDI
from musicxml_export import build_musicxml


def pn(onset, dur, string, fret, role="melody"):
    return PlacedNote(pitch=OPEN_MIDI[string] + fret, onset=onset,
                      duration=dur, string=string, fret=fret,
                      finger=0, pluck="", role=role)


def _root(placed, **kw):
    xml = build_musicxml(placed, **kw)
    return ET.fromstring(xml)


class TestMusicXmlExport(unittest.TestCase):
    def test_wellformed_and_structure(self):
        placed = [pn(0.0, 1.0, 1, 0)]
        root = _root(placed, tempo=90.0)
        self.assertEqual(root.tag, "score-partwise")
        self.assertEqual(root.get("version"), "4.0")
        self.assertEqual(len(root.findall("./part/measure")), 1)

    def test_tab_clef_and_tunings(self):
        # DADGAD: line 1 = LOWEST string (D2), line 6 = highest (D4)
        tuning = [62, 57, 55, 50, 45, 38]
        root = _root([pn(0.0, 1.0, 1, 0)], tuning=tuning)
        attrs = root.find("./part/measure/attributes")
        self.assertEqual(attrs.find("clef/sign").text, "TAB")
        self.assertEqual(attrs.find("staff-details/staff-lines").text, "6")
        lines = {int(st.get("line")): (st.find("tuning-step").text,
                                       st.find("tuning-octave").text)
                 for st in attrs.findall("staff-details/staff-tuning")}
        self.assertEqual(lines[1], ("D", "2"))       # lowest string first
        self.assertEqual(lines[6], ("D", "4"))

    def test_fret_comes_before_string(self):
        root = _root([pn(0.0, 1.0, 6, 3)])
        tech = root.find(".//technical")
        self.assertEqual([e.tag for e in tech], ["fret", "string"])
        self.assertEqual(tech.find("fret").text, "3")
        self.assertEqual(tech.find("string").text, "6")

    def test_pitch_matches_string_and_fret(self):
        root = _root([pn(0.0, 1.0, 5, 7)])
        pitch = root.find(".//note/pitch")
        step = pitch.find("step").text
        alter_el = pitch.find("alter")
        alter = int(alter_el.text) if alter_el is not None else 0
        octv = int(pitch.find("octave").text)
        midi = [0, 2, 4, 5, 7, 9, 11]["CDEFGAB".index(step)] + alter + 12 * (octv + 1)
        self.assertEqual(midi, OPEN_MIDI[5] + 7)

    def test_simultaneous_notes_form_a_chord(self):
        root = _root([pn(0.0, 1.0, 6, 0, "bass"), pn(0.0, 1.0, 1, 3, "melody")])
        notes = [n for n in root.findall(".//note") if n.find("rest") is None]
        self.assertEqual(len(notes), 2)
        self.assertIsNone(notes[0].find("chord"))
        self.assertIsNotNone(notes[1].find("chord"))

    def test_overlapping_ring_emits_backup(self):
        # bass rings 2 beats, another note enters 1 beat later -> the second
        # event needs <backup> to rewind the voice cursor
        root = _root([pn(0.0, 2.0, 6, 0, "bass"), pn(1.0, 1.0, 1, 0, "melody")])
        self.assertIsNotNone(root.find(".//backup"))

    def test_measures_fill_with_rests(self):
        # one note in measure 1, nothing else -> the rest of the bar is rests
        root = _root([pn(0.0, 0.5, 1, 0)])
        measure = root.find("./part/measure")
        durs = [int(n.find("duration").text) for n in measure.findall("note")]
        self.assertEqual(sum(durs), 8)               # 4/4 at divisions=2 -> 8

    def test_tempo_written(self):
        root = _root([pn(0.0, 1.0, 1, 0)], tempo=93.0)
        metro = root.find(".//metronome")
        self.assertEqual(metro.find("per-minute").text, "93")

    def test_measures_count(self):
        # 120 BPM -> one bar every 2 s; a note at 4.0 s lands in bar 3
        placed = [pn(0.0, 0.5, 1, 0), pn(4.0, 0.5, 1, 0)]
        root = _root(placed)
        self.assertEqual(len(root.findall("./part/measure")), 3)

    def test_rejects_bad_tuning(self):
        with self.assertRaises(ValueError):
            _root([pn(0.0, 1.0, 1, 0)], tuning=[64, 59])


if __name__ == "__main__":
    unittest.main()
