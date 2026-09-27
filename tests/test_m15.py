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

    def test_same_span_shares_a_voice_as_a_chord(self):
        # two strings struck together and released together are one chord in
        # one voice -- not two voices, which is what shredded the page layout
        root = _root([pn(0.0, 1.0, 6, 0, "bass"), pn(0.0, 1.0, 1, 3, "melody")])
        notes = [n for n in root.findall(".//note") if n.find("rest") is None]
        self.assertEqual(len(notes), 2)
        self.assertEqual({n.find("voice").text for n in notes}, {"1"})
        self.assertIsNone(notes[0].find("chord"))
        self.assertIsNotNone(notes[1].find("chord"))
        # <chord> has to be the note's first child, before <pitch>
        self.assertEqual([c.tag for c in notes[1]][0], "chord")

    def test_diverging_timelines_open_a_second_voice(self):
        # a bass ringing 2 beats while the melody moves underneath cannot share
        # a voice (that is what needed <backup>), and must not be cut short
        placed = [pn(0.0, 2.0, 6, 0, "bass"), pn(1.0, 0.5, 1, 0, "melody")]
        root = _root(placed, tempo=120.0)
        self.assertIsNone(root.find(".//backup"))
        self.assertIsNone(root.find(".//forward"))
        voiced = {}
        for n in root.find("./part/measure").findall("note"):
            voiced.setdefault(n.find("voice").text, []).append(
                "rest" if n.find("rest") is not None else n.find("pitch"))
        self.assertEqual(len(voiced), 2, "expected exactly two voices")
        self.assertTrue(all(v for v in voiced.values()))

    def test_ringing_note_needs_no_backup(self):
        # bass rings 2 beats while the melody enters one beat later. Written
        # as one voice this needs <backup>, which alphaTab refuses to read
        # ("cannot fill new beats into already filled area of voice"); as one
        # voice per string both timelines are linear.
        placed = [pn(0.0, 2.0, 6, 0, "bass"), pn(1.0, 1.0, 1, 0, "melody")]
        root = _root(placed)
        self.assertIsNone(root.find(".//backup"))
        self.assertIsNone(root.find(".//forward"))

    def test_every_voice_fills_its_measure_exactly(self):
        # readers key layout off voice coverage, so each voice in each measure
        # must add up to the bar length and never overlap itself
        placed = [pn(0.0, 2.0, 6, 0, "bass"), pn(1.0, 0.5, 3, 5, "harmony"),
                  pn(1.6, 0.5, 1, 0, "melody"), pn(6.0, 3.0, 5, 3, "bass")]
        root = _root(placed, tempo=120.0)
        filled = 0
        for measure in root.findall("./part/measure"):
            per_voice = {}
            for n in measure.findall("note"):
                per_voice.setdefault(n.find("voice").text, 0)
                per_voice[n.find("voice").text] += int(n.find("duration").text)
            self.assertTrue(per_voice, f"measure {measure.get('number')} empty")
            for voice, total in per_voice.items():
                self.assertEqual(total, 8,
                                 f"m{measure.get('number')} v{voice} = {total}")
            filled += 1
        self.assertEqual(filled, 5)

    def test_empty_measures_get_a_single_rest(self):
        # a bar with nothing in it must not emit an empty voice per string
        placed = [pn(0.0, 0.5, 1, 0), pn(6.0, 0.5, 4, 2)]
        root = _root(placed, tempo=120.0)
        bar2 = root.findall("./part/measure")[1]
        self.assertEqual(len(bar2.findall("note")), 1)
        self.assertIsNotNone(bar2.find("note/rest"))

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

    def test_duration_and_type_always_agree(self):
        # <duration> is the timeline but readers lay out on <type>, so a pair
        # that disagrees (5 eighths written as a half) shifts everything after
        # it in that voice. Only 1/2/3/4/6/8 slots exist as one value, and
        # ties are what bridges the rest.
        expected = {1: ("eighth", 0), 2: ("quarter", 0), 3: ("quarter", 1),
                    4: ("half", 0), 6: ("half", 1), 8: ("whole", 0)}
        quarters = {"whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5}
        placed = [pn(0.0, 5 * 0.5, 6, 0, "bass"), pn(3.2, 0.5, 1, 0, "melody")]
        root = _root(placed, tempo=120.0)
        for n in root.findall(".//note"):
            dur = int(n.find("duration").text)
            tname = n.find("type").text
            dots = len(n.findall("dot"))
            self.assertEqual((tname, dots), expected[dur])
            value = quarters[tname] * (1.5 if dots else 1.0)
            self.assertEqual(dur / 2.0, value,
                             f"duration {dur} slots != type {tname}{'.'*dots}")

    def test_note_ringing_over_the_barline_is_tied_not_cut(self):
        # cutting at the barline re-articulates a note that sustains. At 90 BPM
        # one slot is 1/3 s, so a 12-slot note runs 4 slots into bar 2.
        placed = [pn(0.0, 4.0, 6, 3, "bass")]
        root = _root(placed, tempo=90.0)
        measures = root.findall("./part/measure")
        self.assertGreater(len(measures), 1)
        tied = [t.get("type") for t in root.findall(".//tie")]
        self.assertEqual(tied, ["start", "stop"])
        self.assertEqual([int(n.find("duration").text)
                          for n in measures[1].findall("note")
                          if n.find("rest") is None], [4])
        # <tie> must sit between <duration> and <voice>, and <tied> must be
        # the first child of <notations> -- both are DTD order requirements
        for n in root.findall(".//note"):
            kids = [c.tag for c in n]
            if "tie" in kids:
                self.assertEqual(kids.index("tie"), kids.index("duration") + 1)
            notations = n.find("notations")
            if notations is not None:
                self.assertEqual([c.tag for c in notations][:2],
                                 ["tied", "technical"]
                                 if n.find("tie") is not None else
                                 ["technical"])

    def test_arranged_pipeline_writes_musicxml(self):
        # regression: --musicxml was silently dropped on the audio/MIDI path
        # (only import mode honoured it), so the documented quickstart
        # produced no file at all.
        from pipeline import run
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "score.musicxml")
            run(demo=True, musicxml_path=out, emit_midi=False, emit_wav=False)
            self.assertTrue(os.path.getsize(out) > 0)
            root = ET.parse(out).getroot()
            self.assertEqual(root.tag, "score-partwise")
            self.assertGreater(len(root.findall("./part/measure")), 1)
            self.assertEqual(root.find(".//work-title").text, "Resonote")

    def test_rejects_bad_tuning(self):
        with self.assertRaises(ValueError):
            _root([pn(0.0, 1.0, 1, 0)], tuning=[64, 59])


if __name__ == "__main__":
    unittest.main()
