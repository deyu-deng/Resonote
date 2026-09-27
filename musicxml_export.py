"""Export placed notes to MusicXML 4.0 tablature — the standard interchange
format that MuseScore, Finale, Sibelius, Guitar Pro 7, alphaTab and OSMD all
read.

Why this exists: the hand-written PDF is a fallback for machines with no
notation software. MusicXML hands the arrangement to engravers whose whole
job is making scores look right, and gives the user a file they can open and
edit anywhere. Syntax follows the MusicXML tablature tutorial
(musicxml.com/tutorial/tablature): fret/string live in
notations/technical (fret first), tunings in attributes/staff-details with
line 1 = lowest string, clef sign TAB.

Timing model: notes are quantized to the eighth-note grid the exporter uses;
a note's duration is how long it rings (capped at the next attack on the
same string and at the barline — cross-bar ties are not written yet).
"""
import math
from typing import List, Optional, Sequence
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from models import PlacedNote

SLOTS_PER_MEASURE = 8          # 4/4, one slot = one eighth note
DIVISIONS = 2                  # divisions per quarter -> one slot = 1 division
STANDARD_TUNING = [64, 59, 55, 50, 45, 40]   # string 1..6, MIDI

_STEP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# how long a note rings, in slots -> notated type + dots (display only)
_TYPES = {1: ("eighth", 0), 2: ("quarter", 0), 3: ("quarter", 1),
          4: ("half", 0), 5: ("quarter", 1), 6: ("half", 1),
          7: ("half", 1), 8: ("whole", 0)}


def _pitch_el(parent, midi: int):
    p = SubElement(parent, "pitch")
    name = _STEP[midi % 12]
    SubElement(p, "step").text = name[0]
    if len(name) > 1:                       # sharp
        SubElement(p, "alter").text = "1"
    SubElement(p, "octave").text = str(midi // 12 - 1)
    return p


def _note_el(measure, midi: int, dur: int, string: int, fret: int,
             chord: bool, voice: str = "1"):
    n = SubElement(measure, "note")
    if chord:
        SubElement(n, "chord")
    _pitch_el(n, midi)
    SubElement(n, "duration").text = str(dur)
    SubElement(n, "voice").text = voice
    tname, dots = _TYPES.get(dur, ("quarter", 1))
    SubElement(n, "type").text = tname
    for _ in range(dots):
        SubElement(n, "dot")
    notations = SubElement(n, "notations")
    tech = SubElement(notations, "technical")
    SubElement(tech, "fret").text = str(fret)      # fret BEFORE string
    SubElement(tech, "string").text = str(string)
    return n


def _rest_el(measure, dur: int, voice: str = "1"):
    n = SubElement(measure, "note")
    SubElement(n, "rest")
    SubElement(n, "duration").text = str(dur)
    SubElement(n, "voice").text = voice
    tname, dots = _TYPES.get(dur, ("quarter", 1))
    SubElement(n, "type").text = tname
    for _ in range(dots):
        SubElement(n, "dot")
    return n


def _tuning_name(midi: int):
    """MIDI pitch -> (step, alter, octave) for staff-tuning."""
    name = _STEP[midi % 12]
    step, alter = name[0], ("1" if len(name) > 1 else None)
    return step, alter, midi // 12 - 1


def build_musicxml(placed: Sequence[PlacedNote], tempo: float = 120.0,
                   title: str = "Resonote",
                   tuning: Optional[Sequence[int]] = None) -> str:
    tuning = list(tuning) if tuning else list(STANDARD_TUNING)
    if len(tuning) != 6:
        raise ValueError(f"tuning must list 6 pitches, got {len(tuning)}")

    if placed:
        anchor = min(p.onset for p in placed)
    else:
        anchor = 0.0
    slots_per_sec = tempo / 30.0                  # eighth notes per second

    # (start_slot, length, note) -- length capped at the next attack on the
    # same string, exactly like the GP5 exporter
    items: List = []
    by_string: dict = {}
    for p in placed:
        s = max(0, int(round((p.onset - anchor) * slots_per_sec)))
        by_string.setdefault(p.string, []).append((s, p))
    for string, notes in by_string.items():
        notes.sort(key=lambda x: x[0])
        for i, (s, p) in enumerate(notes):
            length = max(1, int(round(p.duration * slots_per_sec)))
            if i + 1 < len(notes):
                length = min(length, notes[i + 1][0] - s)
            items.append((s, length, p))
    items.sort(key=lambda x: (x[0], x[2].string))

    total = max((s + l for s, l, _ in items), default=0)
    total = max(total, SLOTS_PER_MEASURE)
    n_measures = ((total + SLOTS_PER_MEASURE - 1) // SLOTS_PER_MEASURE)

    score = Element("score-partwise", version="4.0")
    work = SubElement(score, "work")
    SubElement(work, "work-title").text = title
    ident = SubElement(score, "identification")
    enc = SubElement(ident, "encoding")
    SubElement(enc, "software").text = "Resonote"

    plist = SubElement(score, "part-list")
    spart = SubElement(plist, "score-part", id="P1")
    SubElement(spart, "part-name").text = "Guitar"

    part = SubElement(score, "part", id="P1")

    # measure 1 attributes: divisions, key, time, TAB clef, string tunings
    m1 = SubElement(part, "measure", number="1")
    attrs = SubElement(m1, "attributes")
    SubElement(attrs, "divisions").text = str(DIVISIONS)
    key = SubElement(attrs, "key")
    SubElement(key, "fifths").text = "0"
    time = SubElement(attrs, "time")
    SubElement(time, "beats").text = "4"
    SubElement(time, "beat-type").text = "4"
    clef = SubElement(attrs, "clef")
    SubElement(clef, "sign").text = "TAB"
    SubElement(clef, "line").text = "5"
    details = SubElement(attrs, "staff-details")
    SubElement(details, "staff-lines").text = "6"
    # MusicXML staff line 1 = LOWEST string -> our string 6
    for line in range(1, 7):
        midi = tuning[6 - line]
        st = SubElement(details, "staff-tuning", line=str(line))
        step, alter, octv = _tuning_name(midi)
        SubElement(st, "tuning-step").text = step
        if alter:
            SubElement(st, "tuning-alter").text = alter
        SubElement(st, "tuning-octave").text = str(octv)
    direction = SubElement(m1, "direction", placement="above")
    dtype = SubElement(direction, "direction-type")
    metro = SubElement(dtype, "metronome")
    SubElement(metro, "beat-unit").text = "quarter"
    SubElement(metro, "per-minute").text = str(int(round(tempo)))

    # per-measure events
    attacks: dict = {}
    for (s, length, p) in items:
        attacks.setdefault(s, []).append((p, length))
    n_slots = n_measures * SLOTS_PER_MEASURE

    for mi in range(n_measures):
        m = m1 if mi == 0 else SubElement(part, "measure", number=str(mi + 1))
        cursor = 0                                   # in divisions
        for slot in range(SLOTS_PER_MEASURE):
            g = mi * SLOTS_PER_MEASURE + slot
            group = attacks.get(g)
            t = slot * 1                             # divisions from bar start
            if not group:
                continue
            if t > cursor:
                _rest_el(m, t - cursor)
                cursor = t
            if t < cursor:                           # ring crosses the event
                backup = SubElement(m, "backup")
                SubElement(backup, "duration").text = str(cursor - t)
                cursor = t
            dur0 = None
            for j, (p, length) in enumerate(group):
                # never let a note cross the barline (no ties written yet)
                room = n_slots - g
                dur = max(1, min(length, room))
                if dur0 is None:
                    dur0 = dur
                    _note_el(m, p.pitch, dur, p.string, p.fret, chord=False)
                else:
                    _note_el(m, p.pitch, dur, p.string, p.fret, chord=True)
            cursor = t + (dur0 or 0)
        if cursor < SLOTS_PER_MEASURE:
            _rest_el(m, SLOTS_PER_MEASURE - cursor)

    rough = tostring(score, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    lines = [ln for ln in pretty.splitlines() if ln.strip()]
    if lines[0].startswith("<?xml"):       # minidom adds its own declaration
        lines = lines[1:]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + "\n".join(lines) + "\n")


def write_musicxml(placed, out_path, tempo: float = 120.0,
                   title: str = "Resonote",
                   tuning: Optional[Sequence[int]] = None) -> str:
    xml = build_musicxml(placed, tempo=tempo, title=title, tuning=tuning)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_path
