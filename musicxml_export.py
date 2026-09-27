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
a note's duration is how long it rings (capped at the next attack on the same
string), split across barlines and note values with ties as needed.

Voice model: one MusicXML voice per string, because a string cannot sound two
notes at once. Simultaneous notes therefore land in different voices meeting on
the same beat, which is how tab is laid out, and it keeps every voice a linear
timeline that needs no <backup> to interleave.
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
# slot length -> notated type + dots. Only these lengths exist as a single
# note; 5 and 7 do not, which is why ringing lengths are split into tie
# chains by _split().
_TYPES = {1: ("eighth", 0), 2: ("quarter", 0), 3: ("quarter", 1),
          4: ("half", 0), 6: ("half", 1), 8: ("whole", 0)}
_CLEAN = sorted(_TYPES, reverse=True)


def _segments(slot: int, slots: int):
    """Yield (start, length) covering [slot, slot+slots), where every piece
    stays inside one bar and every piece has a real note type.

    <duration> and <type> have to agree: write duration=5 against type=half
    and a reader lays the voice out on the type, so everything after it in
    that voice drifts by a slot. Five eighths is a half tied to an eighth.
    """
    remaining, cur = slots, slot
    while remaining > 0:
        room = SLOTS_PER_MEASURE - (cur % SLOTS_PER_MEASURE)
        take = next((c for c in _CLEAN if c <= remaining and c <= room), 1)
        yield cur, take
        cur += take
        remaining -= take


def _pitch_el(parent, midi: int):
    p = SubElement(parent, "pitch")
    name = _STEP[midi % 12]
    SubElement(p, "step").text = name[0]
    if len(name) > 1:                       # sharp
        SubElement(p, "alter").text = "1"
    SubElement(p, "octave").text = str(midi // 12 - 1)
    return p


def _note_el(measure, midi: int, dur: int, string: int, fret: int,
             voice: str = "1", ties: Sequence[str] = ()):
    n = SubElement(measure, "note")
    _pitch_el(n, midi)
    SubElement(n, "duration").text = str(dur)
    for tie in ties:                    # <tie> sits between duration and voice
        SubElement(n, "tie", type=tie)
    SubElement(n, "voice").text = voice
    tname, dots = _TYPES[dur]
    SubElement(n, "type").text = tname
    for _ in range(dots):
        SubElement(n, "dot")
    notations = SubElement(n, "notations")
    for tie in ties:                    # <tied> is what the tab staff draws
        SubElement(notations, "tied", type=tie)
    tech = SubElement(notations, "technical")
    SubElement(tech, "fret").text = str(fret)      # fret BEFORE string
    SubElement(tech, "string").text = str(string)
    return n


def _rest_el(measure, dur: int, voice: str = "1"):
    n = SubElement(measure, "note")
    SubElement(n, "rest")
    SubElement(n, "duration").text = str(dur)
    SubElement(n, "voice").text = voice
    tname, dots = _TYPES[dur]
    SubElement(n, "type").text = tname
    for _ in range(dots):
        SubElement(n, "dot")
    return n


def _rest_run(measure, start_slot: int, slots: int, voice: str):
    """Silence, split the same way notes are — a 5-eighth gap is not one rest
    any more than it is one note."""
    for g, dur in _segments(start_slot, slots):
        _rest_el(measure, dur, voice=voice)


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

    # One MusicXML voice per string. A string cannot sound twice at once, so
    # each string is a strictly linear timeline of attacks and silence -- which
    # is also how tablature is laid out by hand.
    #
    # The alternative (everything in voice 1, interleaved with <backup>) is
    # what this exporter used to do, and alphaTab rejects it outright with
    # "Unsupported forward/backup detected. Cannot fill new beats into already
    # filled area of voice" on every ringing-note-over-a-new-attack slot. The
    # point of this file is to be read by other people's engravers, so the
    # voice layout has to be one they can follow.
    events_by_string: dict = {}
    for (s, length, p) in items:
        events_by_string.setdefault(p.string, {})[s] = (p, length)

    # Split every ringing note into writable note values. A note that crosses
    # the barline continues as a tie -- cutting it at the barline instead would
    # re-articulate something that is supposed to sustain.
    timeline: dict = {}
    for string, events in events_by_string.items():
        segments = []
        for s in sorted(events):
            p, length = events[s]
            pieces = list(_segments(s, length))
            for i, (g, dur) in enumerate(pieces):
                ties = (["stop"] if i else []) + \
                       (["start"] if i < len(pieces) - 1 else [])
                segments.append((g, dur, p, ties))
        timeline[string] = segments

    for mi in range(n_measures):
        m = m1 if mi == 0 else SubElement(part, "measure", number=str(mi + 1))
        bar_start = mi * SLOTS_PER_MEASURE
        bar_end = bar_start + SLOTS_PER_MEASURE
        active = [st for st in sorted(timeline)
                  if any(bar_start <= g < bar_end for g, _, _, _ in timeline[st])]
        if not active:
            _rest_el(m, SLOTS_PER_MEASURE, voice="1")
            continue
        for string in active:
            # voice number = string number, stable across the whole part.
            # Numbering per measure instead (1,2,3.. over whichever strings are
            # active) splits a tie chain across two voices the moment the
            # active set changes, and readers then place the continuation at
            # the wrong offset.
            v = str(string)
            cursor = 0
            for g, dur, p, ties in timeline[string]:
                if not bar_start <= g < bar_end:
                    continue
                if g - bar_start > cursor:
                    _rest_run(m, bar_start + cursor, g - bar_start - cursor, v)
                _note_el(m, p.pitch, dur, string, p.fret, voice=v, ties=ties)
                cursor = g - bar_start + dur
            if cursor < SLOTS_PER_MEASURE:
                _rest_run(m, bar_start + cursor, SLOTS_PER_MEASURE - cursor, v)

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
