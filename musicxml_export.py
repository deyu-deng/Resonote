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

Voice model: the fewest voices that need no <backup>. Notes starting and
stopping together share a voice as a <chord>; only genuinely independent
timelines open another voice. One voice per string reads fine but spreads a
fingerstyle arrangement over far too many pages, and one voice with <backup>
is refused by alphaTab.
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


def _staves_el(n, staves: int):
    """Which staff(s) this note appears on. <staff> sits after the dots and
    before <notations>; with two staves the same note is written once and
    drawn on both, which is how a score shows staff and tab together."""
    for s in range(1, staves + 1):
        SubElement(n, "staff").text = str(s)


def _note_el(measure, midi: int, dur: int, string: int, fret: int,
             voice: str = "1", ties: Sequence[str] = (), chord: bool = False,
             staves: int = 1):
    n = SubElement(measure, "note")
    if chord:                             # <chord> must be the first child
        SubElement(n, "chord")
    # guitar is written an octave above its sounding pitch so the treble
    # clef + clef-octave-change -1 reads the way every printed score does;
    # the tab staff ignores <pitch> and uses <fret>/<string> anyway
    _pitch_el(n, midi + (12 if staves == 2 else 0))
    SubElement(n, "duration").text = str(dur)
    for tie in ties:                    # <tie> sits between duration and voice
        SubElement(n, "tie", type=tie)
    SubElement(n, "voice").text = voice
    tname, dots = _TYPES[dur]
    SubElement(n, "type").text = tname
    for _ in range(dots):
        SubElement(n, "dot")
    _staves_el(n, staves)
    notations = SubElement(n, "notations")
    for tie in ties:                    # <tied> is what the tab staff draws
        SubElement(notations, "tied", type=tie)
    tech = SubElement(notations, "technical")
    SubElement(tech, "fret").text = str(fret)      # fret BEFORE string
    SubElement(tech, "string").text = str(string)
    return n


def _rest_el(measure, dur: int, voice: str = "1", staves: int = 1):
    n = SubElement(measure, "note")
    SubElement(n, "rest")
    SubElement(n, "duration").text = str(dur)
    SubElement(n, "voice").text = voice
    tname, dots = _TYPES[dur]
    SubElement(n, "type").text = tname
    for _ in range(dots):
        SubElement(n, "dot")
    _staves_el(n, staves)
    return n


def _rest_run(measure, start_slot: int, slots: int, voice: str, staves: int = 1):
    """Silence, split the same way notes are — a 5-eighth gap is not one rest
    any more than it is one note."""
    for g, dur in _segments(start_slot, slots):
        _rest_el(measure, dur, voice=voice, staves=staves)


def _tuning_name(midi: int):
    """MIDI pitch -> (step, alter, octave) for staff-tuning."""
    name = _STEP[midi % 12]
    step, alter = name[0], ("1" if len(name) > 1 else None)
    return step, alter, midi // 12 - 1


def build_musicxml(placed: Sequence[PlacedNote], tempo: float = 120.0,
                   title: str = "Resonote",
                   tuning: Optional[Sequence[int]] = None,
                   staves: int = 1) -> str:
    """Render an arrangement as MusicXML.

    ``staves=1`` (default) is tablature, which MuseScore renders correctly.

    ``staves=2`` asks for standard notation above the tab and is NOT working
    yet: MuseScore 4.7.5 imports it as an empty five-line staff with the fret
    numbers floating above the tab lines. Multi-staff MusicXML appears to want
    each staff written as a full pass through the measure separated by a
    measure-length <backup>, which is a different layout job from the voice
    grouping below. Kept here because the encoding is already correct for
    alphaTab; do not ship it as an engraving option until it is verified.
    """
    if staves not in (1, 2):
        raise ValueError(f"staves must be 1 (tab) or 2 (staff + tab), got {staves}")
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

    # measure 1 attributes: divisions, key, time, staves, clefs, string tunings
    m1 = SubElement(part, "measure", number="1")
    attrs = SubElement(m1, "attributes")
    SubElement(attrs, "divisions").text = str(DIVISIONS)
    key = SubElement(attrs, "key")
    SubElement(key, "fifths").text = "0"
    time = SubElement(attrs, "time")
    SubElement(time, "beats").text = "4"
    SubElement(time, "beat-type").text = "4"
    if staves == 2:
        # the published form of a fingerstyle score: pitch on top so the rhythm
        # and voicing are readable, tab underneath so the fingers are. One
        # voice serves both staves; each note says which staves to draw on.
        SubElement(attrs, "staves").text = str(staves)
        std = SubElement(attrs, "clef", number="1")
        SubElement(std, "sign").text = "G"
        SubElement(std, "line").text = "2"
        # guitar sounds an octave below the treble clef; the note is written
        # without that offset, so the clef has to say it
        SubElement(std, "clef-octave-change").text = "-1"
        tab = SubElement(attrs, "clef", number="2")
        SubElement(tab, "sign").text = "TAB"
        SubElement(tab, "line").text = "5"
    else:
        clef = SubElement(attrs, "clef")
        SubElement(clef, "sign").text = "TAB"
        SubElement(clef, "line").text = "5"
    if staves == 2:
        # staff 1 has to be declared as an ordinary five-line staff or a reader
        # inherits the tab setup below it and draws fret numbers on both
        std_details = SubElement(attrs, "staff-details", number="1")
        SubElement(std_details, "staff-lines").text = "5"
    details = SubElement(attrs, "staff-details",
                         **({"number": "2"} if staves == 2 else {}))
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

    # Group the notes into as few voices as possible: two notes that start and
    # stop together share a voice as a <chord>, and a voice only takes its next
    # note once the previous one has stopped, which keeps every voice a linear
    # timeline.
    #
    # Both extremes fail, differently. Cramming everything into voice 1 needs
    # <backup> to interleave a ringing bass with a moving melody, and alphaTab
    # refuses that ("cannot fill new beats into already filled area of voice").
    # One voice per string is machine-readable but shreds the page -- six
    # independent timelines is what made MuseScore lay this arrangement out
    # over 11 pages where the .gp5 fits in 4. Real guitar engraving sits in
    # between; fingerstyle lands at 2-4 voices.
    groups: List[List] = []              # per voice: [[start, dur, [notes]]]
    for (s, length, p) in items:
        for voice in groups:
            last = voice[-1]
            if last[0] == s and last[1] == length:
                last[2].append(p)                 # same span -> same chord
                break
            if last[0] + last[1] <= s:
                voice.append([s, length, [p]])    # this voice is free
                break
        else:
            groups.append([[s, length, [p]]])

    # Split each into writable note values. A note crossing the barline
    # continues as a tie rather than being cut -- cutting re-articulates
    # something that is supposed to sustain.
    voices: List[List] = []              # per voice: [[start, dur, notes, ties]]
    for voice in groups:
        segments = []
        for start, dur, notes in voice:
            pieces = list(_segments(start, dur))
            for i, (g, d) in enumerate(pieces):
                ties = (["stop"] if i else []) + \
                       (["start"] if i < len(pieces) - 1 else [])
                segments.append([g, d, notes, ties])
        voices.append(segments)

    # created up front so the measures stay in order while voices are written
    # into them afterwards
    measures = [m1] + [SubElement(part, "measure", number=str(i + 1))
                       for i in range(1, n_measures)]

    for vi, segments in enumerate(voices):
        v = str(vi + 1)
        for mi, m in enumerate(measures):
            bar_start = mi * SLOTS_PER_MEASURE
            bar_end = bar_start + SLOTS_PER_MEASURE
            cursor = 0
            for g, dur, notes, ties in segments:
                if not bar_start <= g < bar_end:
                    continue
                if g - bar_start > cursor:
                    _rest_run(m, bar_start + cursor, g - bar_start - cursor,
                              v, staves)
                for j, p in enumerate(notes):
                    _note_el(m, p.pitch, dur, p.string, p.fret, voice=v,
                             ties=ties, chord=bool(j), staves=staves)
                cursor = g - bar_start + dur
            # every voice fills every measure, so a reader never has to guess
            # whether a missing voice means silence or a missing file
            if cursor < SLOTS_PER_MEASURE:
                _rest_run(m, bar_start + cursor, SLOTS_PER_MEASURE - cursor,
                          v, staves)

    rough = tostring(score, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    lines = [ln for ln in pretty.splitlines() if ln.strip()]
    if lines[0].startswith("<?xml"):       # minidom adds its own declaration
        lines = lines[1:]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + "\n".join(lines) + "\n")


def write_musicxml(placed, out_path, tempo: float = 120.0,
                   title: str = "Resonote",
                   tuning: Optional[Sequence[int]] = None,
                   staves: int = 1) -> str:
    xml = build_musicxml(placed, tempo=tempo, title=title, tuning=tuning,
                         staves=staves)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_path
