"""Export arranged notes to a Guitar Pro 5 (.gp5) file via pyguitarpro.

Key facts learned from the library source:
  * Note.string is 1-indexed, 1 = high E (thinnest). Note.value = fret (0 = open).
  * A GP5 beat holds a list of notes -- several strings can sound together
    (a chord / strum). A note that rings across several beats is expressed with
    ``NoteType.tie`` continuation beats, NOT by giving one beat a long duration
    (that would eat the slots of every note that should sound in between).
  * The GP5 writer lays out beats inside each Measure purely by their
    durations -- it does NOT read beat.start. So every measure's beats must
    sum to the measure length (4/4 -> 3840 ticks == 8 eighth-note slots).

Why the old exporter was wrong (fixed here)
-------------------------------------------
The previous ``_build_timeline`` collapsed every grid slot to a SINGLE note
(``occ[start] = (lng, p)``). For a fingerstyle arrangement -- where a whole-bar
bass note and the moving melody share the same time span -- this (a) dropped
chord notes that start on the same slot, and (b) let one long note swallow the
slots of every shorter note underneath it. Net result: a 31-note arrangement
exported as 4 notes.

The rewrite builds the timeline from the *sounding set* at each grid slot, emits
one beat per maximal run of identical sounding sets, and uses ties for notes
that continue past a beat boundary. Nothing is dropped; every measure tiles to
exactly 8 eighth-note slots.
"""

import guitarpro as gp
from typing import List

from models import Note, PlacedNote

QUARTER = gp.Duration.quarterTime            # 960 ticks
GRID = QUARTER // 2                          # 480 ticks == one eighth note
SLOTS_PER_MEASURE = 8                        # 8 eighths == 4/4
MEASURE_TICKS = SLOTS_PER_MEASURE * GRID     # 3840

# slot length (in eighths) -> GP Duration.value
_VALUE_FOR_LEN = {1: 8, 2: 4, 4: 2, 8: 1}

_FINGER = {
    0: gp.Fingering.open,
    1: gp.Fingering.index,
    2: gp.Fingering.middle,
    3: gp.Fingering.annular,
    4: gp.Fingering.little,
}


def _split_powers(n: int) -> List[int]:
    """Decompose a slot count into powers of two (for clean GP durations)."""
    out: List[int] = []
    while n > 0:
        step = 1
        while step * 2 <= n:
            step *= 2
        out.append(step)
        n -= step
    return out


def _to_grid(placed, ticks_per_sec: float, anchor: float):
    """Map placed notes to (start_slot, len_slot, PlacedNote) on the eighth grid."""
    items = []
    for p in placed:
        start = (p.onset - anchor) * ticks_per_sec / GRID
        length = p.duration * ticks_per_sec / GRID
        s = max(0, int(round(start)))
        l = max(1, int(round(length)))
        items.append((s, l, p))
    return items


def _same_set(a, b) -> bool:
    return {id(x) for x in a} == {id(x) for x in b}


def _build_beats(items, total: int):
    """Walk grid slots; emit one beat per maximal run of an identical sounding
    set. A beat carries ``sounding`` (all notes ringing) and ``attacks`` (the
    notes that actually start on this beat; the rest are tie continuations)."""
    slot: list = [[] for _ in range(total)]
    for (s, l, p) in items:
        for g in range(s, min(s + l, total)):
            slot[g].append(p)

    beats = []
    s = 0
    while s < total:
        cur = slot[s]
        e = s + 1
        # extend while the sounding set is unchanged AND we stay in this measure
        while (e < total and _same_set(slot[e], cur)
               and (e % SLOTS_PER_MEASURE != 0)):
            e += 1
        seg_len = e - s
        attacks = {id(p) for (sp, lp, p) in items if sp == s}
        for plen in _split_powers(seg_len):
            beats.append((s, plen, cur, attacks))
            attacks = set()          # only the first sub-beat attacks
            s += plen
    return beats


def _emit_beat(voice, sounding, attacks, plen: int) -> None:
    beat = gp.Beat(voice)
    if not sounding:
        beat.status = gp.BeatStatus.empty
        beat.duration = gp.Duration(value=_VALUE_FOR_LEN.get(plen, 8))
        voice.beats.append(beat)
        return

    beat.status = gp.BeatStatus.normal
    beat.duration = gp.Duration(value=_VALUE_FOR_LEN.get(plen, 8))
    # A GP5 beat cannot hold two notes on the same string (physically
    # impossible and it corrupts the file -- Guitar Pro then refuses to open
    # it). Keep one note per string, preferring freshly attacked notes over
    # tie continuations.
    ordered = sorted(sounding, key=lambda p: id(p) not in attacks)
    used: set = set()
    for p in ordered:
        if p.string in used:
            continue
        used.add(p.string)
        note = gp.Note(beat)
        note.string = p.string
        note.value = p.fret
        note.velocity = gp.Velocities.default
        note.effect.leftHandFinger = _FINGER.get(p.finger, gp.Fingering.open)
        note.type = gp.NoteType.normal if id(p) in attacks else gp.NoteType.tie
        beat.notes.append(note)
    voice.beats.append(beat)


def to_gp5(placed, out_path, tempo: float = 120, title: str = "Resonote",
           anchor: float = 0.0):
    """Write placed notes to a .gp5 file. Returns out_path."""
    song = gp.Song()
    song.title = title
    song.tempo = tempo

    track = song.tracks[0]
    track.name = "Resonote Guitar"
    track.strings = [gp.GuitarString(n, v) for n, v in
                     [(1, 64), (2, 59), (3, 55), (4, 50), (5, 45), (6, 40)]]

    ticks_per_sec = QUARTER * tempo / 60.0
    items = _to_grid(placed, ticks_per_sec, anchor)

    total = max((s + l for s, l, _ in items), default=0)
    total = max(total, SLOTS_PER_MEASURE)
    total = ((total + SLOTS_PER_MEASURE - 1) // SLOTS_PER_MEASURE) * SLOTS_PER_MEASURE

    beats = _build_beats(items, total)
    n_measures = total // SLOTS_PER_MEASURE

    while len(track.measures) < n_measures:
        song.newMeasure()

    for mi in range(n_measures):
        g0 = mi * SLOTS_PER_MEASURE
        voice = track.measures[mi].voices[0]
        for (s, plen, sounding, attacks) in beats:
            if g0 <= s < g0 + SLOTS_PER_MEASURE:
                _emit_beat(voice, sounding, attacks, plen)

    gp.write(song, out_path)
    return out_path


def notes_to_gp5(notes, out_path, tempo: float = 120, title: str = "Resonote"):
    """Convenience: arrange Note list then export."""
    from arrange import arrange
    placed = arrange(notes)
    return to_gp5(placed, out_path, tempo=tempo, title=title)
