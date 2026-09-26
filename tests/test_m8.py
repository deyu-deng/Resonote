"""M8 — GP5 export validity & same-string collision safety.

Regression guard for a bug found on the audio path: two notes sounding at the
same instant were fingered onto the SAME string. That is physically
unplayable *and* illegal in GP5 — the written file was corrupt and Guitar Pro
refused to open it (``ValueError: ... is not a valid ChordAlteration``).

Two layers now prevent it:
  * arrange._resolve_string_collisions moves the conflicting note to a free
    string (keeps the note -> no silent loss).
  * gp_export._emit_beat keeps at most one note per string per beat (so the
    file is always structurally valid, even if upstream regresses).
"""
import guitarpro as gp

from arrange import OPEN_MIDI, arrange, _resolve_string_collisions
from gp_export import to_gp5
from models import PlacedNote
from tests.make_sample import sample_notes
from analysis import analyze


def _all_beats(song):
    return [b for m in song.tracks[0].measures
            for v in m.voices for b in v.beats]


def _attacks(song):
    return [nt for b in _all_beats(song)
            for nt in b.notes if nt.type.name != "tie"]


# --- collision resolution (arrange layer) ------------------------------- #
def test_same_onset_same_string_is_resolved():
    # two notes attacking together, both forced onto string 5
    p1 = PlacedNote(pitch=OPEN_MIDI[5] + 15, onset=0.0, duration=0.5,
                    string=5, fret=15, finger=0, pluck="p", role="melody")
    p2 = PlacedNote(pitch=OPEN_MIDI[5] + 19, onset=0.0, duration=0.5,
                    string=5, fret=19, finger=0, pluck="p", role="melody")
    out = _resolve_string_collisions([p1, p2])
    strings = [p.string for p in out]
    assert len(set(strings)) == len(strings), f"still colliding: {strings}"
    # each note is still playable at its (string, fret)
    for p in out:
        assert 0 <= p.fret <= 19
        assert OPEN_MIDI[p.string] + p.fret == p.pitch


def test_non_colliding_notes_untouched():
    p1 = PlacedNote(pitch=OPEN_MIDI[5] + 2, onset=0.0, duration=0.5,
                    string=5, fret=2, finger=0, pluck="p", role="bass")
    p2 = PlacedNote(pitch=OPEN_MIDI[1] + 3, onset=0.0, duration=0.5,
                    string=1, fret=3, finger=0, pluck="a", role="melody")
    out = _resolve_string_collisions([p1, p2])
    assert (out[0].string, out[0].fret) == (5, 2)
    assert (out[1].string, out[1].fret) == (1, 3)


# --- export validity ---------------------------------------------------- #
def test_colliding_arrangement_still_exports_a_readable_file():
    p1 = PlacedNote(pitch=OPEN_MIDI[5] + 15, onset=0.0, duration=1.0,
                    string=5, fret=15, finger=0, pluck="p", role="melody")
    p2 = PlacedNote(pitch=OPEN_MIDI[5] + 19, onset=0.0, duration=1.0,
                    string=5, fret=19, finger=0, pluck="p", role="melody")
    placed = _resolve_string_collisions([p1, p2])
    to_gp5(placed, "/tmp/m8_collide.gp5", tempo=120)
    song = gp.parse("/tmp/m8_collide.gp5")          # must not raise
    for b in _all_beats(song):
        strs = [nt.string for nt in b.notes]
        assert len(set(strs)) == len(strs), "duplicate string in a beat"
    assert len(_attacks(song)) == 2                 # neither note was dropped


def test_demo_arrangement_roundtrip_is_valid():
    notes = sample_notes()
    an = analyze(notes)
    placed = arrange(notes, analysis=an)
    to_gp5(placed, "/tmp/m8_demo.gp5", tempo=an.tempo)
    song = gp.parse("/tmp/m8_demo.gp5")
    for b in _all_beats(song):
        strs = [nt.string for nt in b.notes]
        assert len(set(strs)) == len(strs)
    assert len(_attacks(song)) > 0


def test_every_measure_tiles_to_a_full_bar():
    notes = sample_notes()
    an = analyze(notes)
    placed = arrange(notes, analysis=an)
    to_gp5(placed, "/tmp/m8_tiles.gp5", tempo=an.tempo)
    song = gp.parse("/tmp/m8_tiles.gp5")
    # gp duration.value: 1=whole 2=half 4=quarter 8=eighth (eighth = 1 slot)
    slots = {1: 8, 2: 4, 4: 2, 8: 1}
    for m in song.tracks[0].measures:
        total = sum(slots.get(b.duration.value, 0) for b in m.voices[0].beats)
        assert total == 8, f"measure does not tile to 8 eighths: {total}"
