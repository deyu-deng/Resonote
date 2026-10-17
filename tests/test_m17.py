"""M17 -- the beat grid comes from the audio, not from our own transcription.

analysis.detect_beats() infers a tempo from the inter-onset intervals of the
notes L3 produced, so it can only ever agree with the transcription and cannot
tell where a bar begins. On a real song that was measured, not assumed: the
pipeline reported 94 BPM for 陶喆 Melody while the audio's onset-strength
autocorrelation peaks at 0.540 s (+0.42) and 1.080 s (+0.49) -- about 111 BPM
-- and is flat (-0.05) at the 0.645 s that 94 BPM would require.

These tests run without beat_this installed and without any audio file: what
is pinned is the wiring and the fallbacks, not the model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import beats
from analysis import analyze
from models import Note


def test_available_is_a_bool_and_detect_survives_it_being_false(monkeypatch):
    assert isinstance(beats.available(), bool)
    monkeypatch.setattr(beats, "available", lambda: False)
    assert beats.detect([0.0] * 10, 22050) is None


def test_a_broken_model_degrades_to_the_note_grid_instead_of_killing_the_run(
        monkeypatch):
    monkeypatch.setattr(beats, "available", lambda: True)
    monkeypatch.setattr(beats, "_MODEL", None)

    def boom(*a, **kw):
        raise RuntimeError("no checkpoint on this machine")

    import beat_this.inference as inf
    monkeypatch.setattr(inf, "Audio2Beats", boom)
    assert beats.detect([0.0] * 100, 22050) is None


def test_nonsense_beat_spacing_is_rejected(monkeypatch):
    """A tracker that locks onto 2x or 1/2 the beat is worse than no tracker:
    it silently halves or doubles every note value in the printed score."""
    monkeypatch.setattr(beats, "available", lambda: True)

    class Fake:
        def __call__(self, audio, sr):
            return ([0.0, 0.05, 0.10, 0.15], [0.0])     # 1200 BPM

    beats._MODEL = Fake()
    try:
        assert beats.detect([0.0] * 100, 22050) is None
    finally:
        beats._MODEL = None


def test_analyze_prefers_the_audio_grid_when_a_tracker_answers(monkeypatch):
    notes = [Note(pitch=64 + i % 5, onset=i * 0.5, duration=0.5)
             for i in range(24)]
    # note-based grid would say ~120 BPM off a 0.5 s spacing
    assert analyze(notes).grid_backend == "note-onsets"

    monkeypatch.setattr(beats, "detect", lambda audio, sr: {
        "beats": [i * 0.3 for i in range(40)],
        "downbeats": [i * 1.2 for i in range(10)],
        "tempo": 200.0})
    res = analyze(notes, audio=[0.0] * 1000, sr=22050)
    assert res.grid_backend == "beat_this"
    assert res.tempo == 200.0, "the waveform's answer must win"
    assert res.downbeats and len(res.beats) == 40


def test_analyze_falls_back_when_the_tracker_has_no_answer(monkeypatch):
    notes = [Note(pitch=64, onset=i * 0.5, duration=0.5) for i in range(24)]
    monkeypatch.setattr(beats, "detect", lambda audio, sr: None)
    res = analyze(notes, audio=[0.0] * 1000, sr=22050)
    assert res.grid_backend == "note-onsets"
    assert res.downbeats == []


def test_downbeats_default_to_empty_so_import_mode_is_unchanged():
    notes = [Note(pitch=60, onset=0.0, duration=1.0)]
    assert analyze(notes).downbeats == []


def test_demucs_random_shift_is_off_by_default():
    """demucs' `shifts` shifts the mix by a RANDOM 0-0.5 s and averages the
    result over N passes. At N=1 nothing is averaged, so the default we shipped
    made every run different: the same 30 s clip transcribed to 228 / 246 / 239
    notes on three runs, which is enough to drown out any improvement an eval
    could measure. Two full runs now produce byte-identical .gp5."""
    import inspect

    from separate import DemucsSeparator
    sig = inspect.signature(DemucsSeparator.__init__)
    assert sig.parameters["shifts"].default == 0, (
        "shifts>0 with no averaging reintroduces run-to-run nondeterminism; "
        "if you raise it, seed the RNG and say so here")


def _collide(*pairs):
    from models import PlacedNote
    from arrange import OPEN_MIDI
    return [PlacedNote(pitch=OPEN_MIDI[s] + f, onset=o, duration=0.5,
                       string=s, fret=f, finger=1, role="bass")
            for o, s, f in pairs]


def test_unresolvable_same_string_collision_drops_one_note_not_both():
    """Below the 5th string's open pitch (50) there is exactly ONE string that
    can sound the note, so two of them at one instant cannot both be played.
    Dropping must be by identity: these two are different notes, but a
    dataclass == would say otherwise."""
    from arrange import _resolve_string_collisions
    placed = _collide((1.0, 6, 2), (1.0, 6, 3))
    out = _resolve_string_collisions(placed)
    assert len(out) == 1, f"expected one note kept, got {len(out)}"
    assert out[0].pitch in (42, 43)
    assert {(round(p.onset, 4), p.string) for p in out} == {(1.0, 6)}


def test_identical_duplicate_notes_are_not_both_dropped():
    """The old code tested membership with dataclass equality, so a melody
    note and a bass note with the same fields -- which is exactly what
    assign_roles produces -- vanished together instead of one surviving."""
    from arrange import _resolve_string_collisions
    from models import PlacedNote
    from arrange import OPEN_MIDI
    a = PlacedNote(pitch=44, onset=2.0, duration=0.5, string=6, fret=4,
                   finger=1, role="melody")
    b = PlacedNote(pitch=44, onset=2.0, duration=0.5, string=6, fret=4,
                   finger=1, role="bass")
    out = _resolve_string_collisions([a, b])
    assert len(out) == 1
    assert out[0].role == "melody", "the tune survives, the doubling goes"


def test_a_movable_collision_is_moved_not_dropped():
    from arrange import _resolve_string_collisions, OPEN_MIDI
    # both pitches fit on several strings, so nothing has to be lost
    placed = _collide((3.0, 4, 5), (3.0, 4, 7))
    out = _resolve_string_collisions(placed)
    assert len(out) == 2, "a solvable collision must keep both notes"
    assert len({p.string for p in out}) == 2
    for p in out:
        assert OPEN_MIDI[p.string] + p.fret == p.pitch, "moving changed the pitch"


def test_role_assignment_follows_instrument_word_families():
    """The label depends on who produced it: demucs says 'vocals', MuScriptor
    says 'voice', the GM table would say 'Choir Aahs'. Exact-string matching
    silently sent all of that to the melody pile, which is how the arrangement
    ended up doubling the entire mix as a tune."""
    from arrangement import Role, assign_roles
    from models import Note
    notes = [
        Note(pitch=64, onset=0.0, duration=0.5, instrument="voice"),
        Note(pitch=62, onset=0.5, duration=0.5, instrument="vocals"),
        Note(pitch=40, onset=1.0, duration=0.5, instrument="bass"),
        Note(pitch=55, onset=1.5, duration=0.5, instrument="acoustic guitar"),
        Note(pitch=60, onset=2.0, duration=0.5, instrument="piano"),
        Note(pitch=57, onset=2.5, duration=0.5, instrument=""),      # bare MIDI
    ]
    arr = assign_roles(notes, None)
    assert [n.source for n in arr.melody] == ["voice", "vocals", ""]
    assert [n.source for n in arr.bass] == ["bass"]
    assert [n.source for n in arr.harmony] == ["acoustic guitar", "piano"]


def test_real_instrument_labels_suppress_the_derived_accompaniment():
    """Deriving bass+harmony from chords is for a bare melody line. When the
    transcription already heard a guitar, layering derived notes on top of it
    is what produced the same-string collisions. The guitar's own low end
    becomes the bass voice -- real notes, not invented ones."""
    from arrangement import assign_roles
    from models import Note
    from analysis import AnalysisResult, Chord
    notes = [Note(pitch=64, onset=0.0, duration=0.5, instrument="voice"),
             Note(pitch=52, onset=0.5, duration=0.5, instrument="acoustic guitar"),
             Note(pitch=60, onset=1.0, duration=0.5, instrument="acoustic guitar")]
    an = AnalysisResult(chords=[Chord(label="E", root=4, quality="",
                                      start=0.0, end=4.0, confidence=1.0)])
    arr = assign_roles(notes, an)
    invented = [n for n in arr.bass + arr.harmony if str(n.source).startswith("derived:")]
    assert not invented, f"chord-derived voices appeared over a real track: {invented}"
    assert [n.pitch for n in arr.bass] == [52], "the guitar's low end is the bass line"
    assert [n.pitch for n in arr.harmony] == [60]
    assert [n.pitch for n in arr.melody] == [64]
