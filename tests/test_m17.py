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
