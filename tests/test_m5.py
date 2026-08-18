"""M5 — audible preview export (MIDI + Karplus-Strong WAV).

Verifies that a PlacedNote arrangement can be turned into something you can
actually hear, and that the pitch mapping (OPEN_MIDI[string] + fret) survives
the round-trip exactly.
"""

import os
import tempfile
import wave

from collections import Counter

from models import PlacedNote, Note
from arrange import OPEN_MIDI
from midi_export import placed_to_midi, placed_to_wav

from tests.make_sample import sample_notes


def _placed_fixture():
    """A tiny hand-built arrangement with melody + bass + harmony."""
    return [
        PlacedNote(pitch=60, onset=0.0, duration=0.5, string=3, fret=5,
                   finger=1, pluck="i", role="melody"),
        PlacedNote(pitch=64, onset=0.5, duration=0.5, string=2, fret=9,
                   finger=1, pluck="m", role="melody"),
        PlacedNote(pitch=40, onset=0.0, duration=1.0, string=6, fret=0,
                   finger=0, pluck="p", role="bass"),
        PlacedNote(pitch=55, onset=0.0, duration=1.0, string=4, fret=5,
                   finger=2, pluck="i", role="harmony"),
    ]


class TestMidiExport:
    def test_pitch_mapping_is_open_plus_fret(self):
        # sanity: the docstring contract
        pn = _placed_fixture()[0]
        assert OPEN_MIDI[pn.string] + pn.fret == pn.pitch

    def test_midi_roundtrip_preserves_notes(self):
        placed = _placed_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "preview.mid")
            placed_to_midi(placed, out, tempo=120)
            assert os.path.exists(out)

            import pretty_midi
            mid = pretty_midi.PrettyMIDI(out)
            midi_notes = [(n.pitch, round(n.start, 3))
                          for inst in mid.instruments for n in inst.notes]
            placed_ref = [(OPEN_MIDI[p.string] + p.fret, round(p.onset, 3))
                          for p in placed]
            assert Counter(midi_notes) == Counter(placed_ref)

    def test_midi_from_real_pipeline(self):
        # exercise the full L4->L5 path so we don't just test the fixture
        from analysis import analyze
        from quantize import quantize_notes, quantize_placed
        from arrange import arrange
        notes = sample_notes()
        analysis = analyze(notes)
        qnotes = quantize_notes(notes, analysis, subdivision=2)
        placed = arrange(qnotes, analysis=analysis)
        placed = quantize_placed(placed, analysis, subdivision=2)
        assert placed

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "pipe.mid")
            placed_to_midi(placed, out, tempo=analysis.tempo)
            import pretty_midi
            mid = pretty_midi.PrettyMIDI(out)
            total = sum(len(inst.notes) for inst in mid.instruments)
            # ties/extensions can add entries, but we must not LOSE notes
            assert total >= len(placed)


class TestWavExport:
    def test_wav_is_valid_and_non_silent(self):
        placed = _placed_fixture()
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "preview.wav")
            placed_to_wav(placed, out, sr=22050)
            assert os.path.exists(out)

            with wave.open(out, "rb") as w:
                assert w.getnchannels() == 1
                assert w.getsampwidth() == 2
                assert w.getframerate() == 22050
                n = w.getnframes()
                assert n > 0
                frames = w.readframes(n)
            # must contain actual audio, not an all-zero file
            import numpy as np
            pcm = np.frombuffer(frames, dtype="<i2")
            assert pcm.max() > 0 or pcm.min() < 0
            # duration should cover the longest note + release tail
            assert n / 22050.0 >= 1.0

    def test_wav_handles_empty_list(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "empty.wav")
            placed_to_wav([], out)
            assert os.path.exists(out)
            with wave.open(out, "rb") as w:
                assert w.getnframes() > 0

    def test_wav_from_real_pipeline(self):
        from analysis import analyze
        from quantize import quantize_notes, quantize_placed
        from arrange import arrange
        notes = sample_notes()
        analysis = analyze(notes)
        qnotes = quantize_notes(notes, analysis, subdivision=2)
        placed = arrange(qnotes, analysis=analysis)
        placed = quantize_placed(placed, analysis, subdivision=2)

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "pipe.wav")
            placed_to_wav(placed, out, tempo=analysis.tempo)
            import numpy as np
            with wave.open(out, "rb") as w:
                n = w.getnframes()
                pcm = np.frombuffer(w.readframes(n), dtype="<i2")
            assert pcm.max() > 0  # audible, not silent
