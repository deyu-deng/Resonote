"""M1 tests: models extension, separation layer, transcription factory.

These tests do NOT require heavy ML deps (demucs/torch/yourmt3/basic-pitch).
They verify the architecture, interfaces, and fallback chain.

Run:  python tests/test_m1.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pretty_midi
from models import Note, PlacedNote
from separate import (
    Stem, PassthroughSeparator, DemucsSeparator,
    get_separator, demucs_available,
)
from transcribe import (
    MidiFileBackend, get_backend, get_amt_backend,
    transcribe_audio, yourmt3_available, basic_pitch_available,
)


# --------------------------------------------------------------------------- #
#  helpers                                                                     #
# --------------------------------------------------------------------------- #

def _write_test_midi(path: str) -> str:
    """Write a tiny MIDI file (C major scale) for testing."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0, name="test_melody")
    for i, p in enumerate([60, 62, 64, 65, 67]):
        inst.notes.append(pretty_midi.Note(
            velocity=100, pitch=p, start=i * 0.5, end=i * 0.5 + 0.4))
    midi.instruments.append(inst)
    midi.write(path)
    return path


# --------------------------------------------------------------------------- #
#  1. Model extension                                                          #
# --------------------------------------------------------------------------- #

def test_note_fields():
    """Note has instrument/track_id with backward-compatible defaults."""
    n = Note(pitch=60, onset=0.0, duration=0.5)
    assert n.instrument == "", f"default instrument should be empty, got {n.instrument!r}"
    assert n.track_id == 0, f"default track_id should be 0, got {n.track_id}"

    n2 = Note(pitch=60, onset=0.0, duration=0.5, velocity=80,
              instrument="vocals", track_id=2)
    assert n2.instrument == "vocals"
    assert n2.track_id == 2
    print("OK: Note fields (instrument/track_id) work with defaults and explicit values")


# --------------------------------------------------------------------------- #
#  2. Separation layer                                                        #
# --------------------------------------------------------------------------- #

def test_stem_dataclass():
    """Stem holds name/audio/sr."""
    import numpy as np
    s = Stem(name="vocals", audio=np.zeros(100, dtype=np.float32), sr=44100)
    assert s.name == "vocals"
    assert s.sr == 44100
    assert len(s.audio) == 100
    print("OK: Stem dataclass")


def test_separator_factory():
    """get_separator returns correct type per kind."""
    # passthrough / none always work
    sep = get_separator("passthrough")
    assert isinstance(sep, PassthroughSeparator)
    sep2 = get_separator("none")
    assert isinstance(sep2, PassthroughSeparator)

    # auto: demucs if available, else passthrough
    sep3 = get_separator("auto")
    if demucs_available():
        assert isinstance(sep3, DemucsSeparator), "auto should pick demucs when available"
    else:
        assert isinstance(sep3, PassthroughSeparator), "auto should fall back to passthrough"

    # demucs forced but unavailable → should raise
    if not demucs_available():
        try:
            get_separator("demucs")
            assert False, "should have raised"
        except RuntimeError:
            pass  # expected

    print(f"OK: separator factory (demucs_available={demucs_available()})")


# --------------------------------------------------------------------------- #
#  3. Transcription backends                                                  #
# --------------------------------------------------------------------------- #

def test_midi_backend():
    """MidiFileBackend reads notes with instrument/track_id labels."""
    with tempfile.TemporaryDirectory() as d:
        mid = os.path.join(d, "test.mid")
        _write_test_midi(mid)
        notes = MidiFileBackend().transcribe(mid)

    assert len(notes) == 5, f"expected 5 notes, got {len(notes)}"
    assert notes[0].pitch == 60
    assert notes[0].instrument == "test_melody", f"instrument label wrong: {notes[0].instrument}"
    assert all(n.track_id == 0 for n in notes), "all from track 0"
    assert notes == sorted(notes, key=lambda n: (n.onset, -n.pitch)), "sorted by onset"
    print(f"OK: MidiFileBackend ({len(notes)} notes, instrument tagged)")


def test_backend_factory():
    """get_backend routes MIDI vs audio correctly."""
    # MIDI by extension
    b = get_backend("auto", "song.mid")
    assert isinstance(b, MidiFileBackend)

    # MIDI by kind
    b2 = get_backend("midi")
    assert isinstance(b2, MidiFileBackend)

    # audio auto → some AMT backend (may be BasicPitch or YourMT3)
    b3 = get_backend("auto", "song.mp3")
    assert not isinstance(b3, MidiFileBackend), "audio should not use MIDI backend"
    print(f"OK: backend factory (yourmt3={yourmt3_available()}, basic_pitch={basic_pitch_available()})")


def test_amt_factory():
    """get_amt_backend returns the right type per kind."""
    from transcribe import YourMT3Backend, BasicPitchBackend

    bp = get_amt_backend("basic-pitch")
    assert isinstance(bp, BasicPitchBackend)

    ym = get_amt_backend("yourmt3")
    assert isinstance(ym, YourMT3Backend)

    auto = get_amt_backend("auto")
    if yourmt3_available():
        assert isinstance(auto, YourMT3Backend)
    else:
        assert isinstance(auto, BasicPitchBackend)
    print("OK: AMT backend factory")


# --------------------------------------------------------------------------- #
#  4. transcribe_audio fallback (no heavy deps)                               #
# --------------------------------------------------------------------------- #

def _write_test_wav(path: str) -> str:
    """Write a 2-second 440 Hz sine wav (needs soundfile + numpy)."""
    import numpy as np
    import soundfile as sf
    sr = 22050
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def test_transcribe_audio_passthrough_no_amt():
    """transcribe_audio: audio I/O works (passthrough), AMT raises clearly
    when basic_pitch is not installed."""
    if basic_pitch_available() or yourmt3_available():
        print("SKIP: AMT backend available — skip no-AMT test")
        return

    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "test.wav")
        try:
            _write_test_wav(wav)
        except ImportError:
            print("SKIP: soundfile not installed — skip audio I/O test")
            return

        try:
            transcribe_audio(wav, use_separation=False, amt="basic-pitch")
            assert False, "should have raised (no basic_pitch)"
        except RuntimeError as e:
            assert "basic_pitch" in str(e).lower(), \
                f"error should mention basic_pitch: {e}"
    print("OK: transcribe_audio — I/O works, AMT raises clearly when missing")


# --------------------------------------------------------------------------- #
#  runner                                                                      #
# --------------------------------------------------------------------------- #

def run_all():
    tests = [
        test_note_fields,
        test_stem_dataclass,
        test_separator_factory,
        test_midi_backend,
        test_backend_factory,
        test_amt_factory,
        test_transcribe_audio_passthrough_no_amt,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print(f"\n{'='*50}")
    print(f"M1 tests: {passed}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    run_all()
