"""Transcription layer (L3).

Turns audio (or MIDI) into a list of :class:`Note` objects. Three backends:

* ``MidiFileBackend``    — reads an existing .mid/.midi via pretty_midi.
                           Always available (core dep). Used for tests & demo.
* ``YourMT3Backend``     — YourMT3+ multi-task AMT (2024 SOTA, F1≈0.85).
                           Lazy, requires GPU + model weights to be practical.
                           Tags notes with instrument/program info.
* ``BasicPitchBackend``  — Spotify Basic-Pitch (F1≈0.43). Lazy TF dep.
                           Used as fallback when YourMT3+ is unavailable.

The orchestrator :func:`transcribe_audio` runs the full L2→L3 pipeline:
separate the mix into stems (L2), transcribe each stem (L3), merge with
``instrument`` / ``track_id`` labels.
"""

from __future__ import annotations

import gc
import os
import tempfile
from abc import ABC, abstractmethod

from models import Note
from separate import (
    Stem, read_audio, write_stem_wav,
    get_separator, demucs_available,
)


# --------------------------------------------------------------------------- #
#  Backend interface                                                           #
# --------------------------------------------------------------------------- #

class TranscriptionBackend(ABC):
    """Transcribe audio/MIDI → list of Note."""

    @abstractmethod
    def transcribe(self, path: str) -> list[Note]:
        """Transcribe from a file path."""
        ...

    def transcribe_stem(self, stem: Stem) -> list[Note]:
        """Transcribe a separated stem (audio array).

        Default implementation: write stem to a temp wav, delegate to
        :meth:`transcribe`. Override for backends that accept arrays
        directly (e.g. YourMT3+).
        """
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="resonote_stem_")
        tmp.close()
        try:
            write_stem_wav(stem, tmp.name)
            return self.transcribe(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
#  MIDI backend (always available)                                             #
# --------------------------------------------------------------------------- #

class MidiFileBackend(TranscriptionBackend):
    """Read notes straight from a MIDI file (pretty_midi)."""

    def transcribe(self, path: str) -> list[Note]:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(path)
        notes = []
        for tid, instr in enumerate(midi.instruments):
            for n in instr.notes:
                notes.append(Note(
                    pitch=n.pitch,
                    onset=float(n.start),
                    duration=max(0.05, float(n.end - n.start)),
                    velocity=int(n.velocity),
                    instrument=instr.name or f"track_{tid}",
                    track_id=tid,
                ))
        notes.sort(key=lambda n: (n.onset, -n.pitch))
        return notes


# --------------------------------------------------------------------------- #
#  YourMT3+ backend (SOTA AMT, requires setup)                                #
# --------------------------------------------------------------------------- #

class YourMT3Backend(TranscriptionBackend):
    """YourMT3+ multi-task multitrack music transcription (2024 SOTA).

    Requires the ``yourmt3`` package + pretrained checkpoint.
    See: https://github.com/TEAMuP-dev/yourMT3

    Setup (one-time, on a GPU machine)::

        git clone https://github.com/TEAMuP-dev/yourMT3
        cd yourMT3 && pip install -e .
        # download pretrained checkpoint to a local path

    Then set ``checkpoint`` to the .ckpt path.
    """

    def __init__(self, checkpoint: str | None = None, device: str = "auto"):
        self._checkpoint = checkpoint or os.environ.get("YOURMT3_CKPT", "")
        self._device = device
        self._model = None  # lazy

    def _load(self):
        if self._model is not None:
            return
        if not self._checkpoint:
            raise RuntimeError(
                "YourMT3+ selected but no checkpoint configured.\n"
                "Set checkpoint= path or env YOURMT3_CKPT.\n"
                "Repo: https://github.com/TEAMuP-dev/yourMT3"
            )
        try:
            import torch
            from yourmt3.inference import load_model_from_checkpoint
        except ImportError as e:
            raise RuntimeError(
                "yourmt3 package not installed.\n"
                "git clone https://github.com/TEAMuP-dev/yourMT3 && "
                "pip install -e ."
            ) from e

        dev = self._device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = load_model_from_checkpoint(self._checkpoint, device=dev)
        self._dev = dev

    def transcribe_stem(self, stem: Stem) -> list[Note]:
        """Transcribe a stem's audio array directly (no temp file)."""
        self._load()
        import numpy as np

        # YourMT3+ expects float32 mono at the model's sample rate (16 kHz).
        audio = stem.audio
        target_sr = 16000
        if stem.sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(
                    audio, orig_sr=stem.sr, target_sr=target_sr)
            except ImportError:
                raise RuntimeError(
                    "librosa needed to resample for YourMT3+ (pip install librosa)")

        result = self._model.inference(audio, target_sr)
        # result is expected to be a list of note dicts:
        #   {pitch, onset, duration, velocity, instrument/program}
        notes = []
        for r in result:
            notes.append(Note(
                pitch=int(r["pitch"]),
                onset=float(r["onset"]),
                duration=max(0.05, float(r.get("duration", 0.3))),
                velocity=int(r.get("velocity", 100)),
                instrument=str(r.get("instrument", "")),
            ))
        notes.sort(key=lambda n: (n.onset, -n.pitch))
        return notes

    def transcribe(self, path: str) -> list[Note]:
        audio, sr = read_audio(path)
        return self.transcribe_stem(Stem(name="mix", audio=audio, sr=sr))


# --------------------------------------------------------------------------- #
#  Basic-Pitch backend (fallback)                                             #
# --------------------------------------------------------------------------- #

class BasicPitchBackend(TranscriptionBackend):
    """Transcribe audio with Basic-Pitch. Requires ``basic_pitch`` installed.

    pip install basic_pitch   # ~TF-sized; done on demand, not in base env
    """

    def __init__(self, model=None):
        self._model = model

    def transcribe(self, path: str) -> list[Note]:
        try:
            from basic_pitch.inference import predict
            from basic_pitch import note_creation
        except ImportError as e:
            raise RuntimeError(
                "Basic-Pitch is not installed. Run: pip install basic_pitch\n"
                "Then retry. (It pulls in TensorFlow, so it is an opt-in dep.)"
            ) from e

        # predict returns (model_output, audio_data, audio_rate)
        model_output, _, _ = predict(path)
        # basic_pitch >= 0.4 returns a (PrettyMIDI, note_events) tuple;
        # older versions returned the PrettyMIDI object directly.
        midi_out = note_creation.model_output_to_notes(
            model_output, onset_thresh=0.5, frame_thresh=0.3,
            infer_onsets=True, min_note_len=11, melodia_trick=True)
        midi_data = midi_out[0] if isinstance(midi_out, tuple) else midi_out

        notes = []
        for tid, instr in enumerate(midi_data.instruments):
            for n in instr.notes:
                notes.append(Note(
                    pitch=n.pitch,
                    onset=float(n.start),
                    duration=max(0.05, float(n.end - n.start)),
                    velocity=int(n.velocity),
                    track_id=tid,
                ))
        notes.sort(key=lambda n: (n.onset, -n.pitch))
        return notes


# --------------------------------------------------------------------------- #
#  Availability checks & factory                                              #
# --------------------------------------------------------------------------- #

def yourmt3_available() -> bool:
    try:
        import yourmt3  # noqa: F401
        return True
    except ImportError:
        return False


def basic_pitch_available() -> bool:
    try:
        import basic_pitch  # noqa: F401
        return True
    except ImportError:
        return False


def get_amt_backend(kind: str = "auto") -> TranscriptionBackend:
    """Pick an AMT backend for audio transcription.

    kind:
      "yourmt3"     — force YourMT3+ (raises at transcribe time if unconfigured)
      "basic-pitch" — force Basic-Pitch
      "auto"        — YourMT3+ if installed, else Basic-Pitch
    """
    if kind == "yourmt3":
        return YourMT3Backend()
    if kind == "basic-pitch":
        return BasicPitchBackend()
    if kind == "auto":
        if yourmt3_available():
            return YourMT3Backend()
        return BasicPitchBackend()
    raise ValueError(f"unknown AMT backend: {kind}")


# Backward-compatible alias (main.py previously called get_backend).
def get_backend(kind: str = "auto", audio_path: str = "") -> TranscriptionBackend:
    """Pick a backend by kind / filename extension.

    .mid/.midi → MidiFileBackend; audio → AMT backend (auto = YourMT3+ → Basic-Pitch).
    """
    if kind == "midi" or (audio_path and audio_path.lower().endswith((".mid", ".midi"))):
        return MidiFileBackend()
    if kind in ("yourmt3", "basic-pitch"):
        return get_amt_backend(kind)
    return get_amt_backend("auto")


# --------------------------------------------------------------------------- #
#  L2→L3 orchestrator                                                          #
# --------------------------------------------------------------------------- #

def transcribe_audio(
    path: str,
    use_separation: bool = True,
    amt: str = "auto",
    dump_dir: str | None = None,
    stem_filter: tuple[str, ...] | None = ("vocals", "bass", "other"),
) -> list[Note]:
    """Full L2→L3 pipeline: separate → transcribe each stem → merge.

    Parameters
    ----------
    path : str
        Audio file path (mp3/wav/flac/...).
    use_separation : bool
        If False, skip separation (passthrough — whole mix as one stem).
    amt : str
        AMT backend: "auto" / "yourmt3" / "basic-pitch".
    dump_dir : str | None
        If set, write separated stems (wav) + per-stem MIDI to this directory
        for debugging (``--dump-intermediate``).
    stem_filter : tuple[str, ...] | None
        Which demucs stems to transcribe. None = all. Default skips "drums"
        (drum transcriptions are noise for guitar arrangement). When
        passthrough (no separation), the single "mix" stem is always used.

    Returns
    -------
    list[Note]
        All notes merged & sorted by onset, with ``instrument`` / ``track_id``
        set per stem.
    """
    # L2: separate
    sep_kind = "auto" if use_separation else "passthrough"
    separator = get_separator(sep_kind)
    stems = separator.separate(path)
    del separator
    gc.collect()   # free the demucs model before the AMT backend loads:
                   # demucs (torch) and basic_pitch (TF) never need to be
                   # resident together, and holding both is what gets long
                   # songs killed mid-prediction on small machines

    # filter stems (skip drums for guitar arrangement)
    if stem_filter and len(stems) > 1:
        stems = [s for s in stems if s.name in stem_filter] or stems

    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    # L3: transcribe each stem
    backend = get_amt_backend(amt)
    all_notes: list[Note] = []
    for tid, stem in enumerate(stems):
        notes = backend.transcribe_stem(stem)
        for n in notes:
            if not n.instrument:
                n.instrument = stem.name
            n.track_id = tid
        all_notes.extend(notes)

        if dump_dir:
            stem_wav = os.path.join(dump_dir, f"stem_{tid}_{stem.name}.wav")
            try:
                write_stem_wav(stem, stem_wav)
                print(f"  [dump] stem {tid} ({stem.name}) → {stem_wav}")
            except Exception as e:
                print(f"  [dump] could not write stem {stem.name}: {e}")

    if dump_dir:
        notes_path = os.path.join(dump_dir, "transcribed_notes.tsv")
        with open(notes_path, "w") as f:
            f.write("onset\tduration\tpitch\tvelocity\tinstrument\ttrack_id\n")
            for n in all_notes:
                f.write(f"{n.onset:.3f}\t{n.duration:.3f}\t{n.pitch}\t"
                        f"{n.velocity}\t{n.instrument}\t{n.track_id}\n")
        print(f"  [dump] notes → {notes_path}")

    all_notes.sort(key=lambda n: (n.onset, -n.pitch))
    return all_notes
